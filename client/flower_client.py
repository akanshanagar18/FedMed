"""
Module: client.flower_client

Purpose:
Flower NumPyClient implementation for FedMed hospital nodes.
Handles local training on MRI data and communicates weight updates
to the central Flower server via gRPC.

Usage:
    python -m client.flower_client --server 127.0.0.1:8080 --hospital-id hospital_a
"""

import argparse
import logging
from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common import NDArrays, Scalar

from model.unet3d import UNet3D
from model.trainer import train_one_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_synthetic_dataloader(batch_size: int = 1, num_samples: int = 4, spatial_size: int = 32) -> DataLoader:
    """
    Creates a DataLoader with synthetic random tensors for MVP demo.
    Replaces real BraTS dataset — no download needed.
    """
    images = torch.randn(num_samples, 4, spatial_size, spatial_size, spatial_size)
    # Masks: one-hot encoded (num_samples, 3, D, H, W)
    masks_int = torch.randint(0, 3, (num_samples, 1, spatial_size, spatial_size, spatial_size))
    masks_onehot = torch.zeros(num_samples, 3, spatial_size, spatial_size, spatial_size)
    masks_onehot.scatter_(1, masks_int, 1)

    dataset = TensorDataset(images, masks_onehot)

    # Wrap to return dict format expected by trainer
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, tensor_dataset):
            self.tensor_dataset = tensor_dataset

        def __len__(self):
            return len(self.tensor_dataset)

        def __getitem__(self, idx):
            img, mask = self.tensor_dataset[idx]
            return {"image": img, "mask": mask}

    return DataLoader(DictDataset(dataset), batch_size=batch_size, shuffle=True)


class FedMedClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for a FedMed hospital node.

    Each hospital trains a local UNet3D model on its private data,
    then sends weight updates to the central server for aggregation.
    """

    def __init__(self, hospital_id: str, device: str = "cpu", spatial_size: int = 32):
        self.hospital_id = hospital_id
        self.device = device

        # Initialize model
        self.model = UNet3D()
        self.model.to(self.device)

        # Create synthetic data for MVP
        self.dataloader = get_synthetic_dataloader(batch_size=1, num_samples=4, spatial_size=spatial_size)

        # Training components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()

        logger.info(f"[{self.hospital_id}] FedMedClient initialized on {self.device}")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return model weights as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model weights from a list of numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train locally for one epoch, then return updated weights.
        """
        self.set_parameters(parameters)

        # Train one epoch
        metrics = train_one_epoch(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
        )

        logger.info(
            f"[{self.hospital_id}] Round training complete — "
            f"loss: {metrics['training_loss']:.4f}, dice: {metrics['dice_score']:.4f}"
        )

        return (
            self.get_parameters(config={}),
            len(self.dataloader.dataset),
            {
                "training_loss": float(metrics["training_loss"]),
                "dice_score": float(metrics["dice_score"]),
                "hospital_id": self.hospital_id,
            },
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model (simplified: run one forward pass on synthetic data).
        """
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)
                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)

        avg_loss = total_loss / max(num_samples, 1)
        return avg_loss, num_samples, {"hospital_id": self.hospital_id}


def start_client(server_address: str = "127.0.0.1:8080", hospital_id: str = "hospital_a"):
    """Start a Flower client connecting to the given server."""
    client = FedMedClient(hospital_id=hospital_id, device="cpu", spatial_size=32)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedMed Hospital Client")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--hospital-id", type=str, default="hospital_a", help="Unique hospital identifier")
    args = parser.parse_args()

    start_client(server_address=args.server, hospital_id=args.hospital_id)
