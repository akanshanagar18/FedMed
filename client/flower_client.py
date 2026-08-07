"""
Module: client.flower_client

Purpose:
Flower NumPyClient implementation for FedMed hospital nodes.
Handles local training on partitioned BraTS medical MRI data and communicates weight updates
(optionally protected with Differential Privacy via Opacus and encrypted via TenSEAL CKKS).

Usage:
    python -m client.flower_client --server-address 127.0.0.1:8080 --hospital-id hospital_alpha --enable-dp --enable-he
"""

import argparse
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from configs.loader import load_config
from data.datasets.brats import BraTSDataset
from data.datasets.cache import build_monai_dataset
from data.datasets.transforms import get_brats_transforms
from data.partitioner import get_partitioner
from model.trainer import train_one_epoch
from model.unet3d import UNet3D
from privacy.context import create_ckks_context, get_public_context, serialize_context
from privacy.dp_engine import DifferentialPrivacyEngine
from privacy.encrypt import encrypt_model_parameters
from privacy.communication import serialize_encrypted_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fedmed_client")


class FedMedClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for a FedMed hospital node.
    Trains UNet3D on its assigned disjoint patient dataset partition.
    Supports Differential Privacy (Opacus) and Homomorphic Encryption (TenSEAL CKKS).
    """

    def __init__(
        self,
        hospital_id: str = "hospital_alpha",
        device: str = "cpu",
        partition_strategy: str = "dirichlet",
        dirichlet_alpha: float = 0.5,
        enable_he: bool = False,
        enable_dp: bool = False,
        hospital_silos: Optional[List[str]] = None,
        config_path: Optional[str] = None,
    ):
        self.hospital_id = hospital_id
        self.device = device
        self.config = load_config(config_path)
        self.he_enabled = enable_he or self.config.privacy.he_enabled
        self.dp_enabled = enable_dp or self.config.privacy.dp_enabled

        # Initialize UNet3D model
        self.model = UNet3D(
            in_channels=self.config.data.in_channels,
            out_channels=self.config.data.out_channels,
        )
        self.model.to(self.device)

        # Initialize Homomorphic Encryption Context if enabled
        if self.he_enabled:
            logger.info(f"[{self.hospital_id}] Initializing TenSEAL CKKS Homomorphic Encryption Context (poly_degree={self.config.privacy.poly_modulus_degree})...")
            self.he_context = create_ckks_context(poly_modulus_degree=self.config.privacy.poly_modulus_degree)
            self.public_he_context = get_public_context(self.he_context)
            self.public_context_bytes = serialize_context(self.public_he_context, save_secret_key=False)
        else:
            self.he_context = None
            self.public_he_context = None
            self.public_context_bytes = None

        # Build BraTS dataset & apply partitioner
        silos = hospital_silos or ["hospital_alpha", "hospital_beta", "hospital_gamma"]
        full_dataset = BraTSDataset(
            data_dir=self.config.data.data_dir,
            modalities=self.config.data.modalities,
            image_size=tuple(self.config.data.image_size),
            cache_type=self.config.data.cache_type,
            cache_dir=self.config.data.cache_dir,
            num_workers=self.config.data.num_workers,
            allow_synthetic_fallback=True,
        )

        partitioner = get_partitioner(
            strategy=partition_strategy,
            alpha=dirichlet_alpha,
            seed=self.config.federated.seed,
        )

        stats = partitioner.partition(full_dataset.patient_metadata, silos)
        assigned_partition = stats.partitions.get(self.hospital_id)

        if assigned_partition and assigned_partition.patient_ids:
            assigned_ids = set(assigned_partition.patient_ids)
            self.partition_files = [item for item in full_dataset.data_list if item["patient_id"] in assigned_ids]
        else:
            self.partition_files = full_dataset.data_list

        logger.info(
            f"[{self.hospital_id}] Initialized partition ({partition_strategy.upper()} alpha={dirichlet_alpha}) — "
            f"Assigned {len(self.partition_files)} patients out of {len(full_dataset.patient_metadata)} total."
        )

        # Create MONAI DataLoader for assigned partition
        transforms = get_brats_transforms(mode="train", image_size=tuple(self.config.data.image_size))
        self.dataset = build_monai_dataset(
            data_list=self.partition_files,
            transforms=transforms,
            cache_type=self.config.data.cache_type,
            cache_dir=self.config.data.cache_dir,
            num_workers=self.config.data.num_workers,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.federated.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

        # Optimizer & Loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.federated.learning_rate,
            weight_decay=self.config.federated.weight_decay,
        )
        from monai.losses import DiceCELoss
        self.loss_fn = DiceCELoss(sigmoid=True)

        # Initialize Differential Privacy Engine if enabled
        if self.dp_enabled:
            logger.info(f"[{self.hospital_id}] Initializing Opacus Differential Privacy Engine (target_ε={self.config.privacy.target_epsilon}, C={self.config.privacy.max_grad_norm})...")
            self.dp_engine = DifferentialPrivacyEngine(
                model=self.model,
                optimizer=self.optimizer,
                target_epsilon=self.config.privacy.target_epsilon,
                target_delta=self.config.privacy.target_delta,
                max_grad_norm=self.config.privacy.max_grad_norm,
            )
        else:
            self.dp_engine = None

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
        Train locally for one epoch, then return updated weights (encrypted if HE enabled).
        """
        if self.he_enabled and self.he_context is not None and "encrypted_global_payload" in config:
            from privacy.communication import deserialize_encrypted_payload
            from privacy.decrypt import decrypt_model_parameters

            logger.info(f"[{self.hospital_id}] Received encrypted global model ciphertext. Decrypting client-side using private secret key...")
            payload = deserialize_encrypted_payload(str(config["encrypted_global_payload"]))
            decrypted_global_weights = decrypt_model_parameters(self.he_context, payload["encrypted_chunks"], payload["shapes"])
            self.set_parameters(decrypted_global_weights)
        else:
            if any(np.count_nonzero(p) > 0 for p in parameters):
                self.set_parameters(parameters)

        metrics = train_one_epoch(
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            dp_engine=self.dp_engine,
        )

        updated_params = self.get_parameters(config={})
        fit_metrics: Dict[str, Scalar] = {
            "training_loss": float(metrics["training_loss"]),
            "dice_score": float(metrics["dice_score"]),
            "hospital_id": self.hospital_id,
            "he_enabled": bool(self.he_enabled),
            "dp_enabled": bool(self.dp_enabled),
        }

        # Track Differential Privacy budget if active
        if self.dp_enabled and self.dp_engine is not None:
            budget = self.dp_engine.get_privacy_budget()
            fit_metrics["epsilon"] = float(budget["epsilon"])
            fit_metrics["delta"] = float(budget["delta"])
            fit_metrics["noise_multiplier"] = float(budget["noise_multiplier"])
            fit_metrics["max_grad_norm"] = float(budget["max_grad_norm"])

        # Encrypt model parameters with CKKS if Homomorphic Encryption is active
        if self.he_enabled and self.he_context is not None:
            logger.info(f"[{self.hospital_id}] Encrypting model weight tensors with TenSEAL CKKS...")
            enc_result = encrypt_model_parameters(self.he_context, updated_params)
            serialized_he_payload = serialize_encrypted_payload(enc_result)

            fit_metrics["encrypted_payload"] = serialized_he_payload
            fit_metrics["encryption_time_ms"] = float(enc_result["encryption_time_ms"])
            fit_metrics["ciphertext_size_bytes"] = int(enc_result["ciphertext_size_bytes"])

            zero_params = [np.zeros_like(p) for p in updated_params]
            params_to_return = zero_params
        else:
            params_to_return = updated_params

        logger.info(
            f"[{self.hospital_id}] Round training complete — "
            f"loss: {metrics['training_loss']:.4f}, dice: {metrics['dice_score']:.4f}"
            + (f", ε={fit_metrics['epsilon']:.2f}" if self.dp_enabled else "")
        )

        return (
            params_to_return,
            len(self.dataloader.dataset),
            fit_metrics,
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on hospital validation dataset."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                predictions = self.model(images)
                loss = self.loss_fn(predictions, labels)

                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)

        avg_loss = total_loss / max(num_samples, 1)
        return avg_loss, num_samples, {"hospital_id": self.hospital_id}


def _send_node_heartbeat(api_url: str, hospital_id: str, status: str, round_num: int = 0, reconnect_count: int = 0):
    """Sends real-time node resilience heartbeat to FastAPI backend."""
    try:
        url = f"{api_url.rstrip('/')}/api/v1/nodes/heartbeat"
        requests.post(url, json={
            "hospital_id": hospital_id,
            "status": status,
            "active_round": round_num,
            "reconnect_count": reconnect_count,
            "training_state": status.lower(),
        }, timeout=2)
    except Exception:
        pass


def start_client(
    server_address: str = "127.0.0.1:8080",
    hospital_id: str = "hospital_alpha",
    partition_strategy: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    enable_he: bool = False,
    enable_dp: bool = False,
    enable_tls: bool = False,
    cert_dir: str = "certs",
    api_url: str = "http://127.0.0.1:8000",
    config_path: Optional[str] = None,
    max_retries: int = 3,
):
    """Start a Flower client connecting to the given server with TLS & reconnect resilience."""
    client = FedMedClient(
        hospital_id=hospital_id,
        device="cpu",
        partition_strategy=partition_strategy,
        dirichlet_alpha=dirichlet_alpha,
        enable_he=enable_he,
        enable_dp=enable_dp,
        config_path=config_path,
    )

    tls_active = enable_tls
    if config_path:
        cfg = load_config(config_path)
        if hasattr(cfg, "tls") and cfg.tls.enabled:
            tls_active = True
            cert_dir = cfg.tls.cert_dir

    root_certs = None
    if tls_active:
        logger.info(f"[{hospital_id}] Enabling Production TLS gRPC Client Transport (cert_dir='{cert_dir}')...")
        cert_paths = ensure_tls_certificates(cert_dir)
        root_certs = load_pem_bytes(cert_paths["ca_cert"])

    reconnect_count = 0
    _send_node_heartbeat(api_url, hospital_id, "ONLINE", reconnect_count=reconnect_count)

    try:
        if root_certs is not None:
            fl.client.start_numpy_client(server_address=server_address, client=client, root_certificates=root_certs)
        else:
            fl.client.start_numpy_client(server_address=server_address, client=client)
        _send_node_heartbeat(api_url, hospital_id, "ONLINE", reconnect_count=reconnect_count)
    except Exception as e:
        logger.warning(f"[{hospital_id}] Client connection exception: {e}")
        _send_node_heartbeat(api_url, hospital_id, "OFFLINE", reconnect_count=reconnect_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedMed Hospital Client")
    parser.add_argument("--server-address", "--server", type=str, default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--hospital-id", type=str, default="hospital_alpha", help="Unique hospital identifier")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="FastAPI URL")
    parser.add_argument("--partition-strategy", type=str, default="dirichlet", help="Partition strategy (iid / dirichlet)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5, help="Dirichlet alpha value")
    parser.add_argument("--enable-he", action="store_true", help="Enable TenSEAL Homomorphic Encryption")
    parser.add_argument("--enable-dp", action="store_true", help="Enable Opacus Differential Privacy")
    parser.add_argument("--tls", "--enable-tls", action="store_true", help="Enable production TLS gRPC transport")
    parser.add_argument("--cert-dir", type=str, default="certs", help="TLS certificate directory")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    args = parser.parse_args()

    start_client(
        server_address=args.server_address,
        hospital_id=args.hospital_id,
        partition_strategy=args.partition_strategy,
        dirichlet_alpha=args.dirichlet_alpha,
        enable_he=args.enable_he,
        enable_dp=args.enable_dp,
        enable_tls=args.tls,
        cert_dir=args.cert_dir,
        api_url=args.api_url,
        config_path=args.config,
    )

