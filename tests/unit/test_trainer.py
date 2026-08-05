"""
Unit tests for model.trainer and UNet3D PyTorch execution
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.unet3d import UNet3D
from model.trainer import train_one_epoch


@pytest.mark.unit
def test_unet3d_forward_pass():
    """Verify UNet3D model instantiation and 3D tensor forward pass."""
    model = UNet3D()
    # Batch size 1, 4 channels, spatial dimensions 32x32x32
    dummy_input = torch.randn(1, 4, 32, 32, 32)
    output = model(dummy_input)

    # Output shape should match (Batch=1, Channels=3, D=32, H=32, W=32)
    assert output.shape == (1, 3, 32, 32, 32)


@pytest.mark.unit
def test_train_one_epoch_execution():
    """Verify train_one_epoch loop updates model and computes metrics."""
    model = UNet3D()
    spatial_size = 32
    num_samples = 2

    images = torch.randn(num_samples, 4, spatial_size, spatial_size, spatial_size)
    masks = torch.zeros(num_samples, 3, spatial_size, spatial_size, spatial_size)

    dataset = TensorDataset(images, masks)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, mask = self.ds[idx]
            return {"image": img, "mask": mask}

    dataloader = DataLoader(DictDataset(dataset), batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
    )

    assert "training_loss" in metrics
    assert "dice_score" in metrics
    assert isinstance(metrics["training_loss"], float)
    assert isinstance(metrics["dice_score"], float)
    assert metrics["training_loss"] >= 0.0
    assert 0.0 <= metrics["dice_score"] <= 1.0
