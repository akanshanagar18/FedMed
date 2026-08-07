"""
Integration test for DP-enabled training pipeline (model/trainer.py & privacy/dp_engine.py).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from model.trainer import train_one_epoch
from model.unet3d import UNet3D
from privacy.dp_engine import DifferentialPrivacyEngine


def test_dp_training_epoch_execution():
    """Verify train_one_epoch executes Forward -> Backward -> Clip -> Noise -> Step without crashing."""
    model = UNet3D(in_channels=1, out_channels=3)
    from monai.losses import DiceCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = DiceCELoss(sigmoid=True)

    dp_engine = DifferentialPrivacyEngine(
        model=model,
        optimizer=optimizer,
        target_epsilon=3.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.5,
    )

    dummy_images = torch.randn(4, 1, 32, 32, 32)
    dummy_masks = torch.zeros(4, 3, 32, 32, 32)

    dataset = TensorDataset(dummy_images, dummy_masks)

    def custom_collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        msks = torch.stack([b[1] for b in batch])
        return {"image": imgs, "label": msks}

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)

    res = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        dp_engine=dp_engine,
    )

    assert "training_loss" in res
    assert "dice_score" in res
    assert dp_engine.steps == 2
    budget = dp_engine.get_privacy_budget()
    assert budget["epsilon"] >= 0.0
