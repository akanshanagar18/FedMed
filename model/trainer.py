"""
Module: model.trainer

Purpose:
PyTorch training loop for 3D U-Net supporting Differential Privacy (Opacus).
Executes strictly ordered step pipeline: Forward -> Backward -> Clip -> Noise -> Step.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = "cpu",
    dp_engine: Optional[Any] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Trains the model for one epoch with optional Differential Privacy engine.

    Args:
        model: PyTorch model (UNet3D).
        dataloader: DataLoader yielding {"image": Tensor, "mask": Tensor}.
        optimizer: PyTorch optimizer.
        loss_fn: Loss function (e.g., DiceLoss).
        device: "cpu" or "cuda".
        dp_engine: Optional DifferentialPrivacyEngine instance.

    Returns:
        Dict with "training_loss" and "dice_score" (averaged over batches).
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch.get("label", batch.get("mask")).to(device)

        # Adapt channel dimension if model expected input channels differ from input batch
        expected_in_channels = None
        if hasattr(model, "encoder1") and hasattr(model.encoder1, "block"):
            expected_in_channels = model.encoder1.block[0].weight.shape[1]
        elif hasattr(model, "in_channels"):
            expected_in_channels = model.in_channels
        elif hasattr(model, "model") and hasattr(model.model, "in_channels"):
            expected_in_channels = model.model.in_channels

        if expected_in_channels == 4 and images.shape[1] == 1:
            images = images.repeat(1, 4, 1, 1, 1)
        elif expected_in_channels == 1 and images.shape[1] == 4:
            images = images[:, :1, ...]

        optimizer.zero_grad()
        predictions = model(images)



        # Ensure mask shape matches predictions for loss computation
        if masks.shape != predictions.shape:
            if masks.dim() == 4:
                masks = masks.unsqueeze(1)
            if masks.shape[1] == 1 and predictions.shape[1] > 1:
                num_classes = predictions.shape[1]
                masks_long = masks.long().squeeze(1)
                masks_onehot = torch.zeros_like(predictions)
                masks_onehot.scatter_(1, masks_long.unsqueeze(1), 1)
                masks = masks_onehot

        loss = loss_fn(predictions, masks)
        loss.backward()

        # Apply Differential Privacy Gradient Clipping & Noise Addition
        if dp_engine is not None:
            dp_engine.apply_gradient_clipping_and_noise(batch_size=images.size(0))

        # Apply SCAFFOLD Control Variate Gradient Correction: g_corr = g_i - c_i + c
        if kwargs.get("server_control_variate") is not None and kwargs.get("client_control_variate") is not None:
            s_c = kwargs["server_control_variate"]
            c_c = kwargs["client_control_variate"]
            for idx, p in enumerate(model.parameters()):
                if p.grad is not None and idx < len(s_c) and idx < len(c_c):
                    correction = torch.tensor(s_c[idx] - c_c[idx], device=p.device, dtype=p.grad.dtype)
                    p.grad.add_(correction)

        optimizer.step()

        total_loss += loss.item()

        # Compute Dice score
        with torch.no_grad():
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            intersection = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum()
            dice = (2.0 * intersection / (union + 1e-8)).item()
            total_dice += dice

        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)

    return {"training_loss": avg_loss, "dice_score": avg_dice}
