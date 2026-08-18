"""
Module: evaluation.metrics

Purpose:
Independent Medical Image Segmentation Metric Calculation Engine for FedMed.
Strictly decoupled implementations for Dice Similarity Coefficient (DSC) and Intersection over Union (IoU/Jaccard).
Does NOT analytically proxy IoU from Dice.
Inspects binary overlap directly from tensor/array predictions vs ground truth masks.
Supports multi-region sub-region evaluation (TC, WT, ET) with macro and per-channel reporting.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def to_numpy_binary(tensor_or_array: Union[torch.Tensor, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Converts tensor/array logits or probabilities to binary bool array."""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu()
        if arr.dtype == torch.bool:
            return arr.numpy()
        # If float and contains values outside [0, 1], apply sigmoid
        if arr.min() < 0.0 or arr.max() > 1.0:
            arr = torch.sigmoid(arr)
        return (arr > threshold).numpy().astype(bool)
    else:
        arr = np.asarray(tensor_or_array)
        if arr.dtype == bool:
            return arr
        if np.issubdtype(arr.dtype, np.floating) and (arr.min() < 0.0 or arr.max() > 1.0):
            # Sigmoid activation
            arr = 1.0 / (1.0 + np.exp(-arr))
        return (arr > threshold).astype(bool)


def compute_dice(
    prediction: Union[torch.Tensor, np.ndarray],
    ground_truth: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-8,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes Dice Similarity Coefficient independently:
        Dice = (2 * |P ∩ G|) / (|P| + |G| + eps)

    Args:
        prediction: Predicted logits or binary mask of shape (B, C, D, H, W) or (C, D, H, W) or (D, H, W)
        ground_truth: Target binary mask of matching shape
        threshold: Binarization threshold for logits/probs
        eps: Smoothing epsilon to prevent division by zero
        channel_names: Names for individual channels (e.g. ['TC', 'WT', 'ET'])

    Returns:
        Dict with 'mean_dice' (macro average) and per-channel scores (e.g. 'dice_TC', 'dice_WT', 'dice_ET')
    """
    p_b = to_numpy_binary(prediction, threshold=threshold)
    g_b = to_numpy_binary(ground_truth, threshold=0.5)

    if p_b.shape != g_b.shape:
        raise ValueError(f"Shape mismatch: prediction {p_b.shape} != ground truth {g_b.shape}")

    # Standardize to 2D channel-wise list of arrays
    if p_b.ndim == 3:
        # Single channel, single volume (D, H, W)
        p_channels = [p_b]
        g_channels = [g_b]
    elif p_b.ndim == 4:
        # (C, D, H, W)
        p_channels = [p_b[c] for c in range(p_b.shape[0])]
        g_channels = [g_b[c] for c in range(g_b.shape[0])]
    elif p_b.ndim == 5:
        # (B, C, D, H, W) -> flatten batch dimension per channel
        num_c = p_b.shape[1]
        p_channels = [p_b[:, c, ...].flatten() for c in range(num_c)]
        g_channels = [g_b[:, c, ...].flatten() for c in range(num_c)]
    else:
        raise ValueError(f"Unsupported array dimension: {p_b.ndim}")

    names = channel_names or ([f"channel_{c}" for c in range(len(p_channels))] if len(p_channels) > 1 else ["default"])
    if len(names) != len(p_channels):
        names = [f"channel_{c}" for c in range(len(p_channels))]

    results: Dict[str, float] = {}
    channel_scores: List[float] = []

    for name, p_c, g_c in zip(names, p_channels, g_channels):
        p_sum = float(np.sum(p_c))
        g_sum = float(np.sum(g_c))
        intersection = float(np.sum(p_c & g_c))

        if p_sum == 0 and g_sum == 0:
            # Both empty: perfect match on empty background
            d_val = 1.0
        elif p_sum == 0 or g_sum == 0:
            # One is empty, other is not: zero overlap
            d_val = 0.0
        else:
            d_val = float((2.0 * intersection) / (p_sum + g_sum + eps))

        results[f"dice_{name}"] = round(d_val, 6)
        channel_scores.append(d_val)

    results["mean_dice"] = round(float(np.mean(channel_scores)), 6)
    return results


def compute_iou(
    prediction: Union[torch.Tensor, np.ndarray],
    ground_truth: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-8,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes Intersection over Union (Jaccard Index) independently:
        IoU = |P ∩ G| / (|P ∪ G| + eps) = |P ∩ G| / (|P| + |G| - |P ∩ G| + eps)

    Args:
        prediction: Predicted logits or binary mask
        ground_truth: Target binary mask of matching shape
        threshold: Binarization threshold for logits/probs
        eps: Smoothing epsilon
        channel_names: Names for individual channels (e.g. ['TC', 'WT', 'ET'])

    Returns:
        Dict with 'mean_iou' (macro average) and per-channel scores (e.g. 'iou_TC', 'iou_WT', 'iou_ET')
    """
    p_b = to_numpy_binary(prediction, threshold=threshold)
    g_b = to_numpy_binary(ground_truth, threshold=0.5)

    if p_b.shape != g_b.shape:
        raise ValueError(f"Shape mismatch: prediction {p_b.shape} != ground truth {g_b.shape}")

    if p_b.ndim == 3:
        p_channels = [p_b]
        g_channels = [g_b]
    elif p_b.ndim == 4:
        p_channels = [p_b[c] for c in range(p_b.shape[0])]
        g_channels = [g_b[c] for c in range(g_b.shape[0])]
    elif p_b.ndim == 5:
        num_c = p_b.shape[1]
        p_channels = [p_b[:, c, ...].flatten() for c in range(num_c)]
        g_channels = [g_b[:, c, ...].flatten() for c in range(num_c)]
    else:
        raise ValueError(f"Unsupported array dimension: {p_b.ndim}")

    names = channel_names or ([f"channel_{c}" for c in range(len(p_channels))] if len(p_channels) > 1 else ["default"])
    if len(names) != len(p_channels):
        names = [f"channel_{c}" for c in range(len(p_channels))]

    results: Dict[str, float] = {}
    channel_scores: List[float] = []

    for name, p_c, g_c in zip(names, p_channels, g_channels):
        p_sum = float(np.sum(p_c))
        g_sum = float(np.sum(g_c))
        intersection = float(np.sum(p_c & g_c))
        union = float(p_sum + g_sum - intersection)

        if p_sum == 0 and g_sum == 0:
            # Both empty: perfect match on empty background
            i_val = 1.0
        elif p_sum == 0 or g_sum == 0 or union == 0:
            # One is empty: zero overlap
            i_val = 0.0
        else:
            i_val = float(intersection / (union + eps))

        results[f"iou_{name}"] = round(i_val, 6)
        channel_scores.append(i_val)

    results["mean_iou"] = round(float(np.mean(channel_scores)), 6)
    return results


def evaluate_segmentation_suite(
    prediction: Union[torch.Tensor, np.ndarray],
    ground_truth: Union[torch.Tensor, np.ndarray],
    channel_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluates both Dice and IoU independently and combines results into a single report.
    """
    c_names = channel_names or ["TC", "WT", "ET"]
    dice_res = compute_dice(prediction, ground_truth, channel_names=c_names)
    iou_res = compute_iou(prediction, ground_truth, channel_names=c_names)

    combined = {}
    combined.update(dice_res)
    combined.update(iou_res)
    return combined
