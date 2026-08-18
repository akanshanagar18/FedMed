"""
Unit Tests: tests/unit/test_segmentation_metrics.py

Purpose:
Rigorous unit tests for independent Dice Similarity Coefficient and IoU (Jaccard) implementations.
Verifies all boundary conditions on tiny deterministic synthetic tensors:
  1. Perfect prediction
  2. Zero overlap
  3. Partial overlap (exact analytical quantities)
  4. Empty prediction (all zeros)
  5. Empty ground truth (all zeros)
  6. Both empty (empty background agreement)
  7. Multi-channel 3D volume (TC, WT, ET)
  8. PyTorch Tensor and NumPy array input interoperability
"""

import numpy as np
import pytest
import torch

from evaluation.metrics import compute_dice, compute_iou, evaluate_segmentation_suite


def test_perfect_prediction():
    # 4x4x4 cube with 8 active voxels
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True

    d = compute_dice(mask, mask)
    i = compute_iou(mask, mask)

    assert d["mean_dice"] == 1.0
    assert i["mean_iou"] == 1.0


def test_zero_overlap():
    p = np.zeros((4, 4, 4), dtype=bool)
    g = np.zeros((4, 4, 4), dtype=bool)
    p[0:2, :, :] = True
    g[2:4, :, :] = True

    d = compute_dice(p, g)
    i = compute_iou(p, g)

    assert d["mean_dice"] == 0.0
    assert i["mean_iou"] == 0.0


def test_partial_overlap_exact_quantities():
    # p has 4 voxels, g has 4 voxels, intersection is 2 voxels
    # |P| = 4, |G| = 4, |P ∩ G| = 2
    # Dice = 2 * 2 / (4 + 4) = 4/8 = 0.5
    # Union = 4 + 4 - 2 = 6
    # IoU  = 2 / 6 = 1/3 ≈ 0.333333
    p = np.zeros((4, 4, 4), dtype=bool)
    g = np.zeros((4, 4, 4), dtype=bool)

    p[0, 0, 0:4] = True  # voxels (0,0,0), (0,0,1), (0,0,2), (0,0,3)
    g[0, 0, 2:6] = True  # voxels (0,0,2), (0,0,3), (0,0,4), (0,0,5) -> 4 voxels in (4,4,6)

    p_full = np.zeros((4, 4, 6), dtype=bool)
    g_full = np.zeros((4, 4, 6), dtype=bool)
    p_full[0, 0, 0:4] = True
    g_full[0, 0, 2:6] = True

    d = compute_dice(p_full, g_full)
    i = compute_iou(p_full, g_full)

    assert pytest.approx(d["mean_dice"], abs=1e-5) == 0.5
    assert pytest.approx(i["mean_iou"], abs=1e-5) == 1.0 / 3.0


def test_empty_prediction():
    p = np.zeros((4, 4, 4), dtype=bool)
    g = np.ones((4, 4, 4), dtype=bool)

    d = compute_dice(p, g)
    i = compute_iou(p, g)

    assert d["mean_dice"] == 0.0
    assert i["mean_iou"] == 0.0


def test_empty_ground_truth():
    p = np.ones((4, 4, 4), dtype=bool)
    g = np.zeros((4, 4, 4), dtype=bool)

    d = compute_dice(p, g)
    i = compute_iou(p, g)

    assert d["mean_dice"] == 0.0
    assert i["mean_iou"] == 0.0


def test_both_empty():
    p = np.zeros((4, 4, 4), dtype=bool)
    g = np.zeros((4, 4, 4), dtype=bool)

    d = compute_dice(p, g)
    i = compute_iou(p, g)

    assert d["mean_dice"] == 1.0
    assert i["mean_iou"] == 1.0


def test_multi_channel_3d_evaluation():
    # 3 channels: TC, WT, ET
    # Channel 0: Perfect (Dice=1.0, IoU=1.0)
    # Channel 1: Zero overlap (Dice=0.0, IoU=0.0)
    # Channel 2: Partial overlap (Dice=0.5, IoU=0.333333)
    p = np.zeros((3, 4, 4, 6), dtype=bool)
    g = np.zeros((3, 4, 4, 6), dtype=bool)

    # Ch 0: Perfect
    p[0, 1:3, 1:3, 1:3] = True
    g[0, 1:3, 1:3, 1:3] = True

    # Ch 1: Zero
    p[1, 0, :, :] = True
    g[1, 3, :, :] = True

    # Ch 2: Partial (2 intersection / 4 each)
    p[2, 0, 0, 0:4] = True
    g[2, 0, 0, 2:6] = True

    channel_names = ["TC", "WT", "ET"]
    metrics = evaluate_segmentation_suite(p, g, channel_names=channel_names)

    assert metrics["dice_TC"] == 1.0
    assert metrics["iou_TC"] == 1.0

    assert metrics["dice_WT"] == 0.0
    assert metrics["iou_WT"] == 0.0

    assert pytest.approx(metrics["dice_ET"], abs=1e-5) == 0.5
    assert pytest.approx(metrics["iou_ET"], abs=1e-5) == 1.0 / 3.0

    expected_mean_dice = (1.0 + 0.0 + 0.5) / 3.0
    expected_mean_iou = (1.0 + 0.0 + (1.0 / 3.0)) / 3.0

    assert pytest.approx(metrics["mean_dice"], abs=1e-5) == expected_mean_dice
    assert pytest.approx(metrics["mean_iou"], abs=1e-5) == expected_mean_iou


def test_pytorch_tensor_logits_input():
    # Tensor with raw logits (before sigmoid)
    p_logits = torch.tensor([-5.0, 5.0, 5.0, -5.0]).view(1, 1, 2, 2, 1)  # voxels 1 & 2 are >0 after sigmoid
    g_mask = torch.tensor([0, 1, 1, 0]).view(1, 1, 2, 2, 1)

    d = compute_dice(p_logits, g_mask)
    i = compute_iou(p_logits, g_mask)

    assert d["mean_dice"] == 1.0
    assert i["mean_iou"] == 1.0
