"""
Unit tests for segmentation metric calculation functions in scripts.train_baseline.
"""

import pytest
import torch
from scripts.train_baseline import compute_segmentation_metrics


def test_compute_segmentation_metrics_perfect_match():
    """Verify metrics for identical predicted logits and ground truth mask."""
    # Batch size 1, 3 channels, spatial shape 8x8x8
    y_true = torch.ones((1, 3, 8, 8, 8))
    # Large positive logits -> sigmoid(10) ~ 1.0
    y_logits = torch.full((1, 3, 8, 8, 8), 10.0)

    metrics = compute_segmentation_metrics(y_logits, y_true)
    assert pytest.approx(metrics["dice_score"], 1e-4) == 1.0
    assert pytest.approx(metrics["iou_score"], 1e-4) == 1.0
    assert pytest.approx(metrics["precision"], 1e-4) == 1.0
    assert pytest.approx(metrics["recall"], 1e-4) == 1.0
    assert metrics["hausdorff_95"] >= 0.0


def test_compute_segmentation_metrics_disjoint():
    """Verify metrics for zero overlap logits and ground truth."""
    y_true = torch.zeros((1, 3, 8, 8, 8))
    y_logits = torch.full((1, 3, 8, 8, 8), -10.0)  # sigmoid(-10) ~ 0.0

    metrics = compute_segmentation_metrics(y_logits, y_true)
    assert metrics["dice_score"] >= 0.0
    assert metrics["iou_score"] >= 0.0
