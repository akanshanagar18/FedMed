"""
Module: tests.unit.test_explainability

Purpose:
Unit test suite for explainability modules (GradCAM3D, AttentionVisualizer, UncertaintyEstimator).
"""

import numpy as np
import pytest

from explainability.gradcam import GradCAM3D
from explainability.attention_maps import AttentionVisualizer
from explainability.uncertainty import UncertaintyEstimator


def test_gradcam_3d_heatmap():
    gradcam = GradCAM3D(target_layer_name="conv_final")
    activations = np.random.randn(1, 16, 8, 8, 8).astype(np.float32)
    gradients = np.random.randn(1, 16, 8, 8, 8).astype(np.float32)

    cam = gradcam.generate_heatmap(activations, gradients)
    assert cam.shape == (1, 8, 8, 8)
    assert np.min(cam) >= 0.0 and np.max(cam) <= 1.0


def test_attention_visualizer():
    vis = AttentionVisualizer(num_heads=12)
    weights = np.random.randn(1, 12, 64, 64).astype(np.float32)

    processed = vis.process_attention_weights(weights)
    assert processed.shape == (1, 64, 64)


def test_uncertainty_estimator():
    estimator = UncertaintyEstimator(mc_samples=5)
    mc_preds = np.random.uniform(0.0, 1.0, size=(5, 1, 3, 16, 16)).astype(np.float32)

    res = estimator.estimate_uncertainty(mc_preds)
    assert "mean_prediction" in res
    assert "variance_map" in res
    assert "entropy_map" in res
