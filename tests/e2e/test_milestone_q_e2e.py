"""
Module: tests.e2e.test_milestone_q_e2e

Purpose:
End-to-End simulation test for Milestone Q Personalization, Continual Learning, Foundation Models, and Explainability.
"""

import pytest
import numpy as np

from server.strategies.registry import StrategyRegistry
from continual.ewc import EWC
from foundation.medsam_adapter import MedSAMAdapter
from peft.lora import LoRALayer
from explainability.gradcam import GradCAM3D


def test_milestone_q_e2e_pipeline():
    # 1. Personalized FL Strategies
    strategies = StrategyRegistry.list_strategies()
    for s in ["FedPer", "LG-FedAvg", "FedRep", "Per-FedAvg"]:
        assert any(registered.lower() == s.lower() for registered in strategies)

    # 2. Continual Learning EWC
    ewc = EWC(ewc_lambda=400.0)
    p0 = [np.array([1.0, 2.0], dtype=np.float32)]
    g0 = [np.array([0.1, 0.2], dtype=np.float32)]
    ewc.update_fisher_matrix(p0, g0)
    penalty = ewc.compute_penalty([np.array([1.1, 2.1], dtype=np.float32)])
    assert penalty > 0.0

    # 3. MedSAM Foundation Model & LoRA PEFT
    medsam = MedSAMAdapter()
    stats = medsam.get_parameter_stats()
    assert stats["trainable_percent"] < 5.0

    lora = LoRALayer(in_features=64, out_features=128, r=4)
    x = np.random.randn(1, 64).astype(np.float32)
    y = lora.forward(x)
    assert y.shape == (1, 128)

    # 4. Explainability 3D Grad-CAM
    cam = GradCAM3D()
    act = np.random.randn(1, 4, 8, 8, 8).astype(np.float32)
    grad = np.random.randn(1, 4, 8, 8, 8).astype(np.float32)
    heatmap = cam.generate_heatmap(act, grad)
    assert heatmap.shape == (1, 8, 8, 8)
