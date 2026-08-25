"""
Module: tests.unit.test_peft_lora

Purpose:
Unit test suite for PEFT & LoRA modules.
"""

import numpy as np
import pytest

from peft.lora import LoRALayer
from peft.adapters import BottleneckAdapter, PEFTManager


def test_lora_layer_forward_and_params():
    layer = LoRALayer(in_features=128, out_features=256, r=8, lora_alpha=16.0)
    x = np.random.randn(2, 128).astype(np.float32)

    out = layer.forward(x)
    assert out.shape == (2, 256)

    stats = layer.get_parameter_counts()
    assert stats["trainable_parameters"] == (8 * 128 + 256 * 8)


def test_bottleneck_adapter_and_peft_manager():
    adapter = BottleneckAdapter(hidden_dim=256, bottleneck_dim=32)
    x = np.random.randn(2, 256).astype(np.float32)

    out = adapter.forward(x)
    assert out.shape == (2, 256)

    manager = PEFTManager(total_model_params=50_000_000, lora_r=8)
    p_stats = manager.compute_peft_stats()
    assert p_stats["trainable_percent"] < 5.0
