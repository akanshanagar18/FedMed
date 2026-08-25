"""
Module: tests.unit.test_foundation_models

Purpose:
Unit test suite for vision foundation model adapters (SAM, MedSAM, DINOv2).
"""

import numpy as np
import pytest

from foundation.sam_adapter import SAMAdapter
from foundation.medsam_adapter import MedSAMAdapter
from foundation.dinov2_adapter import DINOv2Adapter


def test_sam_adapter():
    adapter = SAMAdapter(model_variant="vit_b", frozen_encoder=True)
    img = np.random.randn(1, 3, 256, 256).astype(np.float32)

    feats = adapter.extract_features(img)
    assert feats.shape == (1, 256, 64, 64)

    stats = adapter.get_parameter_stats()
    assert stats["trainable_percent"] < 10.0


def test_medsam_adapter():
    adapter = MedSAMAdapter(in_channels=4, out_channels=3, frozen_backbone=True)
    vol = np.random.randn(1, 4, 128, 128, 128).astype(np.float32)

    seg = adapter.predict_3d_segmentation(vol)
    assert seg.shape == (1, 3, 128, 128, 128)

    stats = adapter.get_parameter_stats()
    assert stats["frozen_backbone"] is True


def test_dinov2_adapter():
    adapter = DINOv2Adapter(variant="vit_b14", frozen_backbone=True)
    img = np.random.randn(1, 3, 224, 224).astype(np.float32)

    feats = adapter.extract_features(img)
    assert feats.shape == (1, 768)
