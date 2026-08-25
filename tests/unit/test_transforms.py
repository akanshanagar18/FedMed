"""
Unit tests for data.datasets.transforms MONAI transform sequence.
"""

import pytest
import torch
from data.datasets.transforms import get_brats_transforms


def test_brats_train_transforms_construction():
    """Verify training MONAI Compose pipeline construction."""
    transforms = get_brats_transforms(mode="train", image_size=(32, 32, 32))
    assert transforms is not None
    # Check key transform names exist in Compose pipeline
    transform_names = [t.__class__.__name__ for t in transforms.transforms]
    assert "LoadImaged" in transform_names
    assert "EnsureChannelFirstd" in transform_names
    assert "Orientationd" in transform_names
    assert "Spacingd" in transform_names
    assert "NormalizeIntensityd" in transform_names
    assert "CropForegroundd" in transform_names
    assert "EnsureTyped" in transform_names


def test_brats_val_transforms_construction():
    """Verify validation MONAI Compose pipeline construction."""
    transforms = get_brats_transforms(mode="val", image_size=(32, 32, 32))
    assert transforms is not None
    transform_names = [t.__class__.__name__ for t in transforms.transforms]
    assert "LoadImaged" in transform_names
    assert "EnsureTyped" in transform_names
