"""
Unit tests for Phase 9.2 Real Training Configuration.
Verifies resolution specifications (128³ vs 32³), loss function parameterization (1.0 / 0.2),
sequential client execution, test split immutability, and label semantics integrity.
"""

import json
from pathlib import Path
import pytest
import yaml
import torch
from monai.losses import DiceCELoss

from data.canonical_adapter import DatasetVersion, LabelCanonicalizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_canonical_real_experiment_configuration():
    config_path = PROJECT_ROOT / "configs" / "experiments" / "canonical_real_brats_gli_2024.yaml"
    assert config_path.exists(), f"Configuration file {config_path} does not exist"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Spatial Resolution
    assert cfg["spatial_resolutions"]["real_experiment"] == [128, 128, 128]
    assert cfg["spatial_resolutions"]["development_ci"] == [32, 32, 32]
    assert cfg["training"]["spatial_shape"] == [128, 128, 128]

    # 2. Loss Formulation
    assert cfg["training"]["loss"] == "DiceCELoss"
    assert cfg["training"]["loss_params"]["sigmoid"] is True
    assert cfg["training"]["loss_params"]["lambda_dice"] == 1.0
    assert cfg["training"]["loss_params"]["lambda_ce"] == 0.2

    # 3. Client Execution
    assert cfg["federated"]["client_execution"] == "sequential"
    assert len(cfg["federated"]["hospitals"]) == 4


def test_monai_loss_weights_instantiation():
    loss_fn = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)
    assert loss_fn.lambda_dice == 1.0
    assert loss_fn.lambda_ce == 0.2

    x = torch.randn(1, 3, 16, 16, 16)
    y = torch.randint(0, 2, (1, 3, 16, 16, 16)).float()
    loss = loss_fn(x, y)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_test_split_immutability():
    split_path = PROJECT_ROOT / "reports" / "real_brats2024" / "dataset_split.json"
    assert split_path.exists()

    with open(split_path, "r") as f:
        split = json.load(f)

    assert split["counts"]["total"] == 1350
    assert split["counts"]["train"] == 944
    assert split["counts"]["validation"] == 202
    assert split["counts"]["test"] == 204
    assert split["split_hash"] == "d0358ca42d4bf510624bbe7e86c3e4bc1e25a6174521b1280c0001eb2c66005c"
    assert split["disjoint_check"]["train_test_overlap"] == 0
    assert split["disjoint_check"]["val_test_overlap"] == 0


def test_label_semantics_integrity():
    import numpy as np

    seg = np.zeros((16, 16, 16), dtype=np.int16)
    seg[0:2, 0:2, 0:2] = 1  # NETC
    seg[2:4, 2:4, 2:4] = 2  # SNFH
    seg[4:6, 4:6, 4:6] = 3  # ET
    seg[6:8, 6:8, 6:8] = 4  # RC

    targets, summary = LabelCanonicalizer.build_canonical_targets(seg, version=DatasetVersion.BRATS_GLI_2024)

    # TC = 1 | 3
    assert np.array_equal(targets[0] > 0, (seg == 1) | (seg == 3))
    # WT = 1 | 2 | 3
    assert np.array_equal(targets[1] > 0, (seg == 1) | (seg == 2) | (seg == 3))
    # ET = 3
    assert np.array_equal(targets[2] > 0, (seg == 3))
    # RC (4) excluded from targets but preserved in summary
    assert summary["RC_voxels"] == 8
    assert summary["ET_voxels"] == 8
