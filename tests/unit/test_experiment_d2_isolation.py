"""
Unit and Integration Tests: tests/unit/test_experiment_d2_isolation.py

Purpose:
Verifies static configuration, optimizer instantiation, hyperparameter freezing,
artifact namespace isolation, accountant independence, and non-modification of
historical Experiment D and D1 artifacts for Experiment D2.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch
import yaml
from monai.networks.nets import UNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from privacy.dp_engine import compute_rdp_epsilon


def test_d2_configuration_static_invariants():
    config_path = PROJECT_ROOT / "configs" / "experiments" / "real_brats_dp_d2.yaml"
    assert config_path.exists(), "D2 config file does not exist"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 1. Optimizer is SGD
    assert config["training"]["optimizer"] == "SGD", f"Expected SGD, got {config['training']['optimizer']}"

    # 2. Momentum = 0.9
    assert config["training"]["momentum"] == 0.9, f"Expected momentum 0.9, got {config['training']['momentum']}"

    # 3. lr = 1e-4
    assert config["training"]["learning_rate"] == 1e-4, f"Expected lr 1e-4, got {config['training']['learning_rate']}"

    # 4. weight_decay = 1e-5
    assert config["training"]["weight_decay"] == 1e-5, f"Expected wd 1e-5, got {config['training']['weight_decay']}"

    # 5. C = 0.06
    assert config["privacy"]["max_grad_norm"] == 0.06, f"Expected C=0.06, got {config['privacy']['max_grad_norm']}"

    # 6. sigma = 0.87
    assert config["privacy"]["noise_multiplier"] == 0.87, f"Expected sigma=0.87, got {config['privacy']['noise_multiplier']}"

    # 7. Physical noise std = 0.0522
    physical_noise_std = round(config["privacy"]["noise_multiplier"] * config["privacy"]["max_grad_norm"], 4)
    assert physical_noise_std == 0.0522, f"Expected physical noise std 0.0522, got {physical_noise_std}"

    # 8. q = 1/236
    assert abs(config["privacy"]["sample_rate"] - (1.0 / 236.0)) < 1e-6, "Sampling rate mismatch"

    # 9. delta = 1e-5
    assert config["privacy"]["target_delta"] == 1e-5, "Delta mismatch"

    # 13. Dedicated D2 namespaces
    assert config["checkpoint"]["best_checkpoint_path"] == "checkpoints/fedavg_dp_d2/best.pt"
    assert config["checkpoint"]["latest_checkpoint_path"] == "checkpoints/fedavg_dp_d2/latest.pt"
    assert config["reports"]["report_path"] == "reports/real_brats2024/fedavg_dp_d2.json"
    assert config["reports"]["history_path"] == "reports/real_brats2024/fedavg_dp_d2_history.json"


def test_d2_model_parameter_count_and_optimizer_instantiation():
    m_cfg = {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
        "dropout": 0.0,
    }
    model = UNet(**m_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 16. Parameter count = 4810074
    assert total_params == 4810074, f"Parameter count mismatch: {total_params} != 4810074"

    # Instantiate SGD optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
        momentum=0.9,
        weight_decay=1e-5,
    )
    assert isinstance(optimizer, torch.optim.SGD), "Optimizer instance is not torch.optim.SGD"
    assert optimizer.defaults["momentum"] == 0.9, "Optimizer default momentum is not 0.9"
    assert optimizer.defaults["lr"] == 1e-4, "Optimizer default lr is not 1e-4"
    assert optimizer.defaults["weight_decay"] == 1e-5, "Optimizer default weight_decay is not 1e-5"


def test_accountant_optimizer_independence():
    # 10. Privacy accountant computation is mathematically invariant to optimizer choice
    eps_d = compute_rdp_epsilon(steps=4720, noise_multiplier=0.87, target_delta=1e-5, sample_rate=1.0 / 236.0)
    assert round(eps_d, 4) == 2.8934, f"Expected epsilon 2.8934, got {eps_d}"

    eps_stage_b = compute_rdp_epsilon(steps=708, noise_multiplier=0.87, target_delta=1e-5, sample_rate=1.0 / 236.0)
    assert round(eps_stage_b, 4) == 1.9636, f"Expected epsilon 1.9636, got {eps_stage_b}"


def test_experiment_d_and_d1_artifacts_immutability():
    # 11. Experiment D artifacts untouched
    d_ckpt = PROJECT_ROOT / "checkpoints" / "fedavg_dp_real" / "best.pt"
    d_manifest = PROJECT_ROOT / "reports" / "real_brats2024" / "dp_experiment_manifest.json"
    with open(d_manifest) as f:
        d_m = json.load(f)
    assert hashlib.sha256(d_ckpt.read_bytes()).hexdigest() == d_m["best_checkpoint_sha256"]

    # 12. Experiment D1 artifacts untouched
    d1_report = PROJECT_ROOT / "reports" / "real_brats2024" / "fedavg_dp_d1.json"
    d1_history = PROJECT_ROOT / "reports" / "real_brats2024" / "fedavg_dp_d1_history.json"
    with open(d1_report) as f:
        d1_r = json.load(f)
    with open(d1_history) as f:
        d1_h = json.load(f)
    assert d1_r["status"] == "PARTIAL_3_ROUNDS"
    assert len(d1_h) == 3
    assert d1_h[-1]["val_loss"] == 0.9883
    assert d1_h[-1]["mean_dice"] == 0.0205
