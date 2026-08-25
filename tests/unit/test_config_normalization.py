"""
Unit tests for configuration normalization and effective runtime parameter resolution.
Verifies that training section overrides authoritatively map to data and federated configurations.
"""

import pytest
from configs.loader import load_config, normalize_config_dict, AppConfig
from client.flower_client import FedMedClient


@pytest.mark.unit
def test_real_brats_fedavg_config_normalization():
    """Verify loading real_brats_fedavg.yaml properly sets num_workers=0 and batch_size=1."""
    cfg = load_config("configs/experiments/real_brats_fedavg.yaml")
    assert cfg.data.num_workers == 0
    assert cfg.federated.batch_size == 1
    assert cfg.federated.learning_rate == 0.0001
    assert cfg.training is not None
    assert cfg.training.num_workers == 0
    assert cfg.training.batch_size == 1


@pytest.mark.unit
def test_default_config_preserves_baseline_defaults():
    """Verify loading default.yaml maintains default values."""
    cfg = load_config("configs/default.yaml")
    assert cfg.data.num_workers == 0
    assert cfg.federated.batch_size == 2
    assert cfg.federated.strategy == "FedAvg"


@pytest.mark.unit
def test_normalize_config_dict_custom_overrides():
    """Verify custom training overrides normalize correctly into data and federated sections."""
    raw = {
        "training": {
            "num_workers": 0,
            "batch_size": 4,
            "learning_rate": 0.0005,
            "spatial_shape": [64, 64, 64],
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "local_epochs": 2,
        },
        "federated": {
            "rounds": 10,
        },
        "data": {
            "raw_data_dir": "data/custom",
        }
    }
    normalized = normalize_config_dict(raw)
    assert normalized["data"]["num_workers"] == 0
    assert normalized["data"]["image_size"] == [64, 64, 64]
    assert normalized["data"]["data_dir"] == "data/custom"
    assert normalized["federated"]["batch_size"] == 4
    assert normalized["federated"]["learning_rate"] == 0.0005
    assert normalized["federated"]["num_rounds"] == 10
    assert normalized["federated"]["optimizer"] == "AdamW"


@pytest.mark.unit
def test_fedmed_client_uses_effective_config():
    """Verify FedMedClient uses normalized configuration parameters for DataLoader."""
    client = FedMedClient(
        hospital_id="hospital_alpha",
        config_path="configs/experiments/real_brats_fedavg.yaml",
        device="cpu",
    )
    assert client.config.data.num_workers == 0
    assert client.config.federated.batch_size == 1
    assert client.dataloader.num_workers == 0
    assert client.dataloader.batch_size == 1


@pytest.mark.unit
def test_all_simulation_configs_resolve_zero_workers():
    """Verify all standard simulation and experiment configs resolve data.num_workers == 0."""
    config_paths = [
        "configs/default.yaml",
        "configs/privacy.yaml",
        "configs/dp.yaml",
        "configs/tls.yaml",
        "configs/experiments/real_brats_fedavg.yaml",
        "configs/experiments/real_brats_fedprox.yaml",
        "configs/experiments/real_brats_dp.yaml",
        "configs/experiments/canonical_real_brats_gli_2024.yaml",
    ]
    for cp in config_paths:
        cfg = load_config(cp)
        assert cfg.data.num_workers == 0, f"Config {cp} must resolve data.num_workers == 0, got {cfg.data.num_workers}"


@pytest.mark.unit
def test_client_dataloader_zero_workers_on_default_config():
    """Verify FedMedClient initialized with default config creates DataLoader with num_workers == 0."""
    client = FedMedClient(
        hospital_id="hospital_beta",
        config_path="configs/default.yaml",
        device="cpu",
    )
    assert client.config.data.num_workers == 0
    assert client.dataloader.num_workers == 0
