"""
Unit tests for configs.loader (Modular Configuration Engine)
"""

import pytest
from pydantic import ValidationError
from configs.loader import load_config, AppConfig, ConfigValidationError, deep_merge


@pytest.mark.unit
def test_load_default_config():
    """Verify loading default.yaml produces valid AppConfig instance."""
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.server.port == 8000
    assert cfg.federated.strategy == "FedAvg"
    assert cfg.federated.num_rounds == 3
    assert cfg.data.dataset_name == "BraTS2021"


@pytest.mark.unit
def test_config_inheritance_override():
    """Verify loading fedprox.yaml properly overrides federated.strategy to FedProx."""
    cfg = load_config("configs/fedprox.yaml")
    assert cfg.federated.strategy == "FedProx"
    assert cfg.federated.proximal_mu == 0.01
    # Check baseline defaults preserved
    assert cfg.server.port == 8000
    assert cfg.data.dataset_name == "BraTS2021"


@pytest.mark.unit
def test_deep_merge_utility():
    """Test recursive dictionary merging."""
    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    override = {"nested": {"y": 99, "z": 30}}
    merged = deep_merge(base, override)
    assert merged["a"] == 1
    assert merged["nested"]["x"] == 10
    assert merged["nested"]["y"] == 99
    assert merged["nested"]["z"] == 30


@pytest.mark.unit
def test_config_validation_error_handling():
    """Verify Pydantic validation catches invalid parameter types/ranges."""
    with pytest.raises((ValidationError, ConfigValidationError)):
        AppConfig(server={"port": -5})
