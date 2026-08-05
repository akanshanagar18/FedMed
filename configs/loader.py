"""
Module: configs.loader

Purpose:
Production-grade configuration loading, inheritance merging, and Pydantic validation engine.
Supports loading modular YAML files, overriding default parameters, and throwing human-readable validation errors.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigValidationError(Exception):
    """Custom exception raised when configuration validation fails."""
    pass


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    fl_server_address: str = "127.0.0.1:8080"
    api_url: str = "http://127.0.0.1:8000"
    cors_origins: List[str] = Field(default_factory=list)


class FederatedConfig(BaseModel):
    strategy: str = "FedAvg"
    num_rounds: int = Field(3, ge=1)
    min_clients: int = Field(2, ge=1)
    min_available_clients: int = Field(2, ge=1)
    learning_rate: float = Field(1e-4, gt=0)
    batch_size: int = Field(2, ge=1)
    local_epochs: int = Field(1, ge=1)
    optimizer: str = "adam"
    weight_decay: float = Field(1e-5, ge=0)
    proximal_mu: float = Field(0.01, ge=0)
    seed: int = 42


class DataConfig(BaseModel):
    dataset_name: str = "BraTS2021"
    partition_strategy: str = "IID"
    dirichlet_alpha: float = Field(0.5, gt=0)
    image_size: List[int] = Field(default_factory=lambda: [128, 128, 128])
    in_channels: int = 4
    out_channels: int = 3


class PrivacyConfig(BaseModel):
    dp_enabled: bool = False
    target_epsilon: float = Field(3.0, gt=0)
    target_delta: float = Field(1e-5, ge=0)
    max_grad_norm: float = Field(1.0, gt=0)
    he_enabled: bool = False
    scheme: str = "CKKS"


class LoggingConfig(BaseModel):
    log_level: str = "INFO"
    log_interval: int = Field(10, ge=1)
    export_tensorboard: bool = False
    export_mlflow: bool = False


class CheckpointConfig(BaseModel):
    save_dir: str = "checkpoints"
    best_model_name: str = "best_model.pth"
    save_frequency: int = Field(1, ge=1)


class BenchmarkSubConfig(BaseModel):
    strategies: List[str] = Field(default_factory=lambda: ["FedAvg", "FedProx"])
    partitions: List[str] = Field(default_factory=lambda: ["IID", "NonIID(alpha=0.5)", "NonIID(alpha=0.2)"])
    seeds: List[int] = Field(default_factory=lambda: [42, 123, 999])


class AppConfig(BaseModel):
    """Canonical aggregated configuration contract for FedMed v2.0."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    benchmark: BenchmarkSubConfig = Field(default_factory=BenchmarkSubConfig)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load dictionary from YAML file."""
    path = Path(yaml_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            content = yaml.safe_load(f) or {}
            return content
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"YAML Syntax Error in '{yaml_path}': {e}")


def load_config(
    override_config_path: Optional[str] = None,
    base_config_path: Optional[str] = None,
) -> AppConfig:
    """
    Loads base configuration (default.yaml) and applies recursive overrides
    from override_config_path if provided. Validates through Pydantic.
    """
    base_path = base_config_path or str(DEFAULT_CONFIG_PATH)
    config_dict = load_raw_yaml(base_path)

    if override_config_path:
        override_dict = load_raw_yaml(override_config_path)
        config_dict = deep_merge(config_dict, override_dict)

    try:
        validated_config = AppConfig(**config_dict)
        return validated_config
    except ValidationError as err:
        errors_summary = []
        for e in err.errors():
            loc = " -> ".join(str(item) for item in e["loc"])
            msg = e["msg"]
            errors_summary.append(f"  - [{loc}]: {msg}")
        formatted = "\n".join(errors_summary)
        raise ConfigValidationError(f"Configuration Validation Error:\n{formatted}") from err
