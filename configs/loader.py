"""
Module: configs.loader

Purpose:
Production-grade configuration loading, inheritance merging, and Pydantic validation engine.
Supports loading modular YAML files, overriding default parameters, and throwing human-readable validation errors.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
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


class StrategySpec(BaseModel):
    name: str = "FedAvg"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class FederatedConfig(BaseModel):
    strategy: Union[str, StrategySpec, Dict[str, Any]] = "FedAvg"
    num_rounds: int = Field(3, ge=1)
    min_clients: int = Field(2, ge=1)
    min_available_clients: int = Field(2, ge=1)
    learning_rate: float = Field(1e-4, gt=0)
    batch_size: int = Field(2, ge=1)
    local_epochs: int = Field(1, ge=1)
    optimizer: str = "adam"
    weight_decay: float = Field(1e-5, ge=0)
    proximal_mu: float = Field(0.01, ge=0)
    control_variate_lr: float = Field(1.0, ge=0)
    eta: float = Field(0.01, ge=0)
    beta_1: float = Field(0.9, ge=0)
    beta_2: float = Field(0.999, ge=0)
    tau: float = Field(1e-3, ge=0)
    gmf: float = Field(0.0, ge=0)
    alpha: float = Field(0.01, ge=0)
    num_shared_layers: int = Field(4, ge=1)
    num_local_layers: int = Field(2, ge=1)
    representation_layers: int = Field(4, ge=1)
    seed: int = 42

    def get_strategy_name(self) -> str:
        if isinstance(self.strategy, str):
            return self.strategy
        elif isinstance(self.strategy, StrategySpec):
            return self.strategy.name
        elif isinstance(self.strategy, dict):
            return self.strategy.get("name", "FedAvg")
        return "FedAvg"

    def get_strategy_parameters(self) -> Dict[str, Any]:
        params = {
            "min_fit_clients": self.min_clients,
            "min_available_clients": self.min_available_clients,
            "proximal_mu": self.proximal_mu,
            "eta": self.eta,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "tau": self.tau,
            "gmf": self.gmf,
            "alpha": self.alpha,
            "num_shared_layers": self.num_shared_layers,
            "num_local_layers": self.num_local_layers,
            "representation_layers": self.representation_layers,
        }
        if isinstance(self.strategy, StrategySpec):
            params.update(self.strategy.parameters)
        elif isinstance(self.strategy, dict) and "parameters" in self.strategy:
            params.update(self.strategy.get("parameters", {}))
        return params



class DataConfig(BaseModel):
    dataset_name: str = "BraTS2021"
    data_dir: str = "data/BraTS2021"
    cache_type: str = "persistent"
    cache_dir: str = ".cache/monai"
    num_workers: int = Field(0, ge=0)
    val_split: float = Field(0.2, ge=0.0, le=0.5)
    modalities: List[str] = Field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    partition_strategy: str = "IID"
    dirichlet_alpha: float = Field(0.5, gt=0)
    image_size: List[int] = Field(default_factory=lambda: [32, 32, 32])
    in_channels: int = 4
    out_channels: int = 3



class PrivacyConfig(BaseModel):
    dp_enabled: bool = False
    target_epsilon: float = Field(3.0, gt=0)
    target_delta: float = Field(1e-5, ge=0)
    max_grad_norm: float = Field(1.0, gt=0)
    he_enabled: bool = False
    scheme: str = "CKKS"
    poly_modulus_degree: int = 8192


class TlsConfig(BaseModel):
    enabled: bool = False
    verify_server: bool = True
    verify_client: bool = True
    cert_dir: str = "certs"
    ca_cert: str = "certs/ca.crt"
    server_cert: str = "certs/server.crt"
    server_key: str = "certs/server.key"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_interval: int = Field(10, ge=1)
    export_tensorboard: bool = False
    export_mlflow: bool = False


class CheckpointConfig(BaseModel):
    save_dir: str = "checkpoints"
    best_model_name: str = "best_model.pth"
    save_frequency: int = Field(1, ge=1)


class BenchmarkSubConfig(BaseModel):
    strategies: List[str] = Field(
        default_factory=lambda: [
            "FedAvg", "FedProx", "SCAFFOLD", "FedAdam", "FedYogi", "FedAdagrad", "FedNova", "FedDyn", "FedBN"
        ]
    )
    partitions: List[str] = Field(default_factory=lambda: ["IID", "NonIID(alpha=0.5)", "NonIID(alpha=0.2)"])
    privacy_modes: List[str] = Field(default_factory=lambda: ["none", "dp", "he", "dp_he"])
    client_counts: List[int] = Field(default_factory=lambda: [2, 3, 5])
    seeds: List[int] = Field(default_factory=lambda: [42, 123, 999])


class TrainingConfig(BaseModel):
    spatial_shape: List[int] = Field(default_factory=lambda: [128, 128, 128])
    loss: str = "DiceCELoss"
    loss_function: Optional[str] = None
    loss_params: Dict[str, Any] = Field(default_factory=dict)
    optimizer: str = "adam"
    learning_rate: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    momentum: Optional[float] = None
    batch_size: int = Field(2, ge=1)
    local_epochs: int = Field(1, ge=1)
    num_workers: int = Field(0, ge=0)
    device: Optional[str] = None


class AppConfig(BaseModel):
    """Canonical aggregated configuration contract for FedMed v2.0."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: Optional[TrainingConfig] = None
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    tls: TlsConfig = Field(default_factory=TlsConfig)
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


def normalize_config_dict(config_dict: Dict[str, Any], override_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Normalizes configuration keys across legacy, training, and canonical schemas.
    Ensures single source of truth for num_workers, batch_size, and optimizer settings.
    """
    normalized = config_dict.copy()
    override = override_dict or {}

    # Normalize 'training' section into 'data' and 'federated'
    if "training" in normalized and isinstance(normalized["training"], dict):
        t_dict = normalized["training"]
        data_dict = normalized.setdefault("data", {})
        fed_dict = normalized.setdefault("federated", {})

        t_override = override.get("training", {}) if isinstance(override.get("training"), dict) else {}
        d_override = override.get("data", {}) if isinstance(override.get("data"), dict) else {}
        f_override = override.get("federated", {}) if isinstance(override.get("federated"), dict) else {}

        # num_workers: single authoritative source of truth
        if "num_workers" in d_override and "num_workers" not in t_override:
            t_dict["num_workers"] = d_override["num_workers"]
            data_dict["num_workers"] = d_override["num_workers"]
        elif "num_workers" in t_dict:
            data_dict["num_workers"] = t_dict["num_workers"]
        elif "num_workers" in data_dict:
            t_dict["num_workers"] = data_dict["num_workers"]

        # batch_size
        if "batch_size" in f_override and "batch_size" not in t_override:
            t_dict["batch_size"] = f_override["batch_size"]
            fed_dict["batch_size"] = f_override["batch_size"]
        elif "batch_size" in t_dict:
            fed_dict["batch_size"] = t_dict["batch_size"]
        elif "batch_size" in fed_dict:
            t_dict["batch_size"] = fed_dict["batch_size"]

        # learning_rate
        if "learning_rate" in f_override and "learning_rate" not in t_override:
            t_dict["learning_rate"] = f_override["learning_rate"]
            fed_dict["learning_rate"] = f_override["learning_rate"]
        elif "learning_rate" in t_dict:
            fed_dict["learning_rate"] = t_dict["learning_rate"]

        # weight_decay
        if "weight_decay" in f_override and "weight_decay" not in t_override:
            t_dict["weight_decay"] = f_override["weight_decay"]
            fed_dict["weight_decay"] = f_override["weight_decay"]
        elif "weight_decay" in t_dict:
            fed_dict["weight_decay"] = t_dict["weight_decay"]

        # local_epochs
        if "local_epochs" in f_override and "local_epochs" not in t_override:
            t_dict["local_epochs"] = f_override["local_epochs"]
            fed_dict["local_epochs"] = f_override["local_epochs"]
        elif "local_epochs" in t_dict:
            fed_dict["local_epochs"] = t_dict["local_epochs"]

        # optimizer
        if "optimizer" in f_override and "optimizer" not in t_override:
            t_dict["optimizer"] = f_override["optimizer"]
            fed_dict["optimizer"] = f_override["optimizer"]
        elif "optimizer" in t_dict:
            fed_dict["optimizer"] = t_dict["optimizer"]

        # spatial_shape -> image_size
        if "image_size" in d_override and "spatial_shape" not in t_override:
            t_dict["spatial_shape"] = d_override["image_size"]
            data_dict["image_size"] = d_override["image_size"]
        elif "spatial_shape" in t_dict:
            data_dict["image_size"] = t_dict["spatial_shape"]

    # Normalize 'data' aliases
    if "data" in normalized and isinstance(normalized["data"], dict):
        d_dict = normalized["data"]
        if "raw_data_dir" in d_dict and "data_dir" not in d_dict:
            d_dict["data_dir"] = d_dict["raw_data_dir"]

    # Normalize 'federated' aliases
    if "federated" in normalized and isinstance(normalized["federated"], dict):
        f_dict = normalized["federated"]
        if "rounds" in f_dict and "num_rounds" not in f_dict:
            f_dict["num_rounds"] = f_dict["rounds"]

    return normalized


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

    override_dict = None
    if override_config_path:
        override_dict = load_raw_yaml(override_config_path)
        config_dict = deep_merge(config_dict, override_dict)

    config_dict = normalize_config_dict(config_dict, override_dict)

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
