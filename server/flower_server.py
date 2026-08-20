"""
Module: server.flower_server

Purpose:
Production Flower server orchestrator for FedMed v2.0.
Instantiates pluggable strategy implementations via StrategyRegistry and wraps them
with FlowerStrategyAdapter for telemetry, gRPC transport, production TLS encryption, and fault tolerance.
"""

import argparse
import logging
from typing import Any, Dict, Optional, Tuple, Union

import flwr as fl
import requests

from configs.loader import AppConfig, load_config
from privacy.tls_cert_gen import ensure_tls_certificates, load_pem_bytes
from server.strategies import FlowerStrategyAdapter, StrategyRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Backward compatibility alias
class MetricsReporterStrategy(FlowerStrategyAdapter):
    """
    Backward-compatible strategy wrapper matching FedMed v1.0 interface.
    """
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000",
        experiment_id: str = "default",
        strategy_name: str = "FedAvg",
        **kwargs,
    ):
        base_strat = StrategyRegistry.create(strategy_name, **kwargs)
        super().__init__(strategy=base_strat, api_url=api_url, experiment_id=experiment_id)


def _register_experiment_start(
    api_url: str,
    experiment_id: str,
    num_rounds: int,
    min_clients: int,
    strategy_name: str = "FedAvg",
):
    """Registers experiment metadata and transitions status to RUNNING."""
    payload = {
        "experiment_id": experiment_id,
        "name": f"Federated Run ({experiment_id})",
        "description": f"Cross-silo 3D UNet Brain Tumor MRI Segmentation using {strategy_name}",
        "status": "running",
        "strategy_name": strategy_name,
        "num_clients": min_clients,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "local_epochs": 1,
        "num_rounds": num_rounds,
        "seed": 42,
        "dp_enabled": False,
        "he_enabled": False,
        "dataset_name": "BraTS2021",
        "partition_strategy": "IID",
    }
    try:
        url = f"{api_url.rstrip('/')}/api/v1/experiments"
        requests.post(url, json=payload, timeout=5)
        logger.info(f"Registered experiment '{experiment_id}' ({strategy_name}) as RUNNING")
    except Exception as e:
        logger.warning(f"Could not register experiment start: {e}")


def _register_experiment_complete(api_url: str, experiment_id: str, best_dice: float, best_round: int):
    """Updates experiment status to COMPLETED upon FL training completion."""
    payload = {
        "experiment_id": experiment_id,
        "name": f"Federated Run ({experiment_id})",
        "status": "completed",
        "best_dice_score": best_dice,
        "best_round": best_round,
    }
    try:
        url = f"{api_url.rstrip('/')}/api/v1/experiments"
        requests.post(url, json=payload, timeout=5)
        logger.info(f"Updated experiment '{experiment_id}' status to COMPLETED")
    except Exception as e:
        logger.warning(f"Could not update experiment completion: {e}")


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 3,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    api_url: str = "http://127.0.0.1:8000",
    experiment_id: str = "default",
    enable_tls: bool = False,
    cert_dir: str = "certs",
    config_path: Optional[str] = None,
    strategy_name: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
):
    """Start the Flower server using the Strategy Registry engine."""
    params: Dict[str, Any] = {
        "min_fit_clients": min_fit_clients,
        "min_available_clients": min_available_clients,
    }

    tls_active = enable_tls

    if config_path:
        app_cfg: AppConfig = load_config(config_path)
        num_rounds = app_cfg.federated.num_rounds
        min_fit_clients = app_cfg.federated.min_clients
        min_available_clients = app_cfg.federated.min_available_clients
        resolved_strategy_name = app_cfg.federated.get_strategy_name()
        params.update(app_cfg.federated.get_strategy_parameters())
        if hasattr(app_cfg, "tls") and app_cfg.tls.enabled:
            tls_active = True
            cert_dir = app_cfg.tls.cert_dir
    else:
        resolved_strategy_name = strategy_name or "FedAvg"

    if strategy_params:
        params.update(strategy_params)

    # 2. Instantiate pure Strategy via Registry
    pure_strategy = StrategyRegistry.create(resolved_strategy_name, **params)
    logger.info(f"Instantiated strategy '{pure_strategy.get_metadata().name}' via StrategyRegistry")

    # 3. Wrap pure Strategy with FlowerStrategyAdapter
    adapter = FlowerStrategyAdapter(
        strategy=pure_strategy,
        api_url=api_url,
        experiment_id=experiment_id,
    )

    _register_experiment_start(
        api_url,
        experiment_id,
        num_rounds,
        min_fit_clients,
        strategy_name=resolved_strategy_name,
    )

    # Resolve TLS Certificates if TLS enabled
    certificates_tuple = None
    if tls_active:
        logger.info(f"Enabling Production TLS gRPC Server Transport (cert_dir='{cert_dir}')...")
        cert_paths = ensure_tls_certificates(cert_dir)
        ca_bytes = load_pem_bytes(cert_paths["ca_cert"])
        server_cert_bytes = load_pem_bytes(cert_paths["server_cert"])
        server_key_bytes = load_pem_bytes(cert_paths["server_key"])
        certificates_tuple = (ca_bytes, server_cert_bytes, server_key_bytes)

    mode_str = "TLS-SECURED mTLS gRPC" if tls_active else "INSECURE gRPC"
    logger.info(
        f"Starting Flower server on {server_address} [{mode_str}] using [{resolved_strategy_name}] "
        f"for {num_rounds} rounds (min_fit={min_fit_clients}, min_available={min_available_clients})"
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=adapter,
        certificates=certificates_tuple,
    )

    if adapter.successful_rounds < num_rounds:
        logger.error(
            f"CRITICAL: FL Server completed only {adapter.successful_rounds}/{num_rounds} required aggregation rounds!"
        )
        sys.exit(1)

    _register_experiment_complete(api_url, experiment_id, adapter.best_dice_score, adapter.best_round)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedMed Flower Server")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients per round")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="Dashboard API URL")
    parser.add_argument("--experiment-id", type=str, default="default", help="Experiment identifier")
    parser.add_argument("--tls", "--enable-tls", action="store_true", help="Enable production TLS gRPC transport")
    parser.add_argument("--cert-dir", type=str, default="certs", help="TLS certificate directory")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--strategy", type=str, default=None, help="Strategy name (e.g. FedAvg, FedProx)")
    args = parser.parse_args()

    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        api_url=args.api_url,
        experiment_id=args.experiment_id,
        enable_tls=args.tls,
        cert_dir=args.cert_dir,
        config_path=args.config,
        strategy_name=args.strategy,
    )
