"""Flower strategy configuration for federated averaging."""

from __future__ import annotations

from typing import Dict, List, Tuple

import flwr as fl


def fit_config(server_round: int) -> Dict[str, str]:
    """Broadcast one shared training configuration to all selected clients."""
    return {"server_round": str(server_round), "local_epochs": "1", "learning_rate": "0.001"}


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate client evaluation metrics weighted by each client's sample count."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if not total_examples:
        return {}
    return {
        key: sum(num_examples * values[key] for num_examples, values in metrics if key in values)
        / total_examples
        for key in {key for _, values in metrics for key in values}
    }


def build_strategy(min_clients: int = 3) -> fl.server.strategy.FedAvg:
    """Create FedAvg requiring all hospital clients for each training round."""
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
