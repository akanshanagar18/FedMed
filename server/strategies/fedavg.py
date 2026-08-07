"""
Module: server.strategies.fedavg

Purpose:
Pure, framework-agnostic implementation of the Federated Averaging (FedAvg) algorithm.
Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from server.strategies.base import BaseStrategy, EvaluateResult, FitResult, NDArrays, StrategyMetadata
from server.strategies.registry import register_strategy


@register_strategy("FedAvg")
class FedAvg(BaseStrategy):
    """
    Standard Federated Averaging (FedAvg) strategy.
    Performs sample-weighted elementwise averaging of client parameter updates.
    """

    def __init__(
        self,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedAvg",
            version="1.0.0",
            description="Federated Averaging (McMahan et al., 2017) baseline aggregation strategy.",
            supported_features=["weighted_averaging", "iid_partitioning"],
            supported_config=["min_fit_clients", "min_available_clients"],
            research_reference="https://arxiv.org/abs/1602.05629",
            default_parameters={
                "min_fit_clients": 2,
                "min_available_clients": 2,
            },
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[NDArrays], Dict[str, Any]]:
        """
        Computes sample-weighted elementwise average of parameters across clients.
        """
        if not results:
            return None, {}

        # Calculate total sample count
        total_samples = sum(res.num_examples for res in results)
        if total_samples == 0:
            return None, {}

        # First client weights define the tensor shapes
        first_params = results[0].parameters
        if first_params is None:
            return None, {}

        # Initialize weighted accumulators
        weighted_weights: List[np.ndarray] = [
            np.zeros_like(layer, dtype=np.float64) for layer in first_params
        ]

        # Accumulate sample-weighted weights
        for res in results:
            weight = res.num_examples / total_samples
            for i, layer in enumerate(res.parameters):
                weighted_weights[i] += layer * weight

        # Cast back to original layer dtypes
        aggregated_params: NDArrays = [
            layer.astype(first_params[i].dtype) if isinstance(first_params[i], np.ndarray) else layer
            for i, layer in enumerate(weighted_weights)
        ]

        # Aggregate client metrics
        total_loss = 0.0
        total_dice = 0.0
        for res in results:
            num = res.num_examples
            total_loss += res.metrics.get("training_loss", 0.0) * num
            total_dice += res.metrics.get("dice_score", 0.0) * num

        metrics = {
            "training_loss": total_loss / total_samples,
            "dice_score": total_dice / total_samples,
            "num_clients": len(results),
            "total_samples": total_samples,
        }

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[EvaluateResult],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Computes sample-weighted average loss and dice score for validation.
        """
        if not results:
            return None, {}

        total_samples = sum(res.num_examples for res in results)
        if total_samples == 0:
            return None, {}

        weighted_loss = sum(res.loss * res.num_examples for res in results) / total_samples
        weighted_dice = sum(res.metrics.get("dice_score", 0.0) * res.num_examples for res in results) / total_samples

        metrics = {
            "loss": weighted_loss,
            "dice_score": weighted_dice,
            "total_samples": total_samples,
        }
        return weighted_loss, metrics
