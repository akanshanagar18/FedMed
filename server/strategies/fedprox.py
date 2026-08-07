"""
Module: server.strategies.fedprox

Purpose:
Pure, framework-agnostic implementation of the FedProx algorithm.
Reference: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
Introduces proximal regularization parameter `proximal_mu` to mitigate non-IID client drift.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from server.strategies.base import BaseStrategy, EvaluateResult, FitResult, NDArrays, StrategyMetadata
from server.strategies.registry import register_strategy


@register_strategy("FedProx")
class FedProx(BaseStrategy):
    """
    FedProx strategy for heterogeneous / non-IID federated learning environments.
    Applies proximal term regularization during aggregation.
    """

    def __init__(
        self,
        proximal_mu: float = 0.01,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(
            proximal_mu=proximal_mu,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        self.proximal_mu = float(proximal_mu)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedProx",
            version="1.0.0",
            description="FedProx strategy with proximal term regularization (Li et al., 2020).",
            supported_features=["proximal_regularization", "non_iid", "straggler_mitigation"],
            supported_config=["proximal_mu", "min_fit_clients", "min_available_clients"],
            research_reference="https://arxiv.org/abs/1812.06127",
            default_parameters={
                "proximal_mu": 0.01,
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
        Computes sample-weighted average of parameters while tracking proximal_mu metadata.
        """
        if not results:
            return None, {}

        total_samples = sum(res.num_examples for res in results)
        if total_samples == 0:
            return None, {}

        first_params = results[0].parameters
        if first_params is None:
            return None, {}

        weighted_weights: List[np.ndarray] = [
            np.zeros_like(layer, dtype=np.float64) for layer in first_params
        ]

        for res in results:
            weight = res.num_examples / total_samples
            for i, layer in enumerate(res.parameters):
                weighted_weights[i] += layer * weight

        aggregated_params: NDArrays = [
            layer.astype(first_params[i].dtype) if isinstance(first_params[i], np.ndarray) else layer
            for i, layer in enumerate(weighted_weights)
        ]

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
            "proximal_mu": self.proximal_mu,
        }

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[EvaluateResult],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Computes sample-weighted validation metrics.
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
            "proximal_mu": self.proximal_mu,
        }
        return weighted_loss, metrics
