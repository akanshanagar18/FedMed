"""
Module: server.strategies.lgfedavg

Purpose:
LG-FedAvg (Think Locally, Act Globally: Federated Learning with Local and Global Representations, Liang et al., 2020).
Keeps lower representation layers local to hospital clients while aggregating upper prediction head layers globally.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("lgfedavg")


@register_strategy("LG-FedAvg")
class LGFedAvg(BaseStrategy):
    """
    LG-FedAvg strategy implementing local representations with global prediction head aggregation.
    """

    def __init__(
        self,
        num_local_layers: int = 2,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_local_layers = int(num_local_layers)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="LG-FedAvg",
            version="1.0.0",
            description="Think Locally, Act Globally: Federated Learning with Local and Global Representations (Liang et al., 2020)",
            supported_features=["local_representation", "global_head_aggregation", "personalized_fl"],
            supported_config=["num_local_layers"],
            research_reference="Think Locally, Act Globally: Federated Learning with Local and Global Representations, Liang et al., 2020 (arXiv:2001.01523)",
            default_parameters={
                "algorithm_name": "LG-FedAvg",
                "authors": "Liang et al.",
                "year": 2020,
                "num_local_layers": self.num_local_layers,
            },
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            logger.warning(f"[LG-FedAvg] Insufficient fit results ({len(results)}/{self.min_fit_clients})")
            return None, {}

        total_examples = sum(r.num_examples for r in results)
        weights_results = []

        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            weights_results.append((w, r.num_examples))

        first_w = weights_results[0][0]
        num_total_layers = len(first_w)
        local_limit = min(self.num_local_layers, num_total_layers)

        aggregated_weights = [np.zeros_like(first_w[i], dtype=np.float32) for i in range(num_total_layers)]

        # Keep initial layers (0 to local_limit) local
        for i in range(local_limit):
            aggregated_weights[i] = first_w[i]

        # Aggregate global head layers (local_limit to num_total_layers)
        for i in range(local_limit, num_total_layers):
            for w, num_examples in weights_results:
                weight_factor = num_examples / total_examples
                aggregated_weights[i] += w[i] * weight_factor

        self.global_parameters = aggregated_weights

        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        pers_dices = [r.metrics.get("personalized_dice", r.metrics.get("dice_score", 0.0)) for r in results]

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_dice = float(np.mean(dices)) if dices else 0.0
        avg_pers_dice = float(np.mean(pers_dices)) if pers_dices else avg_dice

        metrics = {
            "training_loss": avg_loss,
            "dice_score": avg_dice,
            "personalized_dice": avg_pers_dice,
            "num_local_layers": local_limit,
            "personalization_gain": float(avg_pers_dice - avg_dice),
        }

        logger.info(f"[LG-FedAvg] Round {server_round} - Local Layers: {local_limit}, Pers Dice: {avg_pers_dice:.4f}")
        return self.global_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[EvaluateResult],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        if not results:
            return None, {}

        total_examples = sum(r.num_examples for r in results)
        weighted_loss = sum(r.loss * r.num_examples for r in results) / total_examples
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        avg_dice = float(np.mean(dices)) if dices else 0.0

        return weighted_loss, {"val_loss": weighted_loss, "val_dice": avg_dice}

    def get_state(self) -> Dict[str, Any]:
        return {
            "global_parameters": [p.tolist() for p in self.global_parameters] if self.global_parameters else None,
            "num_local_layers": self.num_local_layers,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
