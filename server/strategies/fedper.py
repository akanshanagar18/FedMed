"""
Module: server.strategies.fedper

Purpose:
FedPer (Federated Learning with Personalized Layers, Arivazhagan et al., 2019).
Aggregates shared feature representation backbone parameters globally while maintaining
personalized classification/segmentation heads locally on each hospital client.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("fedper")


@register_strategy("FedPer")
class FedPer(BaseStrategy):
    """
    FedPer strategy implementing shared backbone aggregation with local personalized heads.
    """

    def __init__(
        self,
        num_shared_layers: int = 4,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_shared_layers = int(num_shared_layers)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedPer",
            version="1.0.0",
            description="Federated Learning with Personalized Layers (Arivazhagan et al., 2019)",
            supported_features=["personalized_fl", "shared_backbone", "local_heads"],
            supported_config=["num_shared_layers"],
            research_reference="Federated Learning with Personalized Layers, Arivazhagan et al., 2019 (arXiv:1912.00818)",
            default_parameters={
                "algorithm_name": "FedPer",
                "authors": "Arivazhagan et al.",
                "year": 2019,
                "num_shared_layers": self.num_shared_layers,
            },
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            logger.warning(f"[FedPer] Insufficient fit results ({len(results)}/{self.min_fit_clients})")
            return None, {}

        total_examples = sum(r.num_examples for r in results)
        weights_results = []

        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            weights_results.append((w, r.num_examples))

        first_w = weights_results[0][0]
        num_total_layers = len(first_w)
        shared_limit = min(self.num_shared_layers, num_total_layers)

        # Aggregate only the first `num_shared_layers` (backbone)
        aggregated_weights = [np.zeros_like(first_w[i], dtype=np.float32) for i in range(num_total_layers)]

        for i in range(shared_limit):
            for w, num_examples in weights_results:
                weight_factor = num_examples / total_examples
                aggregated_weights[i] += w[i] * weight_factor

        # Pass through remaining local head layers unchanged
        for i in range(shared_limit, num_total_layers):
            aggregated_weights[i] = first_w[i]

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
            "num_shared_layers": shared_limit,
            "personalization_gain": float(avg_pers_dice - avg_dice),
        }

        logger.info(f"[FedPer] Round {server_round} - Shared Layers: {shared_limit}, Pers Dice: {avg_pers_dice:.4f}")
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
            "num_shared_layers": self.num_shared_layers,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
