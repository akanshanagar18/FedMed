"""
Module: server.strategies.median

Purpose:
Coordinate-Wise Median Byzantine-Robust Aggregation (Yin et al., ICML 2018).
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("median")


@register_strategy("Median")
class Median(BaseStrategy):
    """Coordinate-wise Median Byzantine-robust strategy."""

    def __init__(self, min_fit_clients: int = 2, min_available_clients: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Median",
            version="1.0.0",
            description="Coordinate-Wise Median Byzantine-Robust Aggregation (Yin et al., ICML 2018)",
            supported_features=["byzantine_robust", "coordinate_wise"],
            supported_config=[],
            research_reference="Byzantine-Robust Distributed Learning, Yin et al., ICML 2018",
            default_parameters={},
        )

    def aggregate_fit(self, server_round: int, results: List[FitResult], failures: List[Any]) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            return None, {}

        client_updates = []
        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            client_updates.append(w)

        aggregated = []
        for l_idx in range(len(client_updates[0])):
            stacked = np.stack([u[l_idx] for u in client_updates], axis=0)
            aggregated.append(np.median(stacked, axis=0).astype(np.float32))

        self.global_parameters = aggregated
        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        metrics = {
            "training_loss": float(np.mean(losses)) if losses else 0.0,
            "dice_score": float(np.mean(dices)) if dices else 0.0,
        }
        logger.info(f"[Median] Round {server_round} - Coordinate-wise median of {len(results)} clients")
        return self.global_parameters, metrics

    def aggregate_evaluate(self, server_round: int, results: List[EvaluateResult], failures: List[Any]) -> Tuple[Optional[float], Dict[str, Any]]:
        if not results:
            return None, {}
        total = sum(r.num_examples for r in results)
        loss = sum(r.loss * r.num_examples for r in results) / total
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        return loss, {"val_loss": loss, "val_dice": float(np.mean(dices)) if dices else 0.0}

    def get_state(self) -> Dict[str, Any]:
        return {"global_parameters": [p.tolist() for p in self.global_parameters] if self.global_parameters else None}

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
