"""
Module: server.strategies.trimmed_mean

Purpose:
Coordinate-Wise Trimmed Mean Byzantine-Robust Aggregation (Yin et al., ICML 2018).
Removes the largest and smallest beta-fraction of values per coordinate before averaging.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("trimmed_mean")


@register_strategy("TrimmedMean")
class TrimmedMean(BaseStrategy):
    """Coordinate-wise Trimmed Mean Byzantine-robust strategy."""

    def __init__(self, beta: float = 0.1, min_fit_clients: int = 2, min_available_clients: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="TrimmedMean",
            version="1.0.0",
            description="Coordinate-Wise Trimmed Mean (Yin et al., ICML 2018)",
            supported_features=["byzantine_robust", "coordinate_wise"],
            supported_config=["beta"],
            research_reference="Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates, Yin et al., ICML 2018",
            default_parameters={"beta": self.beta},
        )

    def aggregate_fit(self, server_round: int, results: List[FitResult], failures: List[Any]) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            return None, {}

        client_updates = []
        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            client_updates.append(w)

        n = len(client_updates)
        k_trim = max(1, int(n * self.beta))
        aggregated = []
        for l_idx in range(len(client_updates[0])):
            stacked = np.stack([u[l_idx] for u in client_updates], axis=0)
            sorted_l = np.sort(stacked, axis=0)
            upper = max(k_trim, n - k_trim)
            trimmed = sorted_l[k_trim:upper] if upper > k_trim else sorted_l
            aggregated.append(np.mean(trimmed, axis=0))

        self.global_parameters = aggregated
        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        metrics = {
            "training_loss": float(np.mean(losses)) if losses else 0.0,
            "dice_score": float(np.mean(dices)) if dices else 0.0,
            "trim_fraction": self.beta,
            "trimmed_count": k_trim,
        }
        logger.info(f"[TrimmedMean] Round {server_round} - Trimmed {k_trim} per side, beta={self.beta}")
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
