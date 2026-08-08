"""
Module: server.strategies.krum

Purpose:
Krum Byzantine-Robust Aggregation (Blanchard et al., NeurIPS 2017).
Selects the single client update minimizing sum of squared Euclidean distances to its closest neighbors.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("krum")


@register_strategy("Krum")
class Krum(BaseStrategy):
    """Krum Byzantine-robust aggregation strategy."""

    def __init__(self, f_byzantine: int = 1, min_fit_clients: int = 2, min_available_clients: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.f_byzantine = int(f_byzantine)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Krum",
            version="1.0.0",
            description="Krum Byzantine-Robust Aggregation (Blanchard et al., NeurIPS 2017)",
            supported_features=["byzantine_robust", "adversarial_resilient"],
            supported_config=["f_byzantine"],
            research_reference="Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent, Blanchard et al., NeurIPS 2017",
            default_parameters={"f_byzantine": self.f_byzantine},
        )

    def aggregate_fit(self, server_round: int, results: List[FitResult], failures: List[Any]) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            return None, {}

        client_updates = []
        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            client_updates.append(w)

        n = len(client_updates)
        flat = [np.concatenate([p.flatten() for p in u]) for u in client_updates]

        if n <= 2 * self.f_byzantine + 2:
            best_idx = 0
        else:
            m = n - self.f_byzantine - 2
            scores = []
            for i in range(n):
                dists = sorted([np.linalg.norm(flat[i] - flat[j]) ** 2 for j in range(n) if j != i])
                scores.append(sum(dists[:m]))
            best_idx = int(np.argmin(scores))

        self.global_parameters = client_updates[best_idx]

        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        metrics = {
            "training_loss": float(np.mean(losses)) if losses else 0.0,
            "dice_score": float(np.mean(dices)) if dices else 0.0,
            "selected_client_idx": best_idx,
            "byzantine_tolerance": self.f_byzantine,
        }
        logger.info(f"[Krum] Round {server_round} - Selected client {best_idx}, f={self.f_byzantine}")
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
