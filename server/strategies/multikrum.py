"""
Module: server.strategies.multikrum

Purpose:
Multi-Krum Byzantine-Robust Aggregation (Blanchard et al., NeurIPS 2017).
Averages the top-m updates with the lowest Krum scores.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("multikrum")


@register_strategy("MultiKrum")
class MultiKrum(BaseStrategy):
    """Multi-Krum Byzantine-robust aggregation strategy."""

    def __init__(self, f_byzantine: int = 1, m_select: int = 3, min_fit_clients: int = 2, min_available_clients: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.f_byzantine = int(f_byzantine)
        self.m_select = int(m_select)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="MultiKrum",
            version="1.0.0",
            description="Multi-Krum Byzantine-Robust Aggregation (Blanchard et al., NeurIPS 2017)",
            supported_features=["byzantine_robust", "adversarial_resilient"],
            supported_config=["f_byzantine", "m_select"],
            research_reference="Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent, Blanchard et al., NeurIPS 2017",
            default_parameters={"f_byzantine": self.f_byzantine, "m_select": self.m_select},
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
            top_indices = list(range(min(n, self.m_select)))
        else:
            k_neighbors = n - self.f_byzantine - 2
            scores = []
            for i in range(n):
                dists = sorted([np.linalg.norm(flat[i] - flat[j]) ** 2 for j in range(n) if j != i])
                scores.append(sum(dists[:k_neighbors]))
            top_indices = list(np.argsort(scores)[:min(n, self.m_select)])

        num_layers = len(client_updates[0])
        aggregated = [np.mean([client_updates[i][k] for i in top_indices], axis=0) for k in range(num_layers)]
        self.global_parameters = aggregated

        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        metrics = {
            "training_loss": float(np.mean(losses)) if losses else 0.0,
            "dice_score": float(np.mean(dices)) if dices else 0.0,
            "selected_clients": top_indices,
            "m_select": self.m_select,
        }
        logger.info(f"[MultiKrum] Round {server_round} - Selected {len(top_indices)} clients")
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
