"""
Module: server.strategies.bulyan

Purpose:
Bulyan Byzantine-Robust Aggregation (El Mhamdi et al., ICML 2018).
Combines Multi-Krum candidate selection with Trimmed Mean coordinate aggregation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("bulyan")


@register_strategy("Bulyan")
class Bulyan(BaseStrategy):
    """Bulyan Byzantine-robust aggregation strategy."""

    def __init__(self, f_byzantine: int = 1, min_fit_clients: int = 2, min_available_clients: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.f_byzantine = int(f_byzantine)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Bulyan",
            version="1.0.0",
            description="Bulyan Byzantine-Robust Aggregation (El Mhamdi et al., ICML 2018)",
            supported_features=["byzantine_robust", "krum_trimmed_hybrid"],
            supported_config=["f_byzantine"],
            research_reference="The Hidden Vulnerability of Distributed Learning in Byzantium, El Mhamdi et al., ICML 2018",
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
        theta = max(1, n - 2 * self.f_byzantine)
        flat = [np.concatenate([p.flatten() for p in u]) for u in client_updates]

        # Phase 1: Multi-Krum selection of theta candidates
        if n <= 2 * self.f_byzantine + 2:
            selected_indices = list(range(n))
        else:
            k_neighbors = n - self.f_byzantine - 2
            scores = []
            for i in range(n):
                dists = sorted([np.linalg.norm(flat[i] - flat[j]) ** 2 for j in range(n) if j != i])
                scores.append(sum(dists[:k_neighbors]))
            selected_indices = list(np.argsort(scores)[:theta])

        # Phase 2: Trimmed Mean over selected candidates
        selected = [client_updates[i] for i in selected_indices]
        s = len(selected)
        k_trim = max(1, self.f_byzantine)

        aggregated = []
        for l_idx in range(len(selected[0])):
            stacked = np.stack([u[l_idx] for u in selected], axis=0)
            sorted_l = np.sort(stacked, axis=0)
            upper = max(k_trim, s - k_trim)
            trimmed = sorted_l[k_trim:upper] if upper > k_trim else sorted_l
            aggregated.append(np.mean(trimmed, axis=0))

        self.global_parameters = aggregated
        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        metrics = {
            "training_loss": float(np.mean(losses)) if losses else 0.0,
            "dice_score": float(np.mean(dices)) if dices else 0.0,
            "theta_candidates": theta,
            "selected_count": len(selected_indices),
        }
        logger.info(f"[Bulyan] Round {server_round} - theta={theta}, selected {len(selected_indices)}")
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
