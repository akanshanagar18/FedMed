"""
Module: server.strategies.fltrust

Purpose:
FLTrust Byzantine-Robust Aggregation (Cao et al., NDSS 2021).
Computes trust scores ReLU(cos(g_i, g_0)) using a clean server root dataset gradient.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("fltrust")


@register_strategy("FLTrust")
class FLTrust(BaseStrategy):
    """FLTrust Byzantine-robust aggregation strategy."""

    def __init__(self, min_fit_clients: int = 2, min_available_clients: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None
        self.server_root_update: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FLTrust",
            version="1.0.0",
            description="FLTrust Byzantine-Robust Aggregation (Cao et al., NDSS 2021)",
            supported_features=["byzantine_robust", "trust_scoring", "root_dataset"],
            supported_config=[],
            research_reference="FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping, Cao et al., NDSS 2021",
            default_parameters={},
        )

    def aggregate_fit(self, server_round: int, results: List[FitResult], failures: List[Any]) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            return None, {}

        client_updates = []
        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            client_updates.append(w)

        # Use first client as a simulated server root if no root is set
        root = self.server_root_update if self.server_root_update else client_updates[0]
        g0_flat = np.concatenate([p.flatten() for p in root])
        g0_norm = np.linalg.norm(g0_flat) + 1e-8

        trust_scores = []
        normalized = []
        for u in client_updates:
            u_flat = np.concatenate([p.flatten() for p in u])
            u_norm = np.linalg.norm(u_flat) + 1e-8
            cos_sim = float(np.dot(u_flat, g0_flat) / (u_norm * g0_norm))
            ts = max(0.0, cos_sim)
            trust_scores.append(ts)
            normalized.append([(p / u_norm) * g0_norm for p in u])

        sum_ts = sum(trust_scores) + 1e-8
        num_layers = len(client_updates[0])
        aggregated = [np.zeros_like(client_updates[0][k], dtype=np.float32) for k in range(num_layers)]

        for i in range(len(client_updates)):
            weight = trust_scores[i] / sum_ts
            for k in range(num_layers):
                aggregated[k] += weight * normalized[i][k]

        self.global_parameters = aggregated
        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]
        metrics = {
            "training_loss": float(np.mean(losses)) if losses else 0.0,
            "dice_score": float(np.mean(dices)) if dices else 0.0,
            "trust_scores": trust_scores,
            "mean_trust": float(np.mean(trust_scores)),
        }
        logger.info(f"[FLTrust] Round {server_round} - Mean trust: {np.mean(trust_scores):.4f}")
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
