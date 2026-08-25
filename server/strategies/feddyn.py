"""
Module: server.strategies.feddyn

Purpose:
FedDyn (Dynamic Regularization for Federated Learning, Acar et al., ICLR 2021).
Tracks dynamic server state vector h_t to eliminate objective inconsistency and client drift.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("feddyn")


@register_strategy("FedDyn")
class FedDyn(BaseStrategy):
    """
    FedDyn strategy implementing dynamic regularization and server state vector h_t updates.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

        self.global_parameters: Optional[List[np.ndarray]] = None
        self.h_t: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedDyn",
            version="1.0.0",
            description="Federated Learning Based on Dynamic Regularization (Acar et al., ICLR 2021)",
            supported_features=["dynamic_regularization", "server_state_tracking", "drift_elimination"],
            supported_config=["alpha"],
            research_reference="Federated Learning Based on Dynamic Regularization, Acar et al., ICLR 2021 (arXiv:2111.00263)",
            default_parameters={
                "algorithm_name": "FedDyn",
                "authors": "Acar et al.",
                "year": 2021,
                "alpha": self.alpha,
            },
        )

    def _init_state_vector(self, params: List[np.ndarray]) -> None:
        if self.h_t is None:
            self.h_t = [np.zeros_like(p, dtype=np.float32) for p in params]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            logger.warning(f"[FedDyn] Insufficient fit results ({len(results)}/{self.min_fit_clients})")
            return None, {}

        num_clients = len(results)
        weights = [
            [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            for r in results
        ]

        if self.global_parameters is None:
            self.global_parameters = weights[0]
            self._init_state_vector(self.global_parameters)
            return self.global_parameters, {"training_loss": 0.0, "dice_score": 0.0}

        self._init_state_vector(self.global_parameters)

        # 1. Compute mean client weight w_avg = (1/N) * sum(w_i)
        w_avg = [
            np.zeros_like(self.global_parameters[k], dtype=np.float32)
            for k in range(len(self.global_parameters))
        ]
        for w in weights:
            for k in range(len(w)):
                w_avg[k] += w[k] / num_clients

        # 2. Update dynamic server state vector h_t = h_{t-1} - alpha * (w_avg - x_t)
        for k in range(len(self.global_parameters)):
            self.h_t[k] -= self.alpha * (w_avg[k] - self.global_parameters[k])

        # 3. Global update: x_{t+1} = w_avg - (1 / alpha) * h_t
        for k in range(len(self.global_parameters)):
            self.global_parameters[k] = w_avg[k] - (1.0 / self.alpha) * self.h_t[k]

        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_dice = float(np.mean(dices)) if dices else 0.0
        h_norm = float(np.sqrt(sum(np.sum(h ** 2) for h in self.h_t)))

        metrics = {
            "training_loss": avg_loss,
            "dice_score": avg_dice,
            "server_state_norm": h_norm,
            "feddyn_alpha": self.alpha,
        }

        logger.info(f"[FedDyn] Round {server_round} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, h_norm: {h_norm:.4f}")
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
            "h_t": [h.tolist() for h in self.h_t] if self.h_t else None,
            "alpha": self.alpha,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
        if state.get("h_t"):
            self.h_t = [np.array(h, dtype=np.float32) for h in state["h_t"]]
