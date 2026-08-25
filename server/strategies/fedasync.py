"""
Module: server.strategies.fedasync

Purpose:
FedAsync (Asynchronous Federated Learning, Xie et al., 2019).
Supports asynchronous model updates with configurable staleness weighting functions
and bounded delay constraints for real-world heterogeneous edge environments.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("fedasync")


@register_strategy("FedAsync")
class FedAsync(BaseStrategy):
    """
    FedAsync strategy implementing asynchronous client aggregation and staleness weighting.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        staleness_func: str = "polynomial",  # "constant", "polynomial", "hinge"
        a: float = 0.5,
        max_staleness: int = 10,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.staleness_func = str(staleness_func)
        self.a = float(a)
        self.max_staleness = int(max_staleness)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

        self.global_parameters: Optional[List[np.ndarray]] = None
        self.server_timestamp: int = 0

        self.accepted_updates: int = 0
        self.rejected_updates: int = 0
        self.total_staleness: int = 0

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedAsync",
            version="1.0.0",
            description="Asynchronous Federated Learning with Staleness Weighting (Xie et al., 2019)",
            supported_features=["asynchronous_aggregation", "staleness_weighting", "bounded_delay"],
            supported_config=["alpha", "staleness_func", "a", "max_staleness"],
            research_reference="Asynchronous Federated Optimization, Xie et al., 2019 (arXiv:1903.03934)",
            default_parameters={
                "algorithm_name": "FedAsync",
                "authors": "Xie et al.",
                "year": 2019,
                "alpha": self.alpha,
                "staleness_func": self.staleness_func,
                "a": self.a,
                "max_staleness": self.max_staleness,
            },
        )

    def _compute_staleness_weight(self, staleness: int) -> float:
        if self.staleness_func == "constant":
            return 1.0
        elif self.staleness_func == "polynomial":
            return float((staleness + 1) ** (-self.a))
        elif self.staleness_func == "hinge":
            return 1.0 if staleness <= (self.max_staleness // 2) else float(1.0 / (staleness - (self.max_staleness // 2) + 1))
        return float((staleness + 1) ** (-self.a))

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results:
            logger.warning("[FedAsync] No fit results received for asynchronous aggregation.")
            return None, {}

        if self.global_parameters is None:
            # Initialize with first arriving client model
            self.global_parameters = [np.array(p, dtype=np.float32) for p in results[0].parameters]
            self.server_timestamp = 1
            return self.global_parameters, {"training_loss": 0.0, "dice_score": 0.0}

        # Process each arriving asynchronous client result sequentially
        for r in results:
            client_time = int(r.metrics.get("client_timestamp", r.metrics.get("round", self.server_timestamp - 1)))
            staleness = max(0, self.server_timestamp - client_time)

            if staleness > self.max_staleness:
                self.rejected_updates += 1
                logger.warning(f"[FedAsync] Update rejected due to excessive staleness: {staleness} > {self.max_staleness}")
                continue

            self.accepted_updates += 1
            self.total_staleness += staleness

            # S(tau) staleness weighting factor
            s_weight = self._compute_staleness_weight(staleness)
            effective_alpha = self.alpha * s_weight

            w_client = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]

            # x_{t+1} = (1 - alpha_eff) * x_t + alpha_eff * y_i
            for k in range(len(self.global_parameters)):
                self.global_parameters[k] = (1.0 - effective_alpha) * self.global_parameters[k] + effective_alpha * w_client[k]

            self.server_timestamp += 1

        total_updates = self.accepted_updates + self.rejected_updates
        stale_ratio = float(self.rejected_updates / total_updates) if total_updates > 0 else 0.0
        avg_staleness = float(self.total_staleness / self.accepted_updates) if self.accepted_updates > 0 else 0.0

        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_dice = float(np.mean(dices)) if dices else 0.0

        metrics = {
            "training_loss": avg_loss,
            "dice_score": avg_dice,
            "server_timestamp": self.server_timestamp,
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "stale_update_ratio": stale_ratio,
            "average_staleness": avg_staleness,
        }

        logger.info(f"[FedAsync] Server timestamp {self.server_timestamp} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, Staleness: {avg_staleness:.2f}")
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
            "server_timestamp": self.server_timestamp,
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "total_staleness": self.total_staleness,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
        self.server_timestamp = state.get("server_timestamp", 0)
        self.accepted_updates = state.get("accepted_updates", 0)
        self.rejected_updates = state.get("rejected_updates", 0)
        self.total_staleness = state.get("total_staleness", 0)
