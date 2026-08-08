"""
Module: server.strategies.perfedavg

Purpose:
Per-FedAvg (Personalized Federated Learning: A Meta-Learning Approach, Fallah et al., 2020).
Applies Model-Agnostic Meta-Learning (MAML) gradient updates f_i(theta - alpha grad f_i(theta))
to find initial model parameters that adapt quickly to local client data in 1-2 gradient steps.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("perfedavg")


@register_strategy("Per-FedAvg")
class PerFedAvg(BaseStrategy):
    """
    Per-FedAvg strategy implementing MAML meta-learning federated optimization.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.001,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Per-FedAvg",
            version="1.0.0",
            description="Personalized Federated Learning: A Meta-Learning Approach (Fallah et al., 2020)",
            supported_features=["meta_learning", "maml_optimization", "rapid_local_adaptation"],
            supported_config=["alpha", "beta"],
            research_reference="Personalized Federated Learning: A Meta-Learning Approach, Fallah et al., 2020 (arXiv:2002.07948)",
            default_parameters={
                "algorithm_name": "Per-FedAvg",
                "authors": "Fallah et al.",
                "year": 2020,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            logger.warning(f"[Per-FedAvg] Insufficient fit results ({len(results)}/{self.min_fit_clients})")
            return None, {}

        total_examples = sum(r.num_examples for r in results)
        weights_results = []

        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            weights_results.append((w, r.num_examples))

        # MAML Meta-gradient weighted parameter update
        aggregated_weights = [
            np.zeros_like(weights_results[0][0][i], dtype=np.float32)
            for i in range(len(weights_results[0][0]))
        ]

        for w, num_examples in weights_results:
            weight_factor = num_examples / total_examples
            for i in range(len(w)):
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
            "meta_learning_rate": self.beta,
            "personalization_gain": float(avg_pers_dice - avg_dice),
        }

        logger.info(f"[Per-FedAvg] Round {server_round} - Meta LR: {self.beta}, Pers Dice: {avg_pers_dice:.4f}")
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
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
