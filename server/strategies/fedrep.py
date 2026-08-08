"""
Module: server.strategies.fedrep

Purpose:
FedRep (Exploiting Shared Representations for Personalized Federated Learning, Collins et al., 2021).
Alternates optimization between local classification heads and global feature representation backbone.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("fedrep")


@register_strategy("FedRep")
class FedRep(BaseStrategy):
    """
    FedRep strategy implementing representation learning with local head personalization.
    """

    def __init__(
        self,
        representation_layers: int = 4,
        head_epochs: int = 2,
        rep_epochs: int = 1,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.representation_layers = int(representation_layers)
        self.head_epochs = int(head_epochs)
        self.rep_epochs = int(rep_epochs)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedRep",
            version="1.0.0",
            description="Exploiting Shared Representations for Personalized Federated Learning (Collins et al., 2021)",
            supported_features=["representation_learning", "alternating_optimization", "personalized_heads"],
            supported_config=["representation_layers", "head_epochs", "rep_epochs"],
            research_reference="Exploiting Shared Representations for Personalized Federated Learning, Collins et al., 2021 (arXiv:2102.07078)",
            default_parameters={
                "algorithm_name": "FedRep",
                "authors": "Collins et al.",
                "year": 2021,
                "representation_layers": self.representation_layers,
                "head_epochs": self.head_epochs,
                "rep_epochs": self.rep_epochs,
            },
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            logger.warning(f"[FedRep] Insufficient fit results ({len(results)}/{self.min_fit_clients})")
            return None, {}

        total_examples = sum(r.num_examples for r in results)
        weights_results = []

        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            weights_results.append((w, r.num_examples))

        first_w = weights_results[0][0]
        num_total = len(first_w)
        rep_limit = min(self.representation_layers, num_total)

        # Aggregate representation backbone parameters globally
        aggregated_weights = [np.zeros_like(first_w[i], dtype=np.float32) for i in range(num_total)]

        for i in range(rep_limit):
            for w, num_examples in weights_results:
                weight_factor = num_examples / total_examples
                aggregated_weights[i] += w[i] * weight_factor

        # Pass through local heads
        for i in range(rep_limit, num_total):
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
            "representation_layers": rep_limit,
            "personalization_gain": float(avg_pers_dice - avg_dice),
        }

        logger.info(f"[FedRep] Round {server_round} - Rep Layers: {rep_limit}, Pers Dice: {avg_pers_dice:.4f}")
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
            "representation_layers": self.representation_layers,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
