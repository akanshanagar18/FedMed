"""
Module: server.strategies.fedbn

Purpose:
FedBN (Federated Learning with Local Batch Normalization, Li et al., ICLR 2021).
Keeps Batch Normalization statistics and affine parameters local to personalize domain normalization under medical scanner shifts.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("fedbn")


@register_strategy("FedBN")
class FedBN(BaseStrategy):
    """
    FedBN strategy implementing local Batch Normalization statistics preservation.
    """

    def __init__(
        self,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.global_parameters: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedBN",
            version="1.0.0",
            description="Federated Learning on Non-IID Data via Local Batch Normalization (Li et al., ICLR 2021)",
            supported_features=["local_batch_norm", "domain_personalization", "medical_imaging_heterogeneity"],
            supported_config=[],
            research_reference="FedBN: Federated Learning on Non-IID Data via Local Batch Normalization, Li et al., ICLR 2021 (arXiv:2102.02079)",
            default_parameters={
                "algorithm_name": "FedBN",
                "authors": "Li et al.",
                "year": 2021,
            },
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            logger.warning(f"[FedBN] Insufficient fit results ({len(results)}/{self.min_fit_clients})")
            return None, {}

        total_examples = sum(r.num_examples for r in results)
        weights_results = []

        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            weights_results.append((w, r.num_examples))

        # Standard weighted parameter aggregation (client model trainer excludes local BN params during FitRes payload)
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

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_dice = float(np.mean(dices)) if dices else 0.0

        metrics = {
            "training_loss": avg_loss,
            "dice_score": avg_dice,
            "local_bn_active": True,
        }

        logger.info(f"[FedBN] Round {server_round} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
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
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
