"""
Module: server.strategies.fednova

Purpose:
FedNova (Federated Normalized Averaging, Wang et al., NeurIPS 2020).
Eliminates objective inconsistency caused by heterogeneous local step counts tau_i across clients.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("fednova")


@register_strategy("FedNova")
class FedNova(BaseStrategy):
    """
    FedNova strategy implementing normalized update aggregation across heterogeneous local steps.
    """

    def __init__(
        self,
        gmf: float = 0.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gmf = float(gmf)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

        self.global_parameters: Optional[List[np.ndarray]] = None
        self.v_t: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedNova",
            version="1.0.0",
            description="Tackling Heterogeneity in Federated Optimization via Normalized Averaging (Wang et al., NeurIPS 2020)",
            supported_features=["normalized_aggregation", "objective_consistency", "heterogeneous_local_steps"],
            supported_config=["gmf"],
            research_reference="Tackling Heterogeneity in Federated Optimization, Wang et al., NeurIPS 2020 (arXiv:2007.07481)",
            default_parameters={
                "algorithm_name": "FedNova",
                "authors": "Wang et al.",
                "year": 2020,
                "gmf": self.gmf,
            },
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            logger.warning(f"[FedNova] Insufficient fit results ({len(results)}/{self.min_fit_clients})")
            return None, {}

        total_examples = sum(r.num_examples for r in results)

        if self.global_parameters is None:
            # First round initialization
            w_0 = [np.array(p, dtype=np.float32) for p in results[0].parameters]
            self.global_parameters = w_0
            return self.global_parameters, {"training_loss": 0.0, "dice_score": 0.0}

        # 1. Collect normalized updates Delta_i = (y_i - x_t) / tau_i
        tau_list = []
        normalized_deltas = []
        p_weights = []

        for r in results:
            w_i = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            tau_i = float(r.metrics.get("epochs", r.metrics.get("local_steps", 1.0)))
            tau_list.append(tau_i)

            p_i = r.num_examples / total_examples
            p_weights.append(p_i)

            # Delta_i = (y_i - x_t) / tau_i
            delta_i = [(w_i[k] - self.global_parameters[k]) / max(1.0, tau_i) for k in range(len(w_i))]
            normalized_deltas.append(delta_i)

        # 2. Compute effective step tau_eff = sum(p_i * tau_i)
        tau_eff = sum(p_weights[idx] * tau_list[idx] for idx in range(len(results)))

        # 3. Aggregated update Delta_Nova = tau_eff * sum(p_i * Delta_i)
        delta_nova = [
            np.zeros_like(self.global_parameters[k], dtype=np.float32)
            for k in range(len(self.global_parameters))
        ]

        for idx in range(len(results)):
            p_i = p_weights[idx]
            d_i = normalized_deltas[idx]
            for k in range(len(delta_nova)):
                delta_nova[k] += p_i * d_i[k]

        for k in range(len(delta_nova)):
            delta_nova[k] *= tau_eff

        # Apply global momentum if configured
        if self.gmf > 0.0:
            if self.v_t is None:
                self.v_t = [np.zeros_like(d, dtype=np.float32) for d in delta_nova]
            for k in range(len(delta_nova)):
                self.v_t[k] = self.gmf * self.v_t[k] + delta_nova[k]
                delta_nova[k] = self.v_t[k]

        # 4. Server Update: x_{t+1} = x_t + Delta_Nova
        for k in range(len(self.global_parameters)):
            self.global_parameters[k] += delta_nova[k]

        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_dice = float(np.mean(dices)) if dices else 0.0
        norm_val = float(np.sqrt(sum(np.sum(d ** 2) for d in delta_nova)))

        metrics = {
            "training_loss": avg_loss,
            "dice_score": avg_dice,
            "tau_eff": tau_eff,
            "fednova_update_norm": norm_val,
        }

        logger.info(f"[FedNova] Round {server_round} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, tau_eff: {tau_eff:.2f}")
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
            "v_t": [v.tolist() for v in self.v_t] if self.v_t else None,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
        if state.get("v_t"):
            self.v_t = [np.array(v, dtype=np.float32) for v in state["v_t"]]
