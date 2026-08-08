"""
Module: server.strategies.fedyogi

Purpose:
FedYogi (Adaptive Server Optimization for Federated Learning, Reddi et al., ICLR 2021).
Applies Yogi adaptive variance update v_t = v_{t-1} - (1 - beta_2) delta^2 sign(v_{t-1} - delta^2)
to prevent aggressive learning rate decay in non-IID settings.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from server.strategies.base import BaseStrategy, FitResult, EvaluateResult, StrategyMetadata
from server.strategies.registry import register_strategy
from utils.logger import get_logger

logger = get_logger("fedyogi")


@register_strategy("FedYogi")
class FedYogi(BaseStrategy):
    """
    FedYogi strategy implementing server-side Yogi adaptive optimization.
    """

    def __init__(
        self,
        eta: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        tau: float = 1e-3,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eta = float(eta)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.tau = float(tau)
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

        self.global_parameters: Optional[List[np.ndarray]] = None
        self.m_t: Optional[List[np.ndarray]] = None
        self.v_t: Optional[List[np.ndarray]] = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedYogi",
            version="1.0.0",
            description="Adaptive Federated Optimization with Server Yogi (Reddi et al., ICLR 2021)",
            supported_features=["server_momentum", "adaptive_variance", "differential_privacy", "homomorphic_encryption"],
            supported_config=["eta", "beta_1", "beta_2", "tau"],
            research_reference="Adaptive Federated Optimization, Reddi et al., ICLR 2021 (arXiv:2003.00295)",
            default_parameters={
                "algorithm_name": "FedYogi",
                "authors": "Reddi et al.",
                "year": 2021,
                "eta": self.eta,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "tau": self.tau,
            },
        )

    def _init_moments(self, params: List[np.ndarray]) -> None:
        if self.m_t is None:
            self.m_t = [np.zeros_like(p, dtype=np.float32) for p in params]
            self.v_t = [np.full_like(p, fill_value=self.tau ** 2, dtype=np.float32) for p in params]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        if not results or len(results) < self.min_fit_clients:
            logger.warning(f"[FedYogi] Insufficient fit results ({len(results)}/{self.min_fit_clients})")
            return None, {}

        # 1. Standard FedAvg weighted target
        total_examples = sum(r.num_examples for r in results)
        weights_results = []
        for r in results:
            w = [np.array(p, dtype=np.float32) if not isinstance(p, np.ndarray) else p for p in r.parameters]
            weights_results.append((w, r.num_examples))

        w_target = [
            np.zeros_like(weights_results[0][0][i], dtype=np.float32)
            for i in range(len(weights_results[0][0]))
        ]

        for w, num_examples in weights_results:
            weight_factor = num_examples / total_examples
            for i in range(len(w)):
                w_target[i] += w[i] * weight_factor

        if self.global_parameters is None:
            self.global_parameters = w_target
            self._init_moments(self.global_parameters)
            return self.global_parameters, {"training_loss": 0.0, "dice_score": 0.0}

        self._init_moments(self.global_parameters)

        # 2. Compute pseudogradient delta = w_target - x_t
        delta = [w_target[i] - self.global_parameters[i] for i in range(len(w_target))]

        # 3. Update server Yogi 1st & 2nd moments
        for i in range(len(delta)):
            self.m_t[i] = self.beta_1 * self.m_t[i] + (1.0 - self.beta_1) * delta[i]

            delta_sq = delta[i] ** 2
            # Yogi 2nd moment update: v_t = v_{t-1} - (1-beta_2) * delta^2 * sign(v_{t-1} - delta^2)
            self.v_t[i] = self.v_t[i] - (1.0 - self.beta_2) * delta_sq * np.sign(self.v_t[i] - delta_sq)
            self.v_t[i] = np.maximum(self.v_t[i], 1e-8)  # numerical stability

            # Server update: x_{t+1} = x_t + eta * m_t / (sqrt(v_t) + tau)
            self.global_parameters[i] += self.eta * self.m_t[i] / (np.sqrt(self.v_t[i]) + self.tau)

        # 4. Metrics
        losses = [r.metrics.get("training_loss", 0.0) for r in results if "training_loss" in r.metrics]
        dices = [r.metrics.get("dice_score", 0.0) for r in results if "dice_score" in r.metrics]

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_dice = float(np.mean(dices)) if dices else 0.0
        m_norm = float(np.sqrt(sum(np.sum(m ** 2) for m in self.m_t)))
        v_norm = float(np.sqrt(sum(np.sum(v ** 2) for v in self.v_t)))

        metrics = {
            "training_loss": avg_loss,
            "dice_score": avg_dice,
            "server_momentum_norm": m_norm,
            "server_variance_norm": v_norm,
            "adaptive_lr": self.eta,
        }

        logger.info(f"[FedYogi] Round {server_round} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, m_norm: {m_norm:.4f}")
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
            "m_t": [m.tolist() for m in self.m_t] if self.m_t else None,
            "v_t": [v.tolist() for v in self.v_t] if self.v_t else None,
            "eta": self.eta,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if state.get("global_parameters"):
            self.global_parameters = [np.array(p, dtype=np.float32) for p in state["global_parameters"]]
        if state.get("m_t"):
            self.m_t = [np.array(m, dtype=np.float32) for m in state["m_t"]]
        if state.get("v_t"):
            self.v_t = [np.array(v, dtype=np.float32) for v in state["v_t"]]
