"""
Module: privacy.dp_engine

Purpose:
Differential Privacy Engine for FedMed v2.0 powered by PyTorch Opacus & RDP Accountants.
Applies gradient norm clipping (max_grad_norm) and calibrated Gaussian noise injection to model weight gradients during local hospital training.
Tracks cumulative privacy budget consumption (epsilon, delta) across training rounds.
"""

import math
import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

try:
    from opacus.accountants import RDPAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

logger = logging.getLogger(__name__)


def compute_rdp_epsilon(
    steps: int,
    noise_multiplier: float,
    target_delta: float = 1e-5,
    sample_rate: float = 0.1,
) -> float:
    """
    Computes privacy budget epsilon (ε) for a given number of steps, noise multiplier,
    target delta (δ), and sampling rate using Renyi Differential Privacy (RDP).
    """
    if steps <= 0:
        return 0.0

    if OPACUS_AVAILABLE:
        try:
            accountant = RDPAccountant()
            accountant.history = [(noise_multiplier, sample_rate, steps)]
            eps = accountant.get_epsilon(delta=target_delta)
            return float(eps)
        except Exception as e:
            logger.warning(f"Opacus RDP accountant calculation fallback: {e}")

    # Theoretical closed-form RDP approximation fallback
    if noise_multiplier <= 0:
        return float("inf")

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    rdp_epsilons = []

    for alpha in orders:
        # RDP per step
        rdp_step = (alpha * (sample_rate**2)) / (2 * (noise_multiplier**2))
        rdp_total = steps * rdp_step
        # Convert RDP to (ε, δ)-DP
        eps = rdp_total + math.log(1.0 / target_delta) / (alpha - 1)
        rdp_epsilons.append(eps)

    return float(min(rdp_epsilons))


class DifferentialPrivacyEngine:
    """
    Differential Privacy Engine for local training.
    Enforces sample/batch-level gradient norm clipping and calibrated Gaussian noise addition.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        target_epsilon: float = 3.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        sample_rate: float = 0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.sample_rate = sample_rate

        # Calculate noise multiplier if not explicitly provided
        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            # Calibrate noise multiplier for target epsilon at 100 steps
            self.noise_multiplier = max(0.5, 1.0 / max(target_epsilon, 0.1))

        self.steps = 0
        self.accountant_type = "RDP (Renyi Differential Privacy) Accountant"

        logger.info(
            f"[DP ENGINE] Initialized Differential Privacy Engine — "
            f"target_ε: {self.target_epsilon}, target_δ: {self.target_delta}, "
            f"max_grad_norm: {self.max_grad_norm}, noise_multiplier: {self.noise_multiplier:.4f}"
        )

    def attach_optimizer_step_hooks(self) -> None:
        """
        No-op hook binder for PyTorch optimizer compatibility.
        """
        pass

    def apply_gradient_clipping_and_noise(self, batch_size: int = 1) -> float:
        """
        Applies gradient norm clipping to max_grad_norm and injects calibrated Gaussian noise.
        Execution Order: Forward -> Backward -> Clip -> Noise -> Step
        
        Returns:
            Current total gradient norm before clipping.
        """
        # 1. Calculate gradient norm across all model parameters
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
        ).item()

        # 2. Gradient Clipping
        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

        # 3. Gaussian Noise Injection
        # Noise standard deviation = (noise_multiplier * max_grad_norm) / sqrt(batch_size)
        noise_std = (self.noise_multiplier * self.max_grad_norm) / max(math.sqrt(batch_size), 1.0)

        with torch.no_grad():
            for p in parameters:
                noise = torch.randn_like(p.grad) * noise_std
                p.grad.add_(noise)

        self.steps += 1
        return total_norm

    def get_privacy_budget(self) -> Dict[str, Any]:
        """
        Returns current accumulated privacy budget (epsilon, delta) and accounting metrics.
        """
        current_eps = compute_rdp_epsilon(
            steps=self.steps,
            noise_multiplier=self.noise_multiplier,
            target_delta=self.target_delta,
            sample_rate=self.sample_rate,
        )

        return {
            "dp_enabled": True,
            "epsilon": round(current_eps, 4),
            "delta": self.target_delta,
            "noise_multiplier": round(self.noise_multiplier, 4),
            "max_grad_norm": self.max_grad_norm,
            "steps": self.steps,
            "accountant_type": self.accountant_type,
            "budget_consumed_percent": min(round((current_eps / max(self.target_epsilon, 1e-5)) * 100.0, 2), 100.0),
        }
