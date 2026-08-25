"""
Module: security.poisoning

Purpose:
Model and gradient poisoning attack simulation.
"""

import numpy as np
from typing import List


class ModelPoisoningAttack:
    """Injects corrupted weights via sign-flip or Gaussian noise."""

    def __init__(self, attack_type: str = "sign_flip", scale: float = 5.0, noise_std: float = 10.0):
        self.attack_type = attack_type
        self.scale = float(scale)
        self.noise_std = float(noise_std)

    def apply(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        if self.attack_type == "sign_flip":
            return [-w * self.scale for w in weights]
        elif self.attack_type == "gaussian":
            return [np.random.normal(0, self.noise_std, size=w.shape).astype(np.float32) for w in weights]
        return weights


class GradientPoisoningAttack:
    """Scales or inverts gradients to degrade the global model."""

    def __init__(self, scale_factor: float = -10.0):
        self.scale_factor = float(scale_factor)

    def apply(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        return [g * self.scale_factor for g in gradients]
