"""
Module: security.attacks

Purpose:
Adversarial attack simulators for robust federated learning research.
Simulates Label Flipping, Model Poisoning, Gradient Poisoning, Backdoor Attacks, and Sybil Attacks.
"""

from typing import List, Tuple
import numpy as np


class LabelFlippingAttack:
    """Simulates label permutation / flipping attacks on malicious client datasets."""

    def __init__(self, target_class: int = 0, flipped_class: int = 1):
        self.target_class = target_class
        self.flipped_class = flipped_class

    def flip_labels(self, labels: np.ndarray) -> np.ndarray:
        corrupted = labels.copy()
        corrupted[labels == self.target_class] = self.flipped_class
        return corrupted


class ModelPoisoningAttack:
    """Simulates sign flipping or random Gaussian noise injection into model weights."""

    def __init__(self, attack_type: str = "sign_flip", noise_std: float = 10.0):
        self.attack_type = attack_type
        self.noise_std = noise_std

    def poison_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        poisoned = []
        for w in weights:
            if self.attack_type == "sign_flip":
                poisoned.append(-w * 5.0)
            elif self.attack_type == "gaussian":
                poisoned.append(np.random.normal(0, self.noise_std, size=w.shape).astype(np.float32))
            else:
                poisoned.append(w)
        return poisoned


class GradientPoisoningAttack:
    """Simulates adversarial gradient scaling / direction flipping."""

    def __init__(self, scale_factor: float = -10.0):
        self.scale_factor = scale_factor

    def poison_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        return [g * self.scale_factor for g in gradients]


class BackdoorAttack:
    """Simulates backdoor trigger pattern injection into medical volumetric images."""

    def __init__(self, trigger_val: float = 5.0):
        self.trigger_val = trigger_val

    def inject_trigger(self, image: np.ndarray) -> np.ndarray:
        corrupted = image.copy()
        # Add 4x4 trigger pattern to top-left corner
        corrupted[..., :4, :4] = self.trigger_val
        return corrupted


class SybilAttack:
    """Simulates multiple collusion Sybil nodes sending duplicate malicious payloads."""

    def __init__(self, num_sybils: int = 3):
        self.num_sybils = num_sybils

    def replicate_malicious_payload(self, malicious_weights: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [malicious_weights for _ in range(self.num_sybils)]
