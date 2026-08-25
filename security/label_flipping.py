"""
Module: security.label_flipping

Purpose:
Label Flipping attack simulation for robustness evaluation.
"""

import numpy as np
from typing import Optional


class LabelFlippingAttack:
    """Simulates label flipping on a fraction of client data."""

    def __init__(self, source_class: int = 0, target_class: int = 1, flip_ratio: float = 1.0):
        self.source_class = source_class
        self.target_class = target_class
        self.flip_ratio = float(flip_ratio)

    def apply(self, labels: np.ndarray) -> np.ndarray:
        corrupted = labels.copy()
        mask = labels == self.source_class
        n_flip = int(np.sum(mask) * self.flip_ratio)
        indices = np.where(mask)[0][:n_flip]
        corrupted[indices] = self.target_class
        return corrupted
