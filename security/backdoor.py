"""
Module: security.backdoor

Purpose:
Backdoor trigger injection attack simulation.
"""

import numpy as np


class BackdoorAttack:
    """Injects a trigger pattern into medical images."""

    def __init__(self, trigger_value: float = 5.0, trigger_size: int = 4):
        self.trigger_value = float(trigger_value)
        self.trigger_size = int(trigger_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        corrupted = image.copy()
        s = self.trigger_size
        corrupted[..., :s, :s] = self.trigger_value
        return corrupted
