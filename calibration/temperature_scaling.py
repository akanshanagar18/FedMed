"""
Module: calibration.temperature_scaling

Purpose:
Post-hoc temperature scaling calibration: logits / T.
"""

import numpy as np


class TemperatureScaler:
    """Applies temperature scaling to logit vectors."""

    def __init__(self, temperature: float = 1.5):
        self.temperature = float(temperature)

    def scale(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)
