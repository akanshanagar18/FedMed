"""
Module: calibration.brier_score

Purpose:
Brier Score computation: mean squared error between predicted probabilities and true labels.
"""

import numpy as np


def compute_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Brier Score = (1/N) sum (p - y)^2."""
    return float(np.mean((probabilities - labels) ** 2))
