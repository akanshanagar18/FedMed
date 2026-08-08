"""
Module: calibration.ece

Purpose:
Expected Calibration Error and Maximum Calibration Error computation.
"""

import numpy as np


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """ECE = sum |acc(B_m) - conf(B_m)| * (|B_m| / N)."""
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop = np.mean(mask)
        if prop > 0:
            ece += abs(np.mean(accuracies[mask]) - np.mean(confidences[mask])) * prop
    return float(ece)


def compute_mce(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """MCE = max |acc(B_m) - conf(B_m)|."""
    bins = np.linspace(0, 1, num_bins + 1)
    mce = 0.0
    for i in range(num_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.sum(mask) > 0:
            mce = max(mce, abs(np.mean(accuracies[mask]) - np.mean(confidences[mask])))
    return float(mce)
