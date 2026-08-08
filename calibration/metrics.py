"""
Module: calibration.metrics

Purpose:
Uncertainty Calibration & Reliability Metrics.
Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score, and Temperature Scaling.
"""

from typing import Dict, List, Tuple
import numpy as np


class CalibrationMetrics:
    """
    Computes ECE, MCE, Brier Score, and Reliability Diagrams.
    """

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def compute_ece_and_brier(self, confidences: np.ndarray, accuracies: np.ndarray) -> Dict[str, float]:
        """
        Computes ECE = sum |acc(B_m) - conf(B_m)| * (|B_m| / N) and Brier Score = (1/N) sum (p - y)^2.
        """
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                diff = abs(accuracy_in_bin - avg_confidence_in_bin)

                ece += diff * prop_in_bin
                mce = max(mce, diff)

        # Brier Score = mean( (confidences - accuracies)^2 )
        brier_score = float(np.mean((confidences - accuracies) ** 2))

        return {
            "expected_calibration_error": float(ece),
            "maximum_calibration_error": float(mce),
            "brier_score": brier_score,
            "well_calibrated": float(ece) < 0.05,
        }

    def apply_temperature_scaling(self, logits: np.ndarray, temperature: float = 1.5) -> np.ndarray:
        """
        Rescales logits z / T to calibrate prediction probabilities.
        """
        scaled_logits = logits / temperature
        exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)
