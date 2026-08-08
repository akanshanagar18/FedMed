"""
Module: explainability.uncertainty

Purpose:
Monte Carlo Dropout epistemic uncertainty estimator and predictive entropy calculator.
"""

from typing import Dict, Tuple
import numpy as np


class UncertaintyEstimator:
    """
    Monte Carlo Dropout & Predictive Entropy Uncertainty Estimator.
    """

    def __init__(self, mc_samples: int = 10):
        self.mc_samples = mc_samples

    def estimate_uncertainty(self, mc_predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        mc_predictions shape: (MC_Samples, Batch, Classes, H, W) or 3D
        Computes epistemic variance and predictive entropy: H(y) = - sum p log p
        """
        # Predictive Mean Probability Map
        mean_pred = np.mean(mc_predictions, axis=0)  # (Batch, Classes, H, W)

        # Epistemic Uncertainty Variance: Var[p]
        variance_map = np.var(mc_predictions, axis=0)  # (Batch, Classes, H, W)

        # Predictive Entropy: H = - sum p * log(p + eps)
        entropy_map = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)  # (Batch, H, W)

        return {
            "mean_prediction": mean_pred,
            "variance_map": variance_map,
            "entropy_map": entropy_map,
        }
