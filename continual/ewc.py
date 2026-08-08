"""
Module: continual.ewc

Purpose:
Elastic Weight Consolidation (EWC, Kirkpatrick et al., PNAS 2017).
Computes diagonal Fisher Information Matrix (F_i) to penalize parameter updates on critical weights:
L_EWC = L_current + (lambda / 2) * sum F_i * (theta_i - theta_{old, i})^2
"""

from typing import Dict, List, Optional
import numpy as np


class EWC:
    """
    Elastic Weight Consolidation engine calculating Fisher Information penalties.
    """

    def __init__(self, ewc_lambda: float = 400.0):
        self.ewc_lambda = float(ewc_lambda)
        self.optimal_params: Optional[List[np.ndarray]] = None
        self.fisher_matrix: Optional[List[np.ndarray]] = None

    def update_fisher_matrix(self, model_params: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """
        Updates diagonal Fisher Information Matrix F_i = E[ (grad L)^2 ].
        """
        self.optimal_params = [np.array(p, dtype=np.float32) for p in model_params]

        if self.fisher_matrix is None:
            self.fisher_matrix = [np.square(g) for g in gradients]
        else:
            for i in range(len(gradients)):
                self.fisher_matrix[i] = 0.9 * self.fisher_matrix[i] + 0.1 * np.square(gradients[i])

    def compute_penalty(self, current_params: List[np.ndarray]) -> float:
        """
        Computes quadratic EWC penalty loss: (lambda / 2) * sum F_i * (theta_i - theta_{old, i})^2
        """
        if self.optimal_params is None or self.fisher_matrix is None:
            return 0.0

        penalty = 0.0
        for i in range(len(current_params)):
            diff = current_params[i] - self.optimal_params[i]
            penalty += np.sum(self.fisher_matrix[i] * np.square(diff))

        return float((self.ewc_lambda / 2.0) * penalty)
