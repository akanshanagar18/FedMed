"""
Module: security.secure_aggregation

Purpose:
Secure Aggregation protocol simulation (Bonawitz et al., CCS 2017).
Simulates pairwise additive secret masking S_{i,j} = -S_{j,i} to protect individual client updates.
"""

from typing import Dict, List, Tuple
import numpy as np


class SecureAggregationSimulator:
    """
    Pairwise Additive Secret Masking Secure Aggregation Simulator.
    """

    def __init__(self, num_clients: int = 3):
        self.num_clients = num_clients

    def generate_pairwise_masks(self, client_weights: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        Generates pairwise additive masks delta_{i,j} such that sum_i mask_i = 0.
        """
        n = len(client_weights)
        masked_weights = [[np.array(p, dtype=np.float32) for p in w] for w in client_weights]

        # Generate pairwise masks S_{i,j} for 0 <= i < j < n
        for i in range(n):
            for j in range(i + 1, n):
                # Pairwise random mask
                pair_masks = [np.random.normal(0, 1.0, size=p.shape).astype(np.float32) for p in client_weights[i]]

                # Client i adds +S_{i,j}
                for k in range(len(masked_weights[i])):
                    masked_weights[i][k] += pair_masks[k]

                # Client j subtracts -S_{i,j}
                for k in range(len(masked_weights[j])):
                    masked_weights[j][k] -= pair_masks[k]

        return masked_weights

    def secure_sum(self, masked_client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Server sums masked client updates: pairwise masks cancel out perfectly: sum_i (w_i + S_i) = sum_i w_i.
        """
        n = len(masked_client_weights)
        aggregated = [np.zeros_like(masked_client_weights[0][k], dtype=np.float32) for k in range(len(masked_client_weights[0]))]

        for w in masked_client_weights:
            for k in range(len(aggregated)):
                aggregated[k] += w[k] / n

        return aggregated
