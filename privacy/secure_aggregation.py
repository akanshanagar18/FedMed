"""
Module: privacy.secure_aggregation

Purpose:
Pairwise additive secret masking secure aggregation simulation (Bonawitz et al., CCS 2017).
"""

from typing import List
import numpy as np


class SecureAggregationProtocol:
    """Simulates pairwise mask cancellation for privacy-preserving aggregation."""

    def __init__(self, num_clients: int = 3):
        self.num_clients = num_clients

    def mask_and_aggregate(self, client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Applies pairwise masks, then sums — masks cancel to zero."""
        n = len(client_weights)
        masked = [[np.array(p, dtype=np.float32) for p in w] for w in client_weights]

        for i in range(n):
            for j in range(i + 1, n):
                pair_mask = [np.random.normal(0, 1.0, size=p.shape).astype(np.float32) for p in client_weights[i]]
                for k in range(len(masked[i])):
                    masked[i][k] += pair_mask[k]
                    masked[j][k] -= pair_mask[k]

        num_layers = len(masked[0])
        aggregated = [np.zeros_like(masked[0][k], dtype=np.float32) for k in range(num_layers)]
        for w in masked:
            for k in range(num_layers):
                aggregated[k] += w[k] / n
        return aggregated
