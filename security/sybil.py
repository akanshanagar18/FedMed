"""
Module: security.sybil

Purpose:
Sybil attack simulation — multiple colluding nodes sending identical malicious payloads.
"""

import numpy as np
from typing import List


class SybilAttack:
    """Replicates a malicious payload across multiple Sybil identities."""

    def __init__(self, num_sybils: int = 3):
        self.num_sybils = int(num_sybils)

    def replicate(self, malicious_weights: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [malicious_weights for _ in range(self.num_sybils)]
