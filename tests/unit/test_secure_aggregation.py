"""
Module: tests.unit.test_secure_aggregation

Purpose:
Unit test suite for SecureAggregationSimulator.
"""

import numpy as np
import pytest
from security.secure_aggregation import SecureAggregationSimulator


def test_secure_aggregation_zero_sum_cancellation():
    sim = SecureAggregationSimulator(num_clients=3)
    w1 = [np.array([1.0, 2.0], dtype=np.float32)]
    w2 = [np.array([3.0, 4.0], dtype=np.float32)]
    w3 = [np.array([5.0, 6.0], dtype=np.float32)]
    client_weights = [w1, w2, w3]

    masked = sim.generate_pairwise_masks(client_weights)
    assert len(masked) == 3

    sec_sum = sim.secure_sum(masked)

    # Clean mean = ([1,2] + [3,4] + [5,6])/3 = [3.0, 4.0]
    expected = [np.array([3.0, 4.0], dtype=np.float32)]
    np.testing.assert_allclose(sec_sum[0], expected[0], atol=1e-5)
