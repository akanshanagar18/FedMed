"""
Module: tests.unit.test_fedasync

Purpose:
Unit test suite for FedAsync strategy (Xie et al., 2019).
"""

import numpy as np
import pytest

from server.strategies.base import FitResult
from server.strategies.fedasync import FedAsync


def test_fedasync_initialization_and_metadata():
    strategy = FedAsync(alpha=0.5, staleness_func="polynomial", a=0.5, max_staleness=10)
    meta = strategy.get_metadata()
    assert meta.name == "FedAsync"
    assert "asynchronous_aggregation" in meta.supported_features


def test_fedasync_asynchronous_aggregation():
    strategy = FedAsync(alpha=0.5, staleness_func="polynomial", max_staleness=10)
    w0 = [np.array([1.0, 2.0], dtype=np.float32)]
    strategy.global_parameters = w0
    strategy.server_timestamp = 5

    # Arriving client with timestamp 3 (staleness = 2)
    w_client = [np.array([1.2, 1.8], dtype=np.float32)]
    res = FitResult(parameters=w_client, num_examples=10, metrics={"client_timestamp": 3, "training_loss": 0.25, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=5, results=[res], failures=[])

    assert agg_params is not None
    assert strategy.accepted_updates == 1
    assert "average_staleness" in metrics
    assert metrics["average_staleness"] == 2.0


def test_fedasync_excessive_staleness_rejection():
    strategy = FedAsync(alpha=0.5, max_staleness=2)
    w0 = [np.array([1.0, 2.0], dtype=np.float32)]
    strategy.global_parameters = w0
    strategy.server_timestamp = 10

    # Arriving client with timestamp 1 (staleness = 9 > max_staleness=2)
    w_client = [np.array([1.2, 1.8], dtype=np.float32)]
    res = FitResult(parameters=w_client, num_examples=10, metrics={"client_timestamp": 1, "training_loss": 0.25})

    agg_params, metrics = strategy.aggregate_fit(server_round=10, results=[res], failures=[])

    assert strategy.rejected_updates == 1
    assert metrics["stale_update_ratio"] == 1.0
