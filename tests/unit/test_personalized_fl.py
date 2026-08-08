"""
Module: tests.unit.test_personalized_fl

Purpose:
Unit test suite for personalized FL strategies (FedPer, LG-FedAvg, FedRep, Per-FedAvg).
"""

import numpy as np
import pytest

from server.strategies.base import FitResult
from server.strategies.fedper import FedPer
from server.strategies.lgfedavg import LGFedAvg
from server.strategies.fedrep import FedRep
from server.strategies.perfedavg import PerFedAvg


def test_fedper_aggregation():
    strategy = FedPer(num_shared_layers=2, min_fit_clients=2)
    w1 = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32), np.array([5.0], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32), np.array([2.8, 4.2], dtype=np.float32), np.array([9.0], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85, "personalized_dice": 0.90})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88, "personalized_dice": 0.92})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])

    assert agg_params is not None
    assert "personalization_gain" in metrics
    assert metrics["num_shared_layers"] == 2
    # Layer 0 shared averaged: (1.0 + 1.2)/2 = 1.1
    np.testing.assert_allclose(agg_params[0], [1.1, 1.9], atol=1e-5)


def test_lgfedavg_aggregation():
    strategy = LGFedAvg(num_local_layers=1, min_fit_clients=2)
    w1 = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    w2 = [np.array([9.0, 9.0], dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])

    assert agg_params is not None
    assert metrics["num_local_layers"] == 1


def test_fedrep_aggregation():
    strategy = FedRep(representation_layers=2, min_fit_clients=2)
    w1 = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32), np.array([5.0], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32), np.array([2.8, 4.2], dtype=np.float32), np.array([9.0], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])
    assert agg_params is not None
    assert "representation_layers" in metrics


def test_perfedavg_aggregation():
    strategy = PerFedAvg(alpha=0.01, beta=0.001, min_fit_clients=2)
    w1 = [np.array([1.0, 2.0], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])
    assert agg_params is not None
    assert "meta_learning_rate" in metrics
