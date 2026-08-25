"""
Module: tests.unit.test_advanced_strategies

Purpose:
Unit test suite for Milestone O advanced federated optimization strategies
(FedAdam, FedYogi, FedAdagrad, FedNova, FedDyn, FedBN).
"""

import numpy as np
import pytest

from server.strategies.base import FitResult
from server.strategies.registry import StrategyRegistry
from server.strategies.fedadam import FedAdam
from server.strategies.fedyogi import FedYogi
from server.strategies.fedadagrad import FedAdagrad
from server.strategies.fednova import FedNova
from server.strategies.feddyn import FedDyn
from server.strategies.fedbn import FedBN


def test_strategy_registration_and_metadata():
    for name in ["FedAdam", "FedYogi", "FedAdagrad", "FedNova", "FedDyn", "FedBN"]:
        strategy = StrategyRegistry.create(name)
        assert strategy is not None
        meta = strategy.get_metadata()
        assert meta.name.lower() == name.lower()


def test_fedadam_aggregation():
    strategy = FedAdam(eta=0.01, min_fit_clients=2)
    w0 = [np.array([1.0, 2.0], dtype=np.float32)]
    strategy.global_parameters = w0

    w1 = [np.array([1.1, 2.1], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])

    assert agg_params is not None
    assert "server_momentum_norm" in metrics
    assert "server_variance_norm" in metrics


def test_fedyogi_aggregation():
    strategy = FedYogi(eta=0.01, min_fit_clients=2)
    w0 = [np.array([1.0, 2.0], dtype=np.float32)]
    strategy.global_parameters = w0

    w1 = [np.array([1.1, 2.1], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])
    assert agg_params is not None
    assert "server_momentum_norm" in metrics


def test_fedadagrad_aggregation():
    strategy = FedAdagrad(eta=0.1, min_fit_clients=2)
    w0 = [np.array([1.0, 2.0], dtype=np.float32)]
    strategy.global_parameters = w0

    w1 = [np.array([1.1, 2.1], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])
    assert agg_params is not None
    assert "server_accumulator_norm" in metrics


def test_fednova_aggregation():
    strategy = FedNova(min_fit_clients=2)
    # round 1 init
    w1 = [np.array([1.0, 2.0], dtype=np.float32)]
    res1 = FitResult(parameters=w1, num_examples=10, metrics={"local_steps": 1.0})
    res2 = FitResult(parameters=w1, num_examples=10, metrics={"local_steps": 1.0})
    strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])

    # round 2 aggregation
    w1_r2 = [np.array([1.1, 2.1], dtype=np.float32)]
    w2_r2 = [np.array([0.9, 1.9], dtype=np.float32)]
    res1_r2 = FitResult(parameters=w1_r2, num_examples=10, metrics={"local_steps": 2.0})
    res2_r2 = FitResult(parameters=w2_r2, num_examples=10, metrics={"local_steps": 1.0})

    agg_params, metrics = strategy.aggregate_fit(server_round=2, results=[res1_r2, res2_r2], failures=[])
    assert agg_params is not None
    assert "tau_eff" in metrics


def test_feddyn_aggregation():
    strategy = FedDyn(alpha=0.01, min_fit_clients=2)
    w0 = [np.array([1.0, 2.0], dtype=np.float32)]
    strategy.global_parameters = w0

    w1 = [np.array([1.1, 2.1], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])
    assert agg_params is not None
    assert "server_state_norm" in metrics


def test_fedbn_aggregation():
    strategy = FedBN(min_fit_clients=2)
    w1 = [np.array([1.0, 2.0], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32)]

    res1 = FitResult(parameters=w1, num_examples=10, metrics={"training_loss": 0.3, "dice_score": 0.85})
    res2 = FitResult(parameters=w2, num_examples=10, metrics={"training_loss": 0.2, "dice_score": 0.88})

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])
    assert agg_params is not None
    assert metrics["local_bn_active"] is True
