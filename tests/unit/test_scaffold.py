"""
Module: tests.unit.test_scaffold

Purpose:
Unit tests for SCAFFOLD strategy, control variate mathematics, client drift calculation, and state serialization.
"""

import numpy as np
import pytest
from server.strategies.base import FitResult
from server.strategies.registry import StrategyRegistry
from server.strategies.scaffold import SCAFFOLD


def test_scaffold_registration_and_metadata():
    strategy = StrategyRegistry.create("SCAFFOLD")
    assert isinstance(strategy, SCAFFOLD)
    meta = strategy.get_metadata()
    assert meta.name == "SCAFFOLD"
    assert "control_variates" in meta.supported_features
    assert meta.default_parameters["algorithm_name"] == "SCAFFOLD"


def test_scaffold_control_variate_initialization_and_aggregation():
    strategy = SCAFFOLD(min_fit_clients=2, min_available_clients=2)

    w1 = [np.array([1.0, 2.0], dtype=np.float32), np.array([0.5], dtype=np.float32)]
    w2 = [np.array([1.2, 1.8], dtype=np.float32), np.array([0.7], dtype=np.float32)]

    c_delta1 = [np.array([0.1, -0.1], dtype=np.float32), np.array([0.05], dtype=np.float32)]
    c_delta2 = [np.array([-0.05, 0.05], dtype=np.float32), np.array([-0.02], dtype=np.float32)]

    res1 = FitResult(
        parameters=w1,
        num_examples=10,
        metrics={"training_loss": 0.3, "dice_score": 0.85, "control_variate_delta": [d.tolist() for d in c_delta1]},
        cid="hospital_alpha",
    )
    res2 = FitResult(
        parameters=w2,
        num_examples=10,
        metrics={"training_loss": 0.2, "dice_score": 0.88, "control_variate_delta": [d.tolist() for d in c_delta2]},
        cid="hospital_beta",
    )

    agg_params, metrics = strategy.aggregate_fit(server_round=1, results=[res1, res2], failures=[])

    assert agg_params is not None
    assert len(agg_params) == 2
    assert np.allclose(agg_params[0], np.array([1.1, 1.9], dtype=np.float32))

    assert "client_drift" in metrics
    assert "control_variate_norm" in metrics
    assert metrics["client_drift"] >= 0.0
    assert metrics["control_variate_norm"] >= 0.0


def test_scaffold_state_serialization():
    strategy = SCAFFOLD()
    strategy._init_control_variates([np.array([1.0, 2.0], dtype=np.float32)])

    state = strategy.serialize_state()
    assert "server_control_variates" in state
    assert state["server_control_variates"] is not None

    new_strategy = SCAFFOLD()
    new_strategy.load_state(state)
    assert new_strategy.server_control_variates is not None
    assert np.array_equal(new_strategy.server_control_variates[0], np.array([0.0, 0.0], dtype=np.float32))
