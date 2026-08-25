"""
Unit tests for server.strategies.fedavg FedAvg strategy.
Verifies sample-weighted parameter averaging and evaluation aggregation.
"""

import numpy as np
import pytest
from server.strategies.base import EvaluateResult, FitResult
from server.strategies.fedavg import FedAvg


def test_fedavg_metadata():
    """Verify FedAvg strategy metadata."""
    strategy = FedAvg()
    meta = strategy.get_metadata()
    assert meta.name == "FedAvg"
    assert "https://arxiv.org/abs/1602.05629" in meta.research_reference


def test_fedavg_aggregate_fit_weighted_averaging():
    """Verify exact numerical weighted parameter averaging across clients."""
    strategy = FedAvg()

    # Client 1: 100 samples, layer weights = [ [1.0, 2.0] ]
    client1_params = [np.array([1.0, 2.0], dtype=np.float32)]
    res1 = FitResult(
        parameters=client1_params,
        num_examples=100,
        metrics={"training_loss": 0.4, "dice_score": 0.8},
        cid="c1",
    )

    # Client 2: 300 samples, layer weights = [ [3.0, 6.0] ]
    client2_params = [np.array([3.0, 6.0], dtype=np.float32)]
    res2 = FitResult(
        parameters=client2_params,
        num_examples=300,
        metrics={"training_loss": 0.2, "dice_score": 0.9},
        cid="c2",
    )

    aggregated_params, metrics = strategy.aggregate_fit(
        server_round=1,
        results=[res1, res2],
        failures=[],
    )

    assert aggregated_params is not None
    # Expected weighted avg: (100 * [1,2] + 300 * [3,6]) / 400 = ( [100,200] + [900,1800] ) / 400 = [2.5, 5.0]
    expected = np.array([2.5, 5.0], dtype=np.float32)
    np.testing.assert_allclose(aggregated_params[0], expected, rtol=1e-5)

    # Expected metrics: loss = (100*0.4 + 300*0.2)/400 = (40 + 60)/400 = 0.25
    # Expected dice = (100*0.8 + 300*0.9)/400 = (80 + 270)/400 = 0.875
    assert pytest.approx(metrics["training_loss"], 1e-5) == 0.25
    assert pytest.approx(metrics["dice_score"], 1e-5) == 0.875
    assert metrics["num_clients"] == 2
    assert metrics["total_samples"] == 400


def test_fedavg_aggregate_evaluate():
    """Verify validation evaluation aggregation."""
    strategy = FedAvg()
    res1 = EvaluateResult(loss=0.5, num_examples=50, metrics={"dice_score": 0.75})
    res2 = EvaluateResult(loss=0.3, num_examples=150, metrics={"dice_score": 0.85})

    loss, metrics = strategy.aggregate_evaluate(
        server_round=1,
        results=[res1, res2],
        failures=[],
    )

    # Expected loss: (50*0.5 + 150*0.3)/200 = (25 + 45)/200 = 0.35
    assert pytest.approx(loss, 1e-5) == 0.35
    assert pytest.approx(metrics["dice_score"], 1e-5) == 0.825
