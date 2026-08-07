"""
Unit tests for server.strategies.fedprox FedProx strategy.
Verifies proximal regularization hyperparameter support and weighted parameter aggregation.
"""

import numpy as np
import pytest
from server.strategies.base import EvaluateResult, FitResult
from server.strategies.fedprox import FedProx


def test_fedprox_metadata_and_params():
    """Verify FedProx strategy metadata and configurable proximal_mu parameter."""
    strategy = FedProx(proximal_mu=0.05)
    meta = strategy.get_metadata()
    assert meta.name == "FedProx"
    assert "https://arxiv.org/abs/1812.06127" in meta.research_reference
    assert strategy.proximal_mu == 0.05


def test_fedprox_aggregate_fit_with_proximal_mu_tracking():
    """Verify FedProx parameter aggregation tracks proximal_mu in returned metrics."""
    strategy = FedProx(proximal_mu=0.02)

    params1 = [np.array([1.0, 1.0], dtype=np.float32)]
    res1 = FitResult(
        parameters=params1,
        num_examples=100,
        metrics={"training_loss": 0.5, "dice_score": 0.8},
    )

    params2 = [np.array([3.0, 3.0], dtype=np.float32)]
    res2 = FitResult(
        parameters=params2,
        num_examples=100,
        metrics={"training_loss": 0.3, "dice_score": 0.9},
    )

    aggregated_params, metrics = strategy.aggregate_fit(
        server_round=1,
        results=[res1, res2],
        failures=[],
    )

    assert aggregated_params is not None
    expected = np.array([2.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(aggregated_params[0], expected, rtol=1e-5)

    assert metrics["proximal_mu"] == 0.02
    assert pytest.approx(metrics["training_loss"], 1e-5) == 0.4
    assert pytest.approx(metrics["dice_score"], 1e-5) == 0.85
