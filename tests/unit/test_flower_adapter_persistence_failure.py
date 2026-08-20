"""
Unit tests for FlowerStrategyAdapter failure-injection and invariant verification.
Verifies that mandatory direct SQLite persistence failure raises RuntimeError
and strictly prevents successful_rounds from incrementing.
"""

import sqlite3
from unittest.mock import MagicMock
import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from server.strategies.adapters.flower_adapter import FlowerStrategyAdapter
from server.strategies.fedavg import FedAvg


def test_flower_adapter_mandatory_persistence_failure_raises_runtime_error(monkeypatch):
    """
    Failure Injection Test:
    1. Instantiates FlowerStrategyAdapter with real FedAvg strategy.
    2. Constructs valid FitRes client results.
    3. Forces db.commit() inside _report_metrics() to raise sqlite3.OperationalError.
    4. Asserts that RuntimeError is raised and propagates out of aggregate_fit().
    5. Asserts that adapter.successful_rounds is NOT incremented (remains 0).
    6. Asserts that the error message identifies mandatory direct SQLite persistence failure.
    7. Verifies that normal aggregation works cleanly when commit is restored.
    """
    strategy = FedAvg()
    adapter = FlowerStrategyAdapter(strategy=strategy, experiment_id="test_exp_fail_inject")
    assert adapter.successful_rounds == 0

    # Build valid client proxies and fit responses
    mock_proxy1 = MagicMock()
    mock_proxy1.cid = "hospital_alpha"

    mock_proxy2 = MagicMock()
    mock_proxy2.cid = "hospital_beta"

    dummy_params = ndarrays_to_parameters([np.array([1.0, 2.0], dtype=np.float32)])
    fit_res1 = FitRes(
        status=Status(Code.OK, ""),
        parameters=dummy_params,
        num_examples=100,
        metrics={"hospital_id": "hospital_alpha", "training_loss": 0.5, "dice_score": 0.8},
    )
    fit_res2 = FitRes(
        status=Status(Code.OK, ""),
        parameters=dummy_params,
        num_examples=100,
        metrics={"hospital_id": "hospital_beta", "training_loss": 0.4, "dice_score": 0.85},
    )

    results = [(mock_proxy1, fit_res1), (mock_proxy2, fit_res2)]

    # Monkeypatch db.commit() in sqlalchemy Session to raise sqlite3.OperationalError
    def failing_commit(self):
        raise sqlite3.OperationalError("forced persistence failure")

    monkeypatch.setattr("sqlalchemy.orm.Session.commit", failing_commit)

    # Calling aggregate_fit MUST raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        adapter.aggregate_fit(server_round=1, results=results, failures=[])

    # Assert exception message contains the failure details
    assert "Mandatory direct SQLite persistence failed for round 1" in str(exc_info.value)
    assert "forced persistence failure" in str(exc_info.value)

    # Crucial Invariant: successful_rounds MUST remain 0
    assert adapter.successful_rounds == 0

    # Restore commit behavior
    monkeypatch.undo()

    # Now verify that normal execution succeeds cleanly
    params, metrics = adapter.aggregate_fit(server_round=1, results=results, failures=[])
    assert params is not None
    assert metrics is not None
    assert adapter.successful_rounds == 1
