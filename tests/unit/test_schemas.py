"""
Unit tests for common.schemas (Canonical Pydantic Contracts)
"""

import pytest
from datetime import datetime
from common.schemas import (
    TrainingMetric,
    HospitalStatus,
    TrainingRound,
    NodeHealth,
    Experiment,
    SuccessResponse,
    ErrorResponse,
    ConnectionStatus,
    RoundStatus,
    ExperimentStatus,
)


@pytest.mark.unit
def test_training_metric_schema():
    """Verify TrainingMetric serialization and default fields."""
    metric = TrainingMetric(
        experiment_id="exp_test",
        round_number=1,
        training_loss=0.45,
        dice_score=0.82,
        hospital_id="hospital_a",
    )
    assert metric.experiment_id == "exp_test"
    assert metric.round_number == 1
    assert metric.training_loss == 0.45
    assert metric.dice_score == 0.82
    assert metric.hospital_id == "hospital_a"
    assert isinstance(metric.timestamp, datetime)

    # Test dictionary export
    dumped = metric.model_dump()
    assert dumped["experiment_id"] == "exp_test"
    assert dumped["round_number"] == 1


@pytest.mark.unit
def test_hospital_status_schema():
    """Verify HospitalStatus enum serialization and defaults."""
    status = HospitalStatus(
        hospital_id="hospital_alpha",
        name="General Hospital Alpha",
        connection_status=ConnectionStatus.TRAINING,
        client_latency_ms=12,
    )
    assert status.hospital_id == "hospital_alpha"
    assert status.connection_status == ConnectionStatus.TRAINING
    assert status.client_latency_ms == 12


@pytest.mark.unit
def test_training_round_schema():
    """Verify TrainingRound status transition and list attributes."""
    round_state = TrainingRound(
        round_number=2,
        status=RoundStatus.IN_PROGRESS,
        participating_hospitals=["hospital_a", "hospital_b"],
        aggregation_time_seconds=0.45,
    )
    assert round_state.round_number == 2
    assert round_state.status == RoundStatus.IN_PROGRESS
    assert len(round_state.participating_hospitals) == 2


@pytest.mark.unit
def test_node_health_schema():
    """Verify NodeHealth response defaults."""
    health = NodeHealth(status="ok", active_connections=3, uptime_seconds=120)
    assert health.status == "ok"
    assert health.active_connections == 3
    assert health.uptime_seconds == 120


@pytest.mark.unit
def test_experiment_schema():
    """Verify Experiment metadata container."""
    exp = Experiment(
        experiment_id="brats_2021_v1",
        name="BraTS Segmentation Experiment",
        description="Testing 3D UNet with FedAvg",
        status=ExperimentStatus.RUNNING,
        hyperparameters={"lr": 1e-4, "batch_size": 2},
    )
    assert exp.experiment_id == "brats_2021_v1"
    assert exp.status == ExperimentStatus.RUNNING
    assert exp.hyperparameters["batch_size"] == 2


@pytest.mark.unit
def test_response_envelopes():
    """Verify SuccessResponse and ErrorResponse standard API envelopes."""
    success = SuccessResponse(message="Operation successful", data={"key": "value"})
    assert success.message == "Operation successful"
    assert success.data["key"] == "value"

    error = ErrorResponse(
        status_code=400,
        error_code="INVALID_PAYLOAD",
        message="Bad Request payload",
        developer_details="Field 'round_number' missing",
    )
    assert error.status_code == 400
    assert error.error_code == "INVALID_PAYLOAD"
