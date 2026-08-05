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
    Benchmark,
    BenchmarkStatus,
    BenchmarkMatrixConfig,
    LeaderboardEntry,
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
    """Verify Experiment v2.0 research metadata container."""
    exp = Experiment(
        experiment_id="brats_2021_v1",
        name="BraTS Segmentation Experiment",
        description="Testing 3D UNet with FedAvg",
        status=ExperimentStatus.RUNNING,
        strategy_name="FedAvg",
        learning_rate=1e-4,
        batch_size=2,
        num_rounds=5,
        dp_enabled=False,
        he_enabled=True,
    )
    assert exp.experiment_id == "brats_2021_v1"
    assert exp.status == ExperimentStatus.RUNNING
    assert exp.learning_rate == 1e-4
    assert exp.num_rounds == 5
    assert exp.he_enabled is True


@pytest.mark.unit
def test_benchmark_schema():
    """Verify Benchmark v2.0 suite container and leaderboard schemas."""
    bm = Benchmark(
        benchmark_id="bm_pytest_001",
        name="FedAvg vs FedProx Benchmark",
        description="Evaluating statistical heterogeneity",
        status=BenchmarkStatus.RUNNING,
        total_experiments=6,
        completed_experiments=2,
    )
    assert bm.benchmark_id == "bm_pytest_001"
    assert bm.status == BenchmarkStatus.RUNNING
    assert bm.total_experiments == 6
    assert len(bm.matrix_config.strategies) == 2

    entry = LeaderboardEntry(
        rank=1,
        experiment_id="exp_001",
        strategy_name="FedProx",
        partition_strategy="NonIID(alpha=0.2)",
        seed=42,
        best_dice=0.885,
        avg_loss=0.312,
        convergence_round=3,
        runtime_sec=42.1,
        status="completed",
    )
    assert entry.rank == 1
    assert entry.best_dice == 0.885


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
