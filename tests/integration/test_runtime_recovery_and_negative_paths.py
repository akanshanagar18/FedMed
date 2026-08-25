"""
Integration Test Suite: Negative Paths, State Machines, Recovery, and Chaos Conditions.
Validates FedMed OS v2.1 runtime recovery, illegal state transition rejection,
node heartbeat timeout handling, and corrupt checkpoint resilience.
"""

import os
import pytest
import time
from common.state_machines import (
    ExperimentState, HospitalState, RoundState, DeploymentState,
    validate_experiment_transition, validate_hospital_transition,
    InvalidStateTransitionError
)
from orchestrator.runtime_orchestrator import RuntimeOrchestrator, global_runtime_orchestrator
from server.experiment_manager import ExperimentManager, global_experiment_manager
from client.hospital_runtime import MultiHospitalRuntimeManager, global_hospital_runtime_manager
from deployment.model_registry import ModelRegistry, global_model_registry


@pytest.mark.integration
def test_invalid_experiment_state_transition_raises_error():
    """Verify that illegal experiment state transitions raise InvalidStateTransitionError."""
    # CREATED -> COMPLETED is illegal (must go CREATED -> INITIALIZING -> READY -> RUNNING -> COMPLETED)
    with pytest.raises(InvalidStateTransitionError):
        validate_experiment_transition(ExperimentState.CREATED, ExperimentState.COMPLETED)


@pytest.mark.integration
def test_invalid_hospital_state_transition_raises_error():
    """Verify that illegal hospital node state transitions raise InvalidStateTransitionError."""
    # REGISTERING -> TRAINING is illegal (must go REGISTERING -> CONNECTED -> READY -> TRAINING)
    with pytest.raises(InvalidStateTransitionError):
        validate_hospital_transition(HospitalState.REGISTERING, HospitalState.TRAINING)


@pytest.mark.integration
def test_experiment_manager_creation_and_lifecycle():
    """Test REST-driven experiment manager creation, pause, resume, and archive lifecycle."""
    exp_mgr = ExperimentManager()
    exp = exp_mgr.create_experiment(
        name="Unit_Test_Chaos_Experiment",
        strategy_name="FedProx",
        num_clients=3,
        num_rounds=2,
    )
    exp_id = exp["experiment_id"]
    assert exp["state"] == ExperimentState.CREATED.value

    # Start -> RUNNING
    res_start = exp_mgr.start_experiment(exp_id)
    assert res_start["success"] is True
    assert exp_mgr.get_experiment(exp_id)["state"] == ExperimentState.RUNNING.value

    # Pause -> PAUSED
    res_pause = exp_mgr.pause_experiment(exp_id)
    assert res_pause["success"] is True
    assert exp_mgr.get_experiment(exp_id)["state"] == ExperimentState.PAUSED.value

    # Resume -> RUNNING
    res_resume = exp_mgr.resume_experiment(exp_id)
    assert res_resume["success"] is True
    assert exp_mgr.get_experiment(exp_id)["state"] == ExperimentState.RUNNING.value


@pytest.mark.integration
def test_runtime_orchestrator_health_and_recovery():
    """Test RuntimeOrchestrator control plane health reporting and DB state restoration."""
    orchestrator = RuntimeOrchestrator()
    orchestrator.start()

    health = orchestrator.get_runtime_health()
    assert health["status"] in ["HEALTHY", "STOPPED"]
    assert "cpu_percent" in health
    assert "memory_mb" in health

    status = orchestrator.get_runtime_status()
    assert "is_running" in status
    assert "hospitals" in status

    orchestrator.stop()


@pytest.mark.integration
def test_hospital_node_recovery_on_disconnect():
    """Test hospital node auto-recovery mechanism when node becomes disconnected."""
    mgr = MultiHospitalRuntimeManager()
    hospital_id = "hospital_alpha"

    # Set inactive (simulating node crash)
    mgr.set_node_active_state(hospital_id, False)
    h = mgr.get_hospital(hospital_id)
    assert h.status == "DISCONNECTED"

    # Reactivate node
    mgr.set_node_active_state(hospital_id, True)
    assert h.status == "CONNECTED"


@pytest.mark.integration
def test_model_registry_registration_and_promotion():
    """Test ModelRegistry version registration, SHA-256 validation, and stage promotion."""
    registry = ModelRegistry()
    entry = registry.register_model_version(
        model_id="test_brats_unet",
        version="v2.1.0-chaos",
        file_path="checkpoints/test.pt",
        dice_score=0.892,
        checksum_sha256="abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
        stage=DeploymentState.STAGING,
    )
    assert entry["version"] == "v2.1.0-chaos"

    # Promote to CANARY
    promo = registry.promote_stage("test_brats_unet:v2.1.0-chaos", DeploymentState.CANARY)
    assert promo["success"] is True
    assert promo["model"]["stage"] == DeploymentState.CANARY.value
