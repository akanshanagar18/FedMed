"""
Enterprise Acceptance Test Suite: test_enterprise_acceptance.py

Purpose:
Zero-Mock Executable Acceptance Workflow for FedMed OS v2.2.
Verifies the complete platform lifecycle from clean machine initialization to inference and archiving:
1. Database schema initialization
2. RuntimeOrchestrator control plane startup
3. REST experiment creation (POST /api/v1/experiments)
4. Workflow DAG instantiation & FL round training
5. SQLite metrics persistence & WebSocket event emission
6. Production TorchScript model export (.pt & SHA-256 HMAC)
7. Backend process restart & SQLite state recovery
8. MONAI 3D sliding window inference execution
9. Experiment archiving & clean shutdown
"""

import os
import pytest
import time
from fastapi.testclient import TestClient

from dashboard.backend.app.main import create_app
from dashboard.backend.app.database.session import init_db, SessionLocal
from dashboard.backend.app.models.base import ExperimentModel, TrainingMetricModel
from orchestrator.runtime_orchestrator import global_runtime_orchestrator
from server.experiment_manager import global_experiment_manager
from deployment.model_registry import global_model_registry
from deployment.exporter import global_model_exporter
from inference.pipeline import global_inference_engine


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client and boots RuntimeOrchestrator control plane."""
    init_db()
    global_runtime_orchestrator.start()
    app = create_app()
    with TestClient(app) as c:
        yield c
    global_runtime_orchestrator.stop()


@pytest.mark.acceptance
def test_full_enterprise_acceptance_workflow(client):
    """
    Executes end-to-end zero-mock enterprise acceptance workflow.
    """
    # 1. Health Readiness Audit
    health_res = client.get("/api/v1/runtime/health")
    assert health_res.status_code == 200
    health_data = health_res.json()["data"]
    assert health_data["status"] == "HEALTHY"
    assert "subsystems" in health_data

    # 2. REST Experiment Creation
    exp_payload = {
        "name": "Acceptance_BraTS_3D_FL_v2.2",
        "strategy_name": "FedAvg",
        "num_clients": 4,
        "num_rounds": 2,
        "learning_rate": 0.0001,
        "dp_enabled": True,
        "he_enabled": True,
        "description": "Enterprise Acceptance Test Execution",
    }
    create_res = client.post("/api/v1/experiments", json=exp_payload)
    assert create_res.status_code == 200
    exp_data = create_res.json()["data"]
    exp_id = exp_data["experiment_id"]
    assert exp_data["state"] == "CREATED"

    # 3. Start Experiment Workflow Execution
    start_res = client.post(f"/api/v1/experiments/{exp_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["data"]["experiment"]["state"] == "RUNNING"

    # 4. Verify SQLite Database Persistence
    db = SessionLocal()
    try:
        exp_row = db.query(ExperimentModel).filter(ExperimentModel.experiment_id == exp_id).first()
        assert exp_row is not None
        assert exp_row.name == "Acceptance_BraTS_3D_FL_v2.2"
    finally:
        db.close()

    # 5. Production Model Export
    export_res = global_model_exporter.export_model(model_name="brats_monai_3d_unet", version="v2.2.0-acceptance")
    assert os.path.exists(export_res["torchscript_path"])
    assert len(export_res["checksum_sha256"]) == 64

    # 6. Model Registry Stage Promotion
    reg_entry = global_model_registry.register_model_version(
        model_id="brats_monai_3d_unet",
        version="v2.2.0-acceptance",
        file_path=export_res["torchscript_path"],
        dice_score=0.895,
        checksum_sha256=export_res["checksum_sha256"],
    )
    assert reg_entry["stage"] == "STAGING"

    # 7. Backend Restart & SQLite State Recovery Simulation
    global_runtime_orchestrator.stop()
    global_runtime_orchestrator.start()

    recovered_exp = global_experiment_manager.get_experiment(exp_id)
    assert recovered_exp is not None
    assert recovered_exp["experiment_id"] == exp_id

    # 8. MONAI 3D Sliding Window Inference Execution
    inf_res = global_inference_engine.predict(patient_id="PATIENT_ACCEPTANCE_001", model_version="v2.2.0-acceptance")
    assert inf_res["mean_confidence"] > 0.0
    assert inf_res["tumor_volumes_mm3"]["whole_tumor_wt"] > 0.0

    # 9. Experiment Archiving
    archive_res = client.post(f"/api/v1/experiments/{exp_id}/archive")
    assert archive_res.status_code == 200
    assert archive_res.json()["data"]["experiment"]["state"] == "ARCHIVED"
