"""
Module: tests.integration.test_enterprise_platform_api

Purpose:
Integration test suite for Milestone T REST API endpoints (/api/v1/enterprise/*).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_workflows_api():
    # Instantiate
    resp = client.post("/api/v1/enterprise/workflows/instantiate", json={"workflow_name": "Enterprise_EndToEnd_FL_Pipeline"})
    assert resp.status_code == 200
    inst_id = resp.json()["data"]["instance_id"]

    # Step
    step_resp = client.post(f"/api/v1/enterprise/workflows/{inst_id}/step")
    assert step_resp.status_code == 200

    # Pause & Resume
    pause_resp = client.post(f"/api/v1/enterprise/workflows/{inst_id}/pause")
    assert pause_resp.status_code == 200
    resume_resp = client.post(f"/api/v1/enterprise/workflows/{inst_id}/resume")
    assert resume_resp.status_code == 200

    # Run complete
    run_resp = client.post(f"/api/v1/enterprise/workflows/{inst_id}/run")
    assert run_resp.status_code == 200


def test_simulator_api():
    resp = client.post("/api/v1/enterprise/simulator/trigger", json={"scenario_type": "HOSPITAL_FAILURE", "target_node": "hospital_beta"})
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "EXECUTED"

    hist_resp = client.get("/api/v1/enterprise/simulator/history")
    assert hist_resp.status_code == 200


def test_policies_api():
    resp = client.get("/api/v1/enterprise/policies")
    assert resp.status_code == 200
    assert "policies" in resp.json()["data"]

    reload_resp = client.post("/api/v1/enterprise/policies/reload")
    assert reload_resp.status_code == 200


def test_scheduler_api():
    resp = client.post("/api/v1/enterprise/scheduler/jobs", json={"job_type": "DRIFT_SCAN", "interval_seconds": 300})
    assert resp.status_code == 200
    job_id = resp.json()["data"]["job_id"]

    exec_resp = client.post(f"/api/v1/enterprise/scheduler/jobs/{job_id}/execute")
    assert exec_resp.status_code == 200


def test_digital_twin_api():
    resp = client.post("/api/v1/enterprise/digital-twin/predict", json={
        "scenario_description": "Network latency doubles",
        "baseline_dice": 0.865,
        "latency_multiplier": 2.0,
    })
    assert resp.status_code == 200
    assert "predicted_metrics" in resp.json()["data"]


def test_lifecycle_api():
    resp = client.post("/api/v1/enterprise/lifecycle/create", json={"name": "BraTS 3D U-Net Benchmark", "owner": "mri_team"})
    assert resp.status_code == 200
    exp_id = resp.json()["data"]["experiment_id"]

    trans_resp = client.post(f"/api/v1/enterprise/lifecycle/{exp_id}/transition", json={"target_stage": "TRAINING", "notes": "Started FL"})
    assert trans_resp.status_code == 200


def test_persistent_graph_time_travel_api():
    resp = client.get("/api/v1/enterprise/persistent-graph/time-travel")
    assert resp.status_code == 200
    assert "total_nodes" in resp.json()["data"]
