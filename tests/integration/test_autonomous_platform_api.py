"""
Module: tests.integration.test_autonomous_platform_api

Purpose:
Integration test suite for Autonomous Operating System REST endpoints (/api/v1/autonomous/*).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_autonomous_decisions_endpoint():
    resp = client.get("/api/v1/autonomous/decisions")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "decisions" in data


def test_autonomous_orchestrate_endpoint():
    payload = {
        "current_round": 3,
        "training_loss": 0.19,
        "validation_dice": 0.86,
        "drift_mmd": 0.04,
        "sla_compliant": True,
        "active_nodes_count": 2,
        "total_nodes_count": 2,
        "privacy_epsilon": 2.0,
        "candidate_dice": None,
    }

    resp = client.post("/api/v1/autonomous/orchestrate", json=payload)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["decision"] == "CONTINUE_TRAINING"


def test_adaptive_strategy_recommend_endpoint():
    payload = {
        "current_strategy": "FedAvg",
        "participation_rate": 0.50,
        "avg_latency_ms": 120.0,
        "gradient_divergence": 0.05,
        "drift_mmd": 0.02,
        "privacy_epsilon": 2.0,
        "node_dropouts": 1,
    }
    resp = client.post("/api/v1/autonomous/adaptive-strategy/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["recommended_strategy"] == "FedNova"


def test_rca_diagnose_endpoint():
    payload = {
        "current_loss": 0.25,
        "previous_loss": 0.18,
        "current_dice": 0.79,
        "previous_dice": 0.85,
        "drift_mmd": 0.15,
        "active_clients": 1,
        "total_clients": 2,
        "dp_epsilon": 8.5,
        "avg_latency_ms": 2200.0,
    }
    resp = client.post("/api/v1/autonomous/rca/diagnose", json=payload)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["has_degradation"] is True
    assert data["primary_cause"] == "CLIENT_DROPOUT"


def test_recommendations_endpoint():
    resp = client.get("/api/v1/autonomous/recommendations")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "recommendations" in data


def test_experiment_planner_propose_endpoint():
    resp = client.get("/api/v1/autonomous/experiment-planner/propose")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "proposals" in data


def test_self_healing_recover_endpoint():
    payload = {
        "node_id": "hospital_beta",
        "failure_reason": "Heartbeat timeout",
    }
    resp = client.post("/api/v1/autonomous/self-healing/recover", json=payload)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["node_id"] == "hospital_beta"
    assert data["success"] is True


def test_deployments_endpoints():
    # Initiate
    payload = {
        "model_id": "brats_unet_v2",
        "version": "v2.1.0",
        "strategy": "CANARY",
        "candidate_metrics": {"mean_dice": 0.88, "hd95": 3.4},
    }
    resp = client.post("/api/v1/autonomous/deployments/initiate", json=payload)
    assert resp.status_code == 200
    dep = resp.json()["data"]
    dep_id = dep["deployment_id"]

    # List
    list_resp = client.get("/api/v1/autonomous/deployments")
    assert list_resp.status_code == 200

    # Promote
    promo_resp = client.post(f"/api/v1/autonomous/deployments/{dep_id}/promote")
    assert promo_resp.status_code == 200


def test_knowledge_graph_endpoints():
    resp = client.get("/api/v1/autonomous/knowledge-graph/summary")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "total_nodes" in data


def test_executive_reports_endpoint():
    resp = client.get("/api/v1/autonomous/executive-reports/generate")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "kpis" in data
    assert "markdown_report" in data
