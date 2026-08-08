"""
Module: tests.integration.test_governance_api

Purpose:
Integration test suite for Milestone R REST API endpoints (/api/v1/governance/drift, /sla, /hpo, /promote, /overview).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_governance_overview_endpoint():
    resp = client.get("/api/v1/governance/overview")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "platform_version" in data
    assert "supported_governance_features" in data


def test_drift_evaluate_endpoint():
    payload = {
        "node_id": "hospital_alpha",
        "significance_level": 0.05,
        "psi_threshold": 0.2,
    }
    resp = client.post("/api/v1/governance/drift/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["node_id"] == "hospital_alpha"
    assert "metrics" in data
    assert "mmd" in data["metrics"]


def test_sla_audit_endpoint():
    payload = {
        "run_id": "run_test_integration_001",
        "epsilon_consumed": 1.8,
        "delta_consumed": 1e-5,
        "participating_nodes": ["hospital_alpha", "hospital_beta"],
        "total_nodes": ["hospital_alpha", "hospital_beta"],
        "avg_latency_ms": 110.0,
        "encryption_scheme": "TenSEAL_CKKS",
    }
    resp = client.post("/api/v1/governance/sla/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["overall_compliant"] is True
    assert "certificate_hash" in data


def test_hpo_study_endpoint():
    payload = {
        "study_id": "hpo_test_001",
        "num_initial_configs": 4,
        "max_rounds": 6,
    }
    resp = client.post("/api/v1/governance/hpo/study", json=payload)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["study_id"] == "hpo_test_001"
    assert "best_config" in data
    assert "best_val_dice" in data


def test_model_promote_endpoint():
    payload = {
        "model_id": "brats_monai_3d_unet",
        "current_stage": "Staging",
        "target_stage": "Production",
        "candidate_metrics": {"mean_dice": 0.87, "hd95": 3.6},
        "production_baseline_metrics": {"mean_dice": 0.82, "hd95": 4.5},
    }
    resp = client.post("/api/v1/governance/model/promote", json=payload)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["success"] is True
    assert data["new_stage"] == "Production"
