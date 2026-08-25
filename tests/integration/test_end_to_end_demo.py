"""
Test: tests/integration/test_end_to_end_demo.py

Purpose:
End-to-End integration test for FedMed OS Phase 11.
Tests FastAPI backend startup, model hash verification, demo case discovery,
inference execution, and multi-planar slice generation.
"""

from fastapi.testclient import TestClient
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_ROOT = PROJECT_ROOT / "dashboard" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dashboard.backend.app.main import app

client = TestClient(app)


def test_system_health():
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "components" in data["data"]


def test_production_model_endpoint():
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["checkpoint_sha256"] == "f6cd18dc5e05595ca88ad1675190aabb98595ee9f6d6dfed36c25b596f652f3a"
    assert data["parameter_count"] == 4810074
    assert data["privacy_guarantee"]["epsilon"] == 2.8934


def test_available_demo_cases():
    response = client.get("/api/v1/inference/cases")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_cases"] >= 1
    assert any(c["patient_id"] == "BraTS-GLI-00005-100" for c in data["cases"])


def test_run_inference_demo_case():
    payload = {
        "patient_id": "BraTS-GLI-00005-100",
        "model_version": "v2.1.0-dp-prod"
    }
    response = client.post("/api/v1/inference/predict", json=payload)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "SUCCESS"
    assert data["patient_id"] == "BraTS-GLI-00005-100"
    assert "visual_slices" in data
    assert "axial" in data["visual_slices"]
    assert "coronal" in data["visual_slices"]
    assert "sagittal" in data["visual_slices"]
    assert len(data["visual_slices"]["axial"]) > 100
    assert "tumor_volumes_mm3" in data
    assert data["tumor_volumes_mm3"]["tumor_core_tc"] > 0
    assert data["model_metadata"]["checkpoint_sha256"] == "f6cd18dc5e05595ca88ad1675190aabb98595ee9f6d6dfed36c25b596f652f3a"
