"""
Module: tests.integration.test_inference_api

Purpose:
Integration test suite for 3D MONAI Inference REST APIs (/api/v1/inference/*).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_inference_api_endpoints():
    predict_resp = client.post("/api/v1/inference/predict", json={"patient_id": "TEST_PATIENT_INT_1", "model_version": "v2.0-test"})
    assert predict_resp.status_code == 200
    data = predict_resp.json()["data"]
    assert data["patient_id"] == "TEST_PATIENT_INT_1"

    hist_resp = client.get("/api/v1/inference/history")
    assert hist_resp.status_code == 200
    assert "history" in hist_resp.json()["data"]
