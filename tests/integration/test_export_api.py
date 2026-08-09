"""
Module: tests.integration.test_export_api

Purpose:
Integration test suite for Model Export REST APIs (/api/v1/export/*).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_export_api_endpoints():
    resp = client.post("/api/v1/export/model", json={"model_name": "test_brats_api", "version": "v1.0-test"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["success"] is True
    assert "torchscript_path" in data
