"""
Module: tests.integration.test_health_api

Purpose:
Integration test suite for /api/v1/system/health and /metrics endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_system_health_endpoint():
    resp = client.get("/api/v1/system/health")
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert data["status"] == "ok"
    assert "components" in data
    assert "hospitals" in data
    assert "resources" in data
    assert "crypto_features" in data


def test_prometheus_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text

    assert "fedmed_system_cpu_percent" in text
    assert "fedmed_system_ram_percent" in text
