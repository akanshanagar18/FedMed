"""
Module: tests.integration.test_research_api

Purpose:
Integration tests for Milestone J REST API endpoints (/api/v1/mlflow, /api/v1/checkpoints, /api/v1/artifacts, /api/v1/tensorboard, /api/v1/system).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_mlflow_api_endpoints():
    response = client.get("/api/v1/mlflow/status")
    assert response.status_code == 200
    data = response.json()
    assert "tracking_uri" in data["data"]

    runs_resp = client.get("/api/v1/mlflow/runs")
    assert runs_resp.status_code == 200


def test_checkpoints_api_endpoints():
    response = client.get("/api/v1/checkpoints")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["data"], list)


def test_artifacts_api_endpoints():
    response = client.get("/api/v1/artifacts")
    assert response.status_code == 200
    data = response.json()
    assert "artifacts" in data["data"]


def test_tensorboard_api_endpoints():
    response = client.get("/api/v1/tensorboard/status")
    assert response.status_code == 200
    data = response.json()
    assert "launch_command" in data["data"]


def test_system_api_endpoints():
    response = client.get("/api/v1/system/reproducibility")
    assert response.status_code == 200
    data = response.json()
    assert "git" in data["data"]
    assert "hardware" in data["data"]

    env_resp = client.get("/api/v1/system/environment")
    assert env_resp.status_code == 200
