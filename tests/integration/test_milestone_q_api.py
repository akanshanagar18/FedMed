"""
Module: tests.integration.test_milestone_q_api

Purpose:
Integration test suite for Milestone Q REST APIs (/personalization, /continual, /foundation, /explainability).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_personalization_endpoint():
    resp = client.get("/api/v1/personalization")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "supported_strategies" in data
    assert "personalization_gain" in data


def test_continual_endpoint():
    resp = client.get("/api/v1/continual")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "ewc_lambda" in data
    assert "forgetting_score" in data


def test_foundation_endpoint():
    resp = client.get("/api/v1/foundation")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "supported_models" in data
    assert "trainable_percent" in data


def test_explainability_endpoint():
    resp = client.get("/api/v1/explainability")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "explainability_methods" in data
    assert "prediction_confidence" in data
