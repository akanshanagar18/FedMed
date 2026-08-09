"""
Module: tests.integration.test_auth_api

Purpose:
Integration test suite for Authentication & RBAC REST APIs (/api/v1/auth/*).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_auth_login_and_me_endpoints():
    login_resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert login_resp.status_code == 200
    token = login_resp.json()["data"]["access_token"]
    assert token is not None

    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["data"]["sub"] == "admin"


def test_auth_audit_logs_endpoint():
    resp = client.get("/api/v1/auth/audit-logs")
    assert resp.status_code == 200
    assert "logs" in resp.json()["data"]
