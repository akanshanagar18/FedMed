"""
Module: tests.integration.test_scaffold_api

Purpose:
Integration tests for SCAFFOLD REST API strategy endpoints (/api/v1/strategies/scaffold).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_scaffold_strategy_api_endpoints():
    # Test strategy listing includes SCAFFOLD
    list_resp = client.get("/api/v1/strategies")
    assert list_resp.status_code == 200
    data = list_resp.json()["data"]
    assert "SCAFFOLD" in data["strategies"]

    # Test individual SCAFFOLD endpoint
    detail_resp = client.get("/api/v1/strategies/scaffold")
    assert detail_resp.status_code == 200
    meta = detail_resp.json()["data"]

    assert meta["name"] == "SCAFFOLD"
    assert "control_variates" in meta["supported_features"]
    assert meta["default_parameters"]["algorithm_name"] == "SCAFFOLD"
    assert meta["default_parameters"]["year"] == 2020
    assert "authors" in meta["default_parameters"]
