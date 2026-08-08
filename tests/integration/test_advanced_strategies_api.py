"""
Module: tests.integration.test_advanced_strategies_api

Purpose:
Integration test suite verifying REST API endpoints for all 9 strategies (/api/v1/strategies).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_strategies_list_includes_all_9_algorithms():
    resp = client.get("/api/v1/strategies")
    assert resp.status_code == 200
    strats = resp.json()["data"]["strategies"]

    for expected in ["FedAvg", "FedProx", "SCAFFOLD", "FedAdam", "FedYogi", "FedAdagrad", "FedNova", "FedDyn", "FedBN"]:
        assert any(s.lower() == expected.lower() for s in strats)


def test_individual_strategy_metadata_endpoints():
    for strat_id in ["fedadam", "fedyogi", "fedadagrad", "fednova", "feddyn", "fedbn"]:
        resp = client.get(f"/api/v1/strategies/{strat_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "name" in data
        assert "research_reference" in data
        assert "default_parameters" in data
