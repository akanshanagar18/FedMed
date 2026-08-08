"""
Module: tests.integration.test_benchmark_analytics_api

Purpose:
Integration tests for Milestone K benchmark analytics REST API endpoints (/analytics, /statistical-tests, /latex-tables, /plots).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_benchmark_analytics_api_endpoints():
    # Register dummy benchmark
    create_resp = client.post("/api/v1/benchmarks", json={
        "benchmark_id": "bm_analytics_test",
        "name": "Analytics Test Suite",
        "status": "completed",
    })
    assert create_resp.status_code in [200, 201]

    # Test analytics
    analytics_resp = client.get("/api/v1/benchmarks/bm_analytics_test/analytics")
    assert analytics_resp.status_code == 200
    assert "rankings" in analytics_resp.json()["data"]

    # Test statistical tests
    stat_resp = client.get("/api/v1/benchmarks/bm_analytics_test/statistical-tests")
    assert stat_resp.status_code == 200

    # Test LaTeX tables
    latex_resp = client.get("/api/v1/benchmarks/bm_analytics_test/latex-tables")
    assert latex_resp.status_code == 200
    assert "latex_tables" in latex_resp.json()["data"]

    # Test plots list
    plots_resp = client.get("/api/v1/benchmarks/bm_analytics_test/plots")
    assert plots_resp.status_code == 200
