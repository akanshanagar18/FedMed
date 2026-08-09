"""
Module: tests.integration.test_distributed_api

Purpose:
Integration test suite for REST API endpoints (/api/v1/distributed-systems/*).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_distributed_network_endpoint():
    resp = client.get("/api/v1/distributed-systems/network")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "latency_ms" in data
    assert "bandwidth_mbps" in data


def test_distributed_compression_endpoint():
    resp = client.get("/api/v1/distributed-systems/compression")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "supported_algorithms" in data
    assert "compression_ratio" in data


def test_distributed_client_selection_endpoint():
    resp = client.get("/api/v1/distributed-systems/client-selection")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "supported_policies" in data


def test_distributed_summary_endpoint():
    resp = client.get("/api/v1/distributed-systems/distributed")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "asynchronous_fl" in data
