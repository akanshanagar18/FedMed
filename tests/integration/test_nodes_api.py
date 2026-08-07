"""
Integration test for REST API /api/v1/nodes endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from dashboard.backend.app.main import app

client = TestClient(app)


def test_get_nodes_api_endpoint():
    """Test GET /api/v1/nodes returns hospital nodes registry."""
    response = client.get("/api/v1/nodes")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "total_nodes" in data
    assert "nodes" in data


def test_post_nodes_heartbeat_api_endpoint():
    """Test POST /api/v1/nodes/heartbeat updates node status."""
    payload = {
        "hospital_id": "hospital_beta",
        "status": "RECONNECTING",
        "active_round": 2,
        "reconnect_count": 1,
        "training_state": "reconnecting",
    }
    response = client.post("/api/v1/nodes/heartbeat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["node"]["status"] == "RECONNECTING"
