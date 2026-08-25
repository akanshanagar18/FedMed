"""
Integration tests for /api/v1/dataset REST API endpoint.
"""

from fastapi.testclient import TestClient
import pytest
from app.main import app


@pytest.fixture
def api_client():
    return TestClient(app)


def test_get_dataset_summary_api_endpoint(api_client):
    """Verify GET /api/v1/dataset returns structured dataset metadata."""
    response = api_client.get("/api/v1/dataset")
    assert response.status_code == 200
    json_data = response.json()
    assert "retrieved successfully" in json_data["message"]
    data = json_data["data"]
    assert "dataset_name" in data
    assert "patient_count" in data
    assert "modalities" in data
    assert "t1" in data["modalities"]
    assert "flair" in data["modalities"]
    assert data["patient_count"] >= 0
