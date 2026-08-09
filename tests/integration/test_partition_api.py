"""
Integration tests for /api/v1/dataset/partitions REST API endpoint.
"""

from fastapi.testclient import TestClient
import pytest
from app.main import app


@pytest.fixture
def api_client():
    return TestClient(app)


def test_get_dataset_partitions_endpoint_dirichlet(api_client):
    """Verify GET /api/v1/dataset/partitions returns Dirichlet non-IID partition details."""
    response = api_client.get("/api/v1/dataset/partitions?strategy=dirichlet&alpha=0.5")
    assert response.status_code == 200
    json_data = response.json()

    assert "retrieved successfully" in json_data["message"]
    data = json_data["data"]
    assert data["partition_strategy"] == "Dirichlet"
    assert data["dirichlet_alpha"] == 0.5
    assert data["disjoint_verified"] is True
    assert "partitions" in data
    assert "hospital_alpha" in data["partitions"]
    assert "hospital_beta" in data["partitions"]
    assert "hospital_gamma" in data["partitions"]


def test_get_dataset_partitions_endpoint_iid(api_client):
    """Verify GET /api/v1/dataset/partitions returns IID partition details."""
    response = api_client.get("/api/v1/dataset/partitions?strategy=iid")
    assert response.status_code == 200
    json_data = response.json()

    data = json_data["data"]
    assert data["partition_strategy"] == "IID"
    assert data["disjoint_verified"] is True
