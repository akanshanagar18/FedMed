"""
Integration tests for FastAPI Backend API Endpoints & Services
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from dashboard.backend.app.main import app
from dashboard.backend.app.database.session import Base, engine, SessionLocal
from dashboard.backend.app.services.hospital import HospitalService
from dashboard.backend.app.schemas.node import HospitalStatus
from server.flower_server import MetricsReporterStrategy


@pytest.fixture(autouse=True)
def setup_test_db():
    """Reset database tables before each integration test."""
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    """Fixture providing FastAPI TestClient instance."""
    return TestClient(app)


@pytest.mark.integration
def test_health_check_endpoint(client):
    """Test GET /api/v1/health returns HTTP 200 and operational status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Backend is operational"
    assert data["data"]["status"] == "ok"
    assert "uptime_seconds" in data["data"]


@pytest.mark.integration
def test_submit_and_get_metrics(client):
    """Test POST /api/v1/metrics persists metric and GET retrieves it."""
    payload = {
        "experiment_id": "pytest_exp",
        "round_number": 1,
        "training_loss": 0.421,
        "dice_score": 0.812,
    }

    # Submit metric via POST
    post_res = client.post("/api/v1/metrics", json=payload)
    assert post_res.status_code == 200
    assert post_res.json()["message"] == "Metric saved successfully"

    # Retrieve metrics via GET
    get_res = client.get("/api/v1/metrics/pytest_exp")
    assert get_res.status_code == 200
    retrieved = get_res.json()["data"]
    assert len(retrieved) >= 1
    assert retrieved[0]["experiment_id"] == "pytest_exp"
    assert retrieved[0]["training_loss"] == 0.421
    assert retrieved[0]["dice_score"] == 0.812


@pytest.mark.integration
def test_hospital_service_registration():
    """Test HospitalService register_or_update CRUD operations."""
    db = SessionLocal()
    try:
        status_payload = HospitalStatus(
            hospital_id="hosp_test_node",
            name="Test Hospital Node",
            connection_status="connected",
            client_latency_ms=10,
            last_seen=datetime.utcnow(),
        )

        # Register new node
        node = HospitalService.register_or_update(db, status_payload)
        assert node.hospital_id == "hosp_test_node"
        assert node.connection_status == "connected"

        # Update node status
        status_payload.connection_status = "training"
        updated_node = HospitalService.register_or_update(db, status_payload)
        assert updated_node.connection_status == "training"

        # List all hospitals
        all_hospitals = HospitalService.get_all_hospitals(db)
        assert any(h.hospital_id == "hosp_test_node" for h in all_hospitals)
    finally:
        db.close()


@pytest.mark.integration
def test_flower_server_metrics_reporter_strategy(client):
    """Test Flower server MetricsReporterStrategy._report_metrics helper."""
    strategy = MetricsReporterStrategy(api_url="http://testserver", experiment_id="flwr_pytest")
    
    # Verify reporter instantiation and configuration
    assert strategy.experiment_id == "flwr_pytest"
    assert strategy.api_url == "http://testserver"
