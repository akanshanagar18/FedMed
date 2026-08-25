"""
Module: tests.integration.test_dataset_mgmt_api

Purpose:
Integration test suite for Dataset Management REST APIs (/api/v1/dataset-management/*).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_dataset_management_endpoints():
    # Register
    payload = {
        "dataset_name": "BraTS_2026_Integration_Set",
        "version": "v1.0.0",
        "modality": "3D_MRI",
        "num_samples": 120,
    }
    reg_resp = client.post("/api/v1/dataset-management/register", json=payload)
    assert reg_resp.status_code == 200
    ds_id = reg_resp.json()["data"]["dataset_id"]

    # List
    list_resp = client.get("/api/v1/dataset-management/datasets")
    assert list_resp.status_code == 200

    # Validate
    val_resp = client.post(f"/api/v1/dataset-management/validate/{ds_id}")
    assert val_resp.status_code == 200
    assert val_resp.json()["data"]["checksum_valid"] is True
