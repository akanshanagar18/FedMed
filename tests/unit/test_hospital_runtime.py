"""
Module: tests.unit.test_hospital_runtime

Purpose:
Unit test suite for MultiHospitalRuntimeManager and HospitalNodeService.
"""

import pytest
from client.hospital_runtime import MultiHospitalRuntimeManager, HospitalNodeService, HospitalConfig


def test_hospital_node_service_heartbeat_and_training():
    cfg = HospitalConfig("hospital_alpha", "Hospital Alpha", "Siemens 3T", 20)
    service = HospitalNodeService(cfg)
    hb = service.send_heartbeat()
    assert hb["hospital_id"] == "hospital_alpha"
    assert hb["status"] == "CONNECTED"

    metrics = service.train_one_round(round_number=1)
    assert metrics["hospital_id"] == "hospital_alpha"
    assert "train_loss" in metrics


def test_multi_hospital_runtime_manager():
    mgr = MultiHospitalRuntimeManager()
    hospitals = mgr.list_hospitals()
    assert len(hospitals) == 4

    results = mgr.execute_round_across_all(round_number=1)
    assert len(results) == 4

    state_changed = mgr.set_node_active_state("hospital_beta", False)
    assert state_changed is True
    assert mgr.get_hospital("hospital_beta").status == "DISCONNECTED"
    mgr.set_node_active_state("hospital_beta", True)
