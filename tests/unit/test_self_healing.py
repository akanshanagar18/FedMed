"""
Module: tests.unit.test_self_healing

Purpose:
Unit test suite for SelfHealingRecoveryEngine (node failure recovery, aggregation failure fallback).
"""

import pytest
from resilience.self_healing import SelfHealingRecoveryEngine


def test_self_healing_node_failure_recovery():
    engine = SelfHealingRecoveryEngine()
    res = engine.recover_node_failure("hospital_beta", "gRPC Timeout")
    assert res["node_id"] == "hospital_beta"
    assert res["success"] is True
    assert "recovery" in res["message"].lower()



def test_self_healing_aggregation_failure_recovery():
    engine = SelfHealingRecoveryEngine()
    res = engine.recover_aggregation_failure(round_number=5, strategy_name="FedAvg", error_msg="Gradient Nan")
    assert res["round_number"] == 5
    assert res["failed_strategy"] == "FedAvg"
    assert res["fallback_strategy"] == "FedProx"
    assert res["success"] is True
