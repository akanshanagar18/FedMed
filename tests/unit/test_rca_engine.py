"""
Module: tests.unit.test_rca_engine

Purpose:
Unit test suite for RootCauseAnalysisEngine (diagnosing dropout, feature drift, DP noise, latency).
"""

import pytest
from analytics.rca_engine import RootCauseAnalysisEngine


def test_rca_engine_client_dropout_diagnosis():
    rca = RootCauseAnalysisEngine()
    diag = rca.diagnose_degradation(
        current_loss=0.20,
        previous_loss=0.18,
        current_dice=0.84,
        previous_dice=0.85,
        drift_mmd=0.04,
        active_clients=1,
        total_clients=2,
        dp_epsilon=2.0,
        avg_latency_ms=100.0,
    )
    assert diag["has_degradation"] is True
    assert diag["primary_cause"] == "CLIENT_DROPOUT"
    assert diag["top_confidence"] >= 0.90


def test_rca_engine_feature_drift_diagnosis():
    rca = RootCauseAnalysisEngine()
    diag = rca.diagnose_degradation(
        current_loss=0.22,
        previous_loss=0.21,
        current_dice=0.82,
        previous_dice=0.83,
        drift_mmd=0.16,
        active_clients=2,
        total_clients=2,
        dp_epsilon=2.0,
        avg_latency_ms=100.0,
    )
    assert diag["has_degradation"] is True
    assert diag["primary_cause"] == "FEATURE_DRIFT"
