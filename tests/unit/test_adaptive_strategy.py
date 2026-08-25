"""
Module: tests.unit.test_adaptive_strategy

Purpose:
Unit test suite for AdaptiveStrategySelector (evaluating dropout, drift, latency, and privacy noise rules).
"""

import pytest
from strategy.adaptive_engine import AdaptiveStrategySelector


def test_adaptive_strategy_dropout_rule():
    selector = AdaptiveStrategySelector()
    res = selector.evaluate_and_select_strategy(
        current_strategy="FedAvg",
        participation_rate=0.50,
        avg_latency_ms=150.0,
        gradient_divergence=0.05,
        drift_mmd=0.02,
        privacy_epsilon=2.0,
        node_dropouts=1,
    )
    assert res["recommended_strategy"] == "FedNova"
    assert res["strategy_changed"] is True


def test_adaptive_strategy_drift_rule():
    selector = AdaptiveStrategySelector()
    res = selector.evaluate_and_select_strategy(
        current_strategy="FedAvg",
        participation_rate=1.0,
        avg_latency_ms=100.0,
        gradient_divergence=0.30,
        drift_mmd=0.18,
        privacy_epsilon=2.0,
        node_dropouts=0,
    )
    assert res["recommended_strategy"] == "Scaffold"
    assert res["strategy_changed"] is True
