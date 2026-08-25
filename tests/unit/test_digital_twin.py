"""
Module: tests.unit.test_digital_twin

Purpose:
Unit test suite for DigitalTwinSimulationEngine (predictive what-if simulation modeling).
"""

import pytest
from digital_twin.twin_engine import DigitalTwinSimulationEngine


def test_digital_twin_prediction():
    engine = DigitalTwinSimulationEngine()
    pred = engine.simulate_what_if(
        scenario_description="Hospital Alpha crashes and latency doubles",
        baseline_dice=0.865,
        hospital_dropout_pct=0.33,
        latency_multiplier=2.0,
        drift_mmd=0.08,
    )
    assert "predicted_metrics" in pred
    assert pred["predicted_metrics"]["expected_dice"] < 0.865
    assert pred["predicted_metrics"]["communication_overhead_mb"] > 0
