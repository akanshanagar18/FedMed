"""
Module: tests.unit.test_scenario_engine

Purpose:
Unit test suite for ScenarioSimulationEngine (simulating 20 failure/scale scenarios).
"""

import pytest
from simulation.scenario_engine import ScenarioSimulationEngine, ScenarioType


def test_scenario_simulation_triggering():
    engine = ScenarioSimulationEngine()
    res = engine.trigger_scenario(ScenarioType.HOSPITAL_FAILURE, "hospital_beta")
    assert res["status"] == "EXECUTED"
    assert res["scenario_type"] == "HOSPITAL_FAILURE"

    history = engine.get_history()
    assert len(history) >= 1
    assert history[-1]["scenario_type"] == "HOSPITAL_FAILURE"


def test_scenario_drift_simulation():
    engine = ScenarioSimulationEngine()
    res = engine.trigger_scenario(ScenarioType.FEATURE_DRIFT, "hospital_alpha", {"mmd": 0.18})
    assert res["status"] == "EXECUTED"
    assert res["scenario_type"] == "FEATURE_DRIFT"
