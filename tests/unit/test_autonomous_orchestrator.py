"""
Module: tests.unit.test_autonomous_orchestrator

Purpose:
Unit test suite for AutonomousOrchestrator (evaluating decisions: CONTINUE, PAUSE, ABORT, LAUNCH_HPO, TRIGGER_RETRAINING).
"""

import pytest
from orchestrator.autonomous_orchestrator import AutonomousOrchestrator, OrchestratorDecision


def test_autonomous_orchestrator_continue():
    orchestrator = AutonomousOrchestrator()
    res = orchestrator.evaluate_system_state_and_decide(
        current_round=5,
        training_loss=0.15,
        validation_dice=0.86,
        drift_mmd=0.03,
        sla_compliant=True,
        active_nodes_count=2,
        total_nodes_count=2,
        privacy_epsilon=2.5,
    )
    assert res["decision"] == OrchestratorDecision.CONTINUE_TRAINING.value
    assert res["severity"] == "INFO"


def test_autonomous_orchestrator_pause_on_sla():
    orchestrator = AutonomousOrchestrator()
    res = orchestrator.evaluate_system_state_and_decide(
        current_round=5,
        training_loss=0.15,
        validation_dice=0.86,
        drift_mmd=0.03,
        sla_compliant=False,
        active_nodes_count=2,
        total_nodes_count=2,
        privacy_epsilon=2.5,
    )
    assert res["decision"] == OrchestratorDecision.PAUSE_TRAINING.value
    assert res["severity"] == "CRITICAL"


def test_autonomous_orchestrator_launch_hpo():
    orchestrator = AutonomousOrchestrator()
    res = orchestrator.evaluate_system_state_and_decide(
        current_round=5,
        training_loss=0.35,
        validation_dice=0.75,
        drift_mmd=0.03,
        sla_compliant=True,
        active_nodes_count=2,
        total_nodes_count=2,
        privacy_epsilon=2.5,
    )
    assert res["decision"] == OrchestratorDecision.LAUNCH_HPO.value
