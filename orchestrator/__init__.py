"""
Module: orchestrator

Purpose:
Autonomous Federated Orchestrator continuously evaluating platform state, consuming signals
from Drift, Metrics, SLA, Governance, Health, and Privacy engines to execute explainable platform decisions.
"""

from orchestrator.autonomous_orchestrator import AutonomousOrchestrator, OrchestratorDecision

__all__ = ["AutonomousOrchestrator", "OrchestratorDecision"]
