"""
Module: tests.unit.test_experiment_planner

Purpose:
Unit test suite for AutonomousExperimentPlanner (generating proposals for strategy sweeps, privacy trade-offs, PEFT).
"""

import pytest
from analytics.experiment_planner import AutonomousExperimentPlanner


def test_experiment_planner_proposals():
    planner = AutonomousExperimentPlanner()
    plan = planner.propose_next_experiments(
        historical_runs_count=10,
        best_observed_dice=0.86,
        current_strategy="FedAvg",
        remaining_privacy_budget=6.0,
    )
    assert plan["proposed_experiments_count"] >= 2
    proposals = plan["proposals"]
    titles = [p["title"] for p in proposals]
    assert any("Strategy Benchmarking" in t for t in titles)
    assert any("MedSAM" in t for t in titles)
