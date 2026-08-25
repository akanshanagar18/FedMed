"""
Module: tests.unit.test_lifecycle_manager

Purpose:
Unit test suite for ExperimentLifecycleManager (creation, stage transition, history logging).
"""

import pytest
from experiments.lifecycle_manager import ExperimentLifecycleManager, ExperimentStage


def test_experiment_lifecycle_transitions():
    mgr = ExperimentLifecycleManager()
    exp = mgr.create_experiment("UNet BraTS Trial", "lead_researcher", "BraTS2021_v1")
    assert exp["stage"] == ExperimentStage.DATASET_PREPARATION.value

    trans = mgr.transition_stage(exp["experiment_id"], ExperimentStage.TRAINING, "Started 10 FL rounds", {"loss": 0.25})
    assert trans["success"] is True
    assert trans["experiment"]["stage"] == ExperimentStage.TRAINING.value
    assert len(trans["experiment"]["stage_history"]) == 2
