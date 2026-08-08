"""
Module: tests.unit.test_mlflow_tracker

Purpose:
Unit test suite for MLflowTracker module.
"""

import os
import pytest
from utils.mlflow_tracker import MLflowTracker, get_default_mlflow_tracker


def test_mlflow_tracker_initialization(tmp_path):
    mlruns_dir = str(tmp_path / "mlruns")
    tracker = MLflowTracker(experiment_name="Test_Experiment", tracking_uri=f"file:{mlruns_dir}")
    assert tracker.experiment_name == "Test_Experiment"
    assert tracker.is_active is True


def test_mlflow_run_lifecycle(tmp_path):
    mlruns_dir = str(tmp_path / "mlruns")
    tracker = MLflowTracker(experiment_name="Test_Lifecycle", tracking_uri=f"file:{mlruns_dir}")

    run = tracker.start_run(run_name="unit_test_run", tags={"environment": "pytest"})
    assert run is not None

    tracker.log_params({"lr": 1e-4, "batch_size": 2, "strategy": "FedAvg"})
    tracker.log_metrics({"dice": 0.85, "loss": 0.15}, step=1)

    dummy_file = tmp_path / "dummy_artifact.txt"
    dummy_file.write_text("FedMed Test Artifact")
    tracker.log_artifact(str(dummy_file))

    tracker.end_run(status="FINISHED")
    assert tracker.current_run is None


def test_default_factory():
    tracker = get_default_mlflow_tracker()
    assert tracker is not None
