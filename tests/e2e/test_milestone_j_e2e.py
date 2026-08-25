"""
Module: tests.e2e.test_milestone_j_e2e

Purpose:
End-to-End verification test for Milestone J Research Experiment Tracking, Artifact Management & Reproducibility Platform.
"""

import os
import subprocess
import sys
import pytest
from utils.checkpoint_registry import get_checkpoint_registry
from utils.export_engine import ExportEngine


def test_milestone_j_end_to_end_pipeline(tmp_path):
    # 1. Execute baseline training
    res_script = os.path.join(os.getcwd(), "scripts", "train_baseline.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    proc = subprocess.run(
        [sys.executable, res_script, "--epochs", "1", "--experiment-id", "e2e_milestone_j"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    # 2. Verify Checkpoint Registry
    registry = get_checkpoint_registry()
    checkpoints = registry.list_checkpoints(experiment_id="e2e_milestone_j")
    assert len(checkpoints) >= 1
    best_chk = registry.get_best_checkpoint(experiment_id="e2e_milestone_j")
    assert best_chk is not None
    assert os.path.exists(best_chk.file_path)

    # 3. Verify Export Engine
    exporter = ExportEngine(output_dir=str(tmp_path / "results"))
    files = exporter.export_benchmark_suite(
        benchmark_id="e2e_bm",
        name="E2E Benchmark",
        results=[{
            "experiment_id": "e2e_milestone_j",
            "strategy_name": "CentralizedBaseline",
            "partition_strategy": "N/A",
            "seed": 42,
            "best_dice": best_chk.dice,
            "avg_loss": best_chk.loss,
            "convergence_round": 1,
            "runtime_sec": 5.0,
            "status": "completed",
        }],
    )

    assert os.path.exists(files["json"])
    assert os.path.exists(files["csv"])
    assert os.path.exists(files["markdown"])
    assert os.path.exists(files["pdf"])
