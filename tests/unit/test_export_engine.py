"""
Module: tests.unit.test_export_engine

Purpose:
Unit test suite for ExportEngine multi-format report generator.
"""

import os
import pytest
from utils.export_engine import ExportEngine


def test_export_engine_benchmark_suite(tmp_path):
    output_dir = str(tmp_path / "results")
    exporter = ExportEngine(output_dir=output_dir)

    dummy_results = [
        {
            "experiment_id": "exp_001",
            "strategy_name": "FedAvg",
            "partition_strategy": "Dirichlet",
            "seed": 42,
            "best_dice": 0.885,
            "avg_loss": 0.145,
            "convergence_round": 3,
            "runtime_sec": 12.5,
            "status": "completed",
        },
        {
            "experiment_id": "exp_002",
            "strategy_name": "FedProx",
            "partition_strategy": "Dirichlet",
            "seed": 42,
            "best_dice": 0.892,
            "avg_loss": 0.138,
            "convergence_round": 2,
            "runtime_sec": 14.1,
            "status": "completed",
        },
    ]

    files = exporter.export_benchmark_suite(
        benchmark_id="test_bm_001",
        name="UnitTest Suite",
        results=dummy_results,
    )

    assert "json" in files and os.path.exists(files["json"])
    assert "leaderboard_csv" in files and os.path.exists(files["leaderboard_csv"])
    assert "markdown_report" in files and os.path.exists(files["markdown_report"])
    assert "pdf_report" in files and os.path.exists(files["pdf_report"])
    assert "latex_tables" in files and os.path.exists(files["latex_tables"])
