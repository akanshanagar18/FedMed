"""
Module: tests.e2e.test_milestone_o_e2e

Purpose:
End-to-End simulation test for Milestone O Advanced Federated Optimization Suite.
"""

import os
import pytest
from server.strategies.registry import StrategyRegistry
from utils.export_engine import ExportEngine


def test_milestone_o_e2e_9_strategy_benchmark_pipeline(tmp_path):
    output_dir = str(tmp_path / "milestone_o_results")

    # 1. Verify all 9 strategies discoverable
    strategies = StrategyRegistry.list_strategies()
    for name in ["FedAvg", "FedProx", "SCAFFOLD", "FedAdam", "FedYogi", "FedAdagrad", "FedNova", "FedDyn", "FedBN"]:
        assert any(s.lower() == name.lower() for s in strategies)

    # 2. Simulate 9-way benchmark results
    results = [
        {"experiment_id": f"exp_o_{i}", "strategy_name": name, "best_dice": 0.85 + (i * 0.01), "avg_loss": 0.20 - (i * 0.01), "runtime_sec": 10.0 + i}
        for i, name in enumerate(["FedAvg", "FedProx", "SCAFFOLD", "FedAdam", "FedYogi", "FedAdagrad", "FedNova", "FedDyn", "FedBN"])
    ]

    exporter = ExportEngine(output_dir=output_dir)
    exported_files = exporter.export_benchmark_suite(
        benchmark_id="bm_milestone_o",
        name="Milestone O 9-Strategy Research Benchmark",
        results=results,
        custom_output_dir=output_dir,
    )

    assert os.path.exists(exported_files["json"])
    assert os.path.exists(exported_files["leaderboard_csv"])
    assert os.path.exists(exported_files["statistical_tests_csv"])
    assert os.path.exists(exported_files["latex_tables"])
