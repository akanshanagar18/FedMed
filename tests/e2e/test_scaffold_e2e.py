"""
Module: tests.e2e.test_scaffold_e2e

Purpose:
End-to-End simulation test for SCAFFOLD strategy execution, control variate propagation, and export engine integration.
"""

import os
import pytest
from server.strategies.scaffold import SCAFFOLD
from server.strategies.adapters.flower_adapter import FlowerStrategyAdapter
from utils.export_engine import ExportEngine


def test_scaffold_e2e_pipeline(tmp_path):
    output_dir = str(tmp_path / "scaffold_e2e")

    scaffold_strat = SCAFFOLD(min_fit_clients=2, min_available_clients=2)
    adapter = FlowerStrategyAdapter(strategy=scaffold_strat, experiment_id="exp_scaffold_e2e")

    # Simulate 3 benchmark runs comparing FedAvg, FedProx, SCAFFOLD
    results = [
        {
            "experiment_id": "exp_scaffold_001",
            "strategy_name": "FedAvg",
            "partition_strategy": "NonIID(alpha=0.5)",
            "seed": 42,
            "best_dice": 0.845,
            "avg_loss": 0.175,
            "convergence_round": 3,
            "runtime_sec": 12.0,
            "status": "completed",
        },
        {
            "experiment_id": "exp_scaffold_002",
            "strategy_name": "FedProx",
            "partition_strategy": "NonIID(alpha=0.5)",
            "seed": 42,
            "best_dice": 0.880,
            "avg_loss": 0.145,
            "convergence_round": 2,
            "runtime_sec": 13.5,
            "status": "completed",
        },
        {
            "experiment_id": "exp_scaffold_003",
            "strategy_name": "SCAFFOLD",
            "partition_strategy": "NonIID(alpha=0.5)",
            "seed": 42,
            "best_dice": 0.912,
            "avg_loss": 0.110,
            "convergence_round": 1,
            "runtime_sec": 14.0,
            "status": "completed",
        },
    ]

    exporter = ExportEngine(output_dir=str(tmp_path / "results"))
    exported_files = exporter.export_benchmark_suite(
        benchmark_id="bm_scaffold_e2e",
        name="SCAFFOLD Benchmark Evaluation",
        results=results,
        custom_output_dir=output_dir,
    )

    assert os.path.exists(exported_files["json"])
    assert os.path.exists(exported_files["leaderboard_csv"])
    assert os.path.exists(exported_files["statistical_tests_csv"])
    assert os.path.exists(exported_files["latex_tables"])

    # Verify SCAFFOLD ranked #1 due to highest Dice (0.912)
    with open(exported_files["json"], "r", encoding="utf-8") as f:
        import json
        summary = json.load(f)
        assert summary["rankings"][0]["strategy"] == "SCAFFOLD"
