"""
Module: tests.e2e.test_milestone_k_e2e

Purpose:
End-to-End verification test for Milestone K Research Evaluation & Publication Benchmark Suite.
"""

import os
import subprocess
import sys
import pytest
from utils.statistical_analysis import StatisticalAnalysisEngine
from utils.benchmark_visualizer import BenchmarkVisualizer
from utils.export_engine import ExportEngine


def test_milestone_k_publication_benchmark_pipeline(tmp_path):
    output_dir = str(tmp_path / "benchmark_k_e2e")

    # 1. Simulate experiment matrix results
    results = [
        {
            "experiment_id": "exp_k_001",
            "strategy_name": "FedAvg",
            "partition_strategy": "IID",
            "seed": 42,
            "best_dice": 0.865,
            "avg_loss": 0.162,
            "convergence_round": 3,
            "runtime_sec": 12.0,
            "status": "completed",
        },
        {
            "experiment_id": "exp_k_002",
            "strategy_name": "FedProx",
            "partition_strategy": "NonIID(alpha=0.5)",
            "seed": 42,
            "best_dice": 0.895,
            "avg_loss": 0.131,
            "convergence_round": 2,
            "runtime_sec": 14.5,
            "status": "completed",
        },
        {
            "experiment_id": "exp_k_003",
            "strategy_name": "FedAvg",
            "partition_strategy": "NonIID(alpha=0.5)",
            "seed": 123,
            "best_dice": 0.850,
            "avg_loss": 0.178,
            "convergence_round": 3,
            "runtime_sec": 11.8,
            "status": "completed",
        },
        {
            "experiment_id": "exp_k_004",
            "strategy_name": "FedProx",
            "partition_strategy": "NonIID(alpha=0.5)",
            "seed": 123,
            "best_dice": 0.902,
            "avg_loss": 0.125,
            "convergence_round": 2,
            "runtime_sec": 14.1,
            "status": "completed",
        },
    ]

    # 2. Statistical Analysis
    stat_engine = StatisticalAnalysisEngine(results)
    ranks = stat_engine.rank_strategies()
    tests = stat_engine.compute_pairwise_hypothesis_tests()

    assert len(ranks) == 2
    assert ranks[0]["strategy"] == "FedProx"

    # 3. Generate Visualizations
    vis = BenchmarkVisualizer(output_dir=output_dir)
    plots = vis.generate_all_plots({"benchmark_id": "bm_k", "results": results})
    assert len(plots) == 20

    # 4. Generate Export Suite
    exporter = ExportEngine(output_dir=str(tmp_path / "results"))
    exported_files = exporter.export_benchmark_suite(
        benchmark_id="bm_k",
        name="Milestone K E2E Benchmark",
        results=results,
        plot_paths=plots,
        custom_output_dir=output_dir,
    )

    assert os.path.exists(exported_files["json"])
    assert os.path.exists(exported_files["leaderboard_csv"])
    assert os.path.exists(exported_files["runtime_csv"])
    assert os.path.exists(exported_files["privacy_csv"])
    assert os.path.exists(exported_files["communication_csv"])
    assert os.path.exists(exported_files["statistical_tests_csv"])
    assert os.path.exists(exported_files["latex_tables"])
    assert os.path.exists(exported_files["markdown_report"])
    assert os.path.exists(exported_files["pdf_report"])
