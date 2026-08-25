"""
Integration test for scripts/run_scaling_benchmark.py.
Verifies multi-client federated scaling execution, throughput calculation,
and communication payload metrics.
"""

from pathlib import Path
import pytest

from scripts.run_scaling_benchmark import run_scaling_benchmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_scaling_benchmark_execution():
    report = run_scaling_benchmark(
        data_dir="data/BraTS2021",
        rounds=2,
        seed=42,
        allow_development=True,
    )

    assert "performance_metrics" in report
    assert report["performance_metrics"]["throughput_samples_per_sec"] > 0.0
    assert report["performance_metrics"]["throughput_rounds_per_sec"] > 0.0
    assert "communication_metrics" in report
    assert report["communication_metrics"]["total_comm_payload_per_round_mb"] > 0.0
    assert "final_metrics" in report
    assert report["final_metrics"]["test_mean_dice"] > 0.0
