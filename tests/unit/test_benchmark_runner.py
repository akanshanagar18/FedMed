"""
Module: tests.unit.test_benchmark_runner

Purpose:
Unit test suite for EnterpriseBenchmarkRunner (matrix benchmarking and report generation).
"""

import pytest
from evaluation.benchmark_runner import EnterpriseBenchmarkRunner


def test_enterprise_benchmark_runner_matrix():
    runner = EnterpriseBenchmarkRunner()
    res = runner.run_benchmark_matrix(num_rounds=3, algorithms=["FedAvg", "Scaffold", "FedNova"])
    assert res["algorithms_evaluated"] == 3
    assert len(res["results"]) == 3
    assert "Comparative Performance Matrix" in res["markdown_report"]
