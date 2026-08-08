"""
Module: tests.unit.test_benchmark_matrix

Purpose:
Unit tests for benchmark matrix generator configuration.
"""

import pytest
from scripts.run_benchmarks import MatrixConfig, generate_experiment_matrix


def test_generate_experiment_matrix():
    config = MatrixConfig(
        strategies=["FedAvg", "FedProx"],
        partitions=["IID", "Dirichlet"],
        seeds=[42, 123],
    )
    matrix = generate_experiment_matrix(config, benchmark_id="bm_test")
    assert len(matrix) == 8  # 2 * 2 * 2
    assert matrix[0]["strategy_name"] in ["FedAvg", "FedProx"]
    assert matrix[0]["seed"] in [42, 123]
