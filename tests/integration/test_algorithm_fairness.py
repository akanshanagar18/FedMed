"""
Integration Test: tests/integration/test_algorithm_fairness.py

Purpose:
Verifies algorithmic fairness across Centralized, FedAvg, and FedProx for Phase 8.5B.4:
  1. Identical data split and split hash across all algorithms
  2. Identical initial parameter hash (H_centralized == H_fedavg == H_fedprox == H_0)
  3. Identical model architecture and parameter count (4,810,074)
  4. Identical preprocessing configuration
  5. Equal cohort optimization budget
  6. Strict test set isolation during training (access_count = 0)
  7. Generation of standardized JSON artifacts and manifest
"""

import json
from pathlib import Path
import pytest

from scripts.run_benchmark import run_benchmark_suite

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_algorithm_fairness_and_manifest():
    """Runs benchmark suite and verifies fairness invariants across Centralized, FedAvg, FedProx."""
    result = run_benchmark_suite(
        data_dir="data/BraTS2021",
        config_path="configs/experiments/real_brats_fedavg.yaml",
        rounds=3,
        seed=42,
        proximal_mu=0.01,
        allow_development=True,
    )

    # 1. Verify Fairness Verification Block
    fv = result["fairness_verification"]
    assert fv["same_split"] is True
    assert fv["same_initialization"] is True
    assert fv["same_model_architecture"] is True
    assert fv["same_parameter_count"] == 4810074
    assert fv["same_optimizer_and_lr"] is True
    assert fv["same_training_budget"] is True
    assert fv["no_static_benchmark_lookups"] is True

    # 2. Check Algorithm Initial Parameter Hashes
    h_init = fv["initial_parameter_hash"]
    assert result["algorithms"]["Centralized"]["initial_parameter_hash"] == h_init
    assert result["algorithms"]["FedAvg"]["initial_parameter_hash"] == h_init
    assert result["algorithms"]["FedProx"]["initial_parameter_hash"] == h_init

    # 3. Check Algorithm Final Hashes Changed
    assert result["algorithms"]["Centralized"]["final_parameter_hash"] != h_init
    assert result["algorithms"]["FedAvg"]["final_parameter_hash"] != h_init
    assert result["algorithms"]["FedProx"]["final_parameter_hash"] != h_init

    # 4. Check Test Firewall Integrity
    tf = result["test_firewall_status"]
    assert tf["TEST_ACCESSED_DURING_TRAINING"] is False
    assert tf["FINAL_TEST_EVALUATED"] is True
    assert tf["TEST_EVALUATION_COUNT_PER_ALGO"] == 1

    # 5. Check Artifact Existence on Disk
    run_id = result["benchmark_run_id"]
    run_dir = PROJECT_ROOT / "reports" / "experiments" / "benchmark" / run_id
    assert (run_dir / "experiment_manifest.json").exists()
    assert (run_dir / "centralized.json").exists()
    assert (run_dir / "fedavg.json").exists()
    assert (run_dir / "fedprox.json").exists()
    assert (run_dir / "comparison.json").exists()
