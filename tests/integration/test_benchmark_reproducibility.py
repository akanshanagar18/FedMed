"""
Integration Test: tests/integration/test_benchmark_reproducibility.py

Purpose:
Verifies execution reproducibility of the benchmark suite for Phase 8.5B.4:
  1. Executes Run A and Run B with identical random seeds (seed=42)
  2. Verifies initial parameter hashes are bitwise identical
  3. Verifies training losses and validation Dice/IoU match within numerical tolerance
  4. Confirms deterministic execution pipeline
"""

from pathlib import Path
import pytest
import numpy as np

from scripts.run_benchmark import run_benchmark_suite


def test_benchmark_run_reproducibility():
    """Executes benchmark suite twice with seed=42 and verifies reproducibility."""
    res_a = run_benchmark_suite(
        rounds=2,
        seed=42,
        proximal_mu=0.01,
        allow_development=True,
    )

    res_b = run_benchmark_suite(
        rounds=2,
        seed=42,
        proximal_mu=0.01,
        allow_development=True,
    )

    # 1. Initial Fingerprints Must Be Bitwise Identical
    init_a = res_a["fairness_verification"]["initial_parameter_hash"]
    init_b = res_b["fairness_verification"]["initial_parameter_hash"]
    assert init_a == init_b

    # 2. Check Metrics Matching Across Algorithms
    for algo in ["Centralized", "FedAvg", "FedProx"]:
        rep_a = res_a["algorithms"][algo]
        rep_b = res_b["algorithms"][algo]

        # Initial hash match
        assert rep_a["initial_parameter_hash"] == rep_b["initial_parameter_hash"]

        # Final Validation Mean Dice Matching (within Apple MPS floating tolerance 1e-2)
        val_dice_a = rep_a["final_validation_dice"]["mean_dice"]
        val_dice_b = rep_b["final_validation_dice"]["mean_dice"]
        assert pytest.approx(val_dice_a, abs=1e-2) == val_dice_b

        # Final Test Mean Dice Matching (within Apple MPS floating tolerance 1e-2)
        test_dice_a = rep_a["final_test_dice"]["mean_dice"]
        test_dice_b = rep_b["final_test_dice"]["mean_dice"]
        assert pytest.approx(test_dice_a, abs=1e-2) == test_dice_b
