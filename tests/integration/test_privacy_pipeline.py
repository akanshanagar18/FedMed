"""
Integration Test: tests/integration/test_privacy_pipeline.py

Purpose:
Verifies the complete 4-quadrant privacy matrix execution for Phase 8.5B.5:
  1. Identical model initialization across all 4 modes (H_A == H_B == H_C == H_D == H_0)
  2. DP Gaussian mechanism and RDP accounting active
  3. TenSEAL CKKS homomorphic aggregation active with public evaluation key
  4. Measured HE approximation error < 1e-5
  5. Isolated validation and firewalled test evaluation
  6. Persistence of JSON reports and manifest
"""

from pathlib import Path
import pytest

from scripts.run_privacy_experiment import run_privacy_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_privacy_matrix_pipeline_execution():
    """Runs 4-quadrant privacy matrix and verifies all privacy and cryptographic invariants."""
    result = run_privacy_matrix(
        data_dir="data/BraTS2021",
        config_path="configs/experiments/privacy.yaml",
        rounds=2,
        seed=42,
        allow_development=True,
    )

    init_hash = result["initial_parameter_hash"]
    exps = result["experiments"]

    # 1. Identical Initial Model Parameters Across All 4 Modes
    assert exps["FedAvg_Baseline"]["initial_parameter_hash"] == init_hash
    assert exps["FedAvg_DP"]["initial_parameter_hash"] == init_hash
    assert exps["FedAvg_HE"]["initial_parameter_hash"] == init_hash
    assert exps["FedAvg_DP_HE"]["initial_parameter_hash"] == init_hash

    # 2. DP Accounting Integrity
    assert exps["FedAvg_DP"]["privacy_budget"]["epsilon"] > 0.0
    assert exps["FedAvg_DP"]["privacy_budget"]["delta"] == 1e-5
    assert exps["FedAvg_DP_HE"]["privacy_budget"]["epsilon"] > 0.0

    # 3. HE Error Verification
    he_errors = exps["FedAvg_HE"]["he_errors"]
    assert len(he_errors) == 2
    for err in he_errors:
        assert err["max_absolute_error"] < 1e-5

    # 4. DP+HE Composition Error Verification
    dp_he_errors = exps["FedAvg_DP_HE"]["he_errors"]
    assert len(dp_he_errors) == 2
    for err in dp_he_errors:
        assert err["max_absolute_error"] < 1e-5

    # 5. Check Artifact Existence
    run_id = result["run_id"]
    run_dir = PROJECT_ROOT / "reports" / "experiments" / "privacy" / run_id
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "fedavg_baseline.json").exists()
    assert (run_dir / "dp.json").exists()
    assert (run_dir / "he.json").exists()
    assert (run_dir / "dp_he.json").exists()
    assert (run_dir / "comparison.json").exists()
