"""
Unit tests for Phase 8.5B.8 Master Reality Audit reports.
Verifies repository inventory, mock audit, split provenance, test firewall,
timing precision, dependency policies, and security invariants.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
B8_DIR = PROJECT_ROOT / "reports" / "b8"


def test_b8_audit_reports_exist():
    expected_reports = [
        "repository_inventory.json",
        "mock_hardcode_audit.json",
        "split_provenance_audit.json",
        "test_firewall_audit.json",
        "threat_model_reality_audit.json",
        "inference_timing_audit.json",
        "reproducibility_audit.json",
        "dependency_audit.json",
        "security_secret_audit.json",
    ]
    for r in expected_reports:
        assert (B8_DIR / r).exists(), f"Missing audit report: {r}"


def test_b8_timing_mathematical_consistency():
    with open(B8_DIR / "inference_timing_audit.json", "r") as f:
        timing = json.load(f)
    assert timing["mathematical_consistency"] is True
    assert timing["discrepancy_ms"] == 0.0


def test_b8_split_provenance_integrity():
    with open(B8_DIR / "split_provenance_audit.json", "r") as f:
        split = json.load(f)
    assert split["current_canonical_split_hash"] == "715cadbe78d2a9669dd331d0681baa060d3a175d9afbb50cc21941609d0238df"
    assert split["disjoint_verification"]["train_val_overlap"] == 0
    assert split["disjoint_verification"]["train_test_overlap"] == 0
    assert split["disjoint_verification"]["val_test_overlap"] == 0


def test_b8_security_audit_zero_secrets():
    with open(B8_DIR / "security_secret_audit.json", "r") as f:
        sec = json.load(f)
    assert sec["tenseal_secret_keys_in_artifacts"] is False
