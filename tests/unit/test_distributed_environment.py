"""
Unit test for scripts/audit_distributed_environment.py.
Verifies genuine hardware environment audit and topology classification.
"""

from pathlib import Path
import pytest

from scripts.audit_distributed_environment import audit_hardware_and_environment

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_hardware_environment_audit():
    report = audit_hardware_and_environment()

    assert "hardware_classification" in report
    assert report["hardware_classification"] in [
        "SINGLE_NODE_SINGLE_GPU",
        "SINGLE_NODE_MULTI_GPU",
        "MULTI_NODE",
        "NO_ACCELERATOR",
    ]
    assert report["world_size_capability"] >= 1
    assert "software_versions" in report
    assert "host" in report
    assert "accelerators" in report

    out_file = PROJECT_ROOT / "reports" / "environment" / "distributed_environment.json"
    assert out_file.exists()
