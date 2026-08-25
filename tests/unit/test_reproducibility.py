"""
Module: tests.unit.test_reproducibility

Purpose:
Unit test suite for reproducibility metadata collector module.
"""

import os
import json
import pytest
from utils.reproducibility import (
    get_git_metadata,
    get_hardware_metadata,
    get_dependency_versions,
    collect_reproducibility_metadata,
    save_reproducibility_report,
)


def test_get_git_metadata():
    meta = get_git_metadata()
    assert "git_branch" in meta
    assert "git_commit" in meta


def test_get_hardware_metadata():
    hw = get_hardware_metadata()
    assert "cpu_count" in hw
    assert "ram_gb" in hw
    assert "cuda_available" in hw


def test_get_dependency_versions():
    deps = get_dependency_versions()
    assert "python_version" in deps
    assert "torch_version" in deps
    assert "monai_version" in deps
    assert "flower_version" in deps


def test_collect_and_save_reproducibility(tmp_path):
    report_file = str(tmp_path / "reproducibility.json")
    saved_path = save_reproducibility_report(report_file, seed=123, config={"test_cfg": True})
    assert os.path.exists(saved_path)

    with open(saved_path, "r") as f:
        data = json.load(f)
        assert data["random_seed"] == 123
        assert data["config"]["test_cfg"] is True
