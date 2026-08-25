"""
Module: tests.unit.test_dataset_manager

Purpose:
Unit test suite for DatasetManager (registration, SHA-256 validation checksums, listing).
"""

import pytest
from dataset.manager import DatasetManager, global_dataset_manager


def test_dataset_manager_registration_and_checksum():
    mgr = global_dataset_manager
    ds = mgr.register_dataset("BraTS_Cohort_2026", "v2.0.0", "3D_MRI", ["T1", "T1ce"], 150)
    assert ds["dataset_name"] == "BraTS_Cohort_2026"
    assert "checksum_sha256" in ds

    valid = mgr.validate_checksum(ds["dataset_id"])
    assert valid is True
