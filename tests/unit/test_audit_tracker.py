"""
Module: tests.unit.test_audit_tracker

Purpose:
Unit test suite for AuditTracker and DatasetLineageManager.
"""

import pytest
from audit.tracker import AuditTracker
from data.versioning import DatasetLineageManager


def test_audit_tracker_record_generation():
    tracker = AuditTracker(experiment_id="exp_test")
    record = tracker.generate_audit_record(dataset_name="BraTS2021", model_name="UNet3D", strategy_name="FedAvg")

    assert record["experiment_id"] == "exp_test"
    assert record["reproducibility_verified"] is True
    assert len(record["audit_signature"]) == 64


def test_dataset_lineage_manager():
    manager = DatasetLineageManager(dataset_name="BraTS2021", version="2.0.0")
    meta = manager.get_lineage_metadata(partition_strategy="Dirichlet(alpha=0.5)", num_hospitals=3)

    assert meta["dataset_name"] == "BraTS2021"
    assert meta["lineage_verified"] is True
