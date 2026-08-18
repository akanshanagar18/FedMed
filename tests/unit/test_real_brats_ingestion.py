"""
Unit test for data/real_brats_pipeline.py.
Verifies real BraTS discovery, integrity validation, dataset mode classification,
deterministic patient-level splitting, and train-only hospital partitioning.
"""

from pathlib import Path
import pytest

from data.real_brats_pipeline import (
    discover_and_classify_dataset,
    create_deterministic_split,
    partition_hospitals,
    run_real_brats_ingestion_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_real_brats_discovery_and_classification():
    data_dir = PROJECT_ROOT / "data" / "BraTS2021"
    manifest = discover_and_classify_dataset(data_dir)

    assert manifest["total_subjects_found"] == 4
    assert manifest["valid_subjects_count"] == 4
    assert manifest["invalid_subjects_count"] == 0
    # Strict refusal to label development cohort as FULL_REAL_BRATS
    assert manifest["dataset_mode"] == "DEVELOPMENT_SYNTHETIC"
    assert manifest["is_real_brats"] is False


def test_deterministic_split_and_leakage():
    data_dir = PROJECT_ROOT / "data" / "BraTS2021"
    manifest = discover_and_classify_dataset(data_dir)
    split = create_deterministic_split(manifest, seed=42)

    assert split["counts"]["train"] == 2
    assert split["counts"]["validation"] == 1
    assert split["counts"]["test"] == 1
    assert split["leakage_verification"]["is_disjoint"] is True
    assert len(split["split_hash"]) == 64


def test_hospital_partitioning_empty_flags():
    data_dir = PROJECT_ROOT / "data" / "BraTS2021"
    manifest = discover_and_classify_dataset(data_dir)
    split = create_deterministic_split(manifest, seed=42)
    partitions = partition_hospitals(split, hospitals=("hospital_alpha", "hospital_beta", "hospital_gamma", "hospital_delta"))

    assert partitions["hospital_count"] == 4
    assert partitions["partitions"]["hospital_alpha"]["status"] == "ACTIVE"
    assert partitions["partitions"]["hospital_beta"]["status"] == "ACTIVE"
    # Unassigned hospitals must be explicitly tagged EMPTY_NO_TRAINING_DATA
    assert partitions["partitions"]["hospital_gamma"]["status"] == "EMPTY_NO_TRAINING_DATA"
    assert partitions["partitions"]["hospital_delta"]["status"] == "EMPTY_NO_TRAINING_DATA"
