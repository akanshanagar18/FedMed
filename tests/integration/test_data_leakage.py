"""
Integration Test: tests/integration/test_data_leakage.py

Purpose:
Rigorous data leakage and patient isolation tests for Phase 8.5B.1.
Verifies that:
  1. Train and Validation sets are strictly disjoint
  2. Train and Test sets are strictly disjoint
  3. Validation and Test sets are strictly disjoint
  4. Union of splits exactly equals the original input cohort
  5. Duplicate subject IDs are rejected with ValueError
  6. Missing / empty subject IDs are rejected
  7. Fixed seed produces 100% deterministic identical splits
  8. Different seeds produce distinct, valid, disjoint splits
"""

import pytest
from data.splitter import PatientLevelSplitter


@pytest.fixture
def sample_subjects():
    return [f"BraTS2021_{i:05d}" for i in range(1, 21)]


def test_pairwise_disjointness(sample_subjects):
    splitter = PatientLevelSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    manifest = splitter.split_subjects(sample_subjects)

    train_set = set(manifest.train_subjects)
    val_set = set(manifest.validation_subjects)
    test_set = set(manifest.test_subjects)

    # 1. Train ∩ Val = ∅
    assert len(train_set.intersection(val_set)) == 0, "Leakage between TRAIN and VALIDATION!"
    # 2. Train ∩ Test = ∅
    assert len(train_set.intersection(test_set)) == 0, "Leakage between TRAIN and TEST!"
    # 3. Val ∩ Test = ∅
    assert len(val_set.intersection(test_set)) == 0, "Leakage between VALIDATION and TEST!"


def test_union_completeness(sample_subjects):
    splitter = PatientLevelSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    manifest = splitter.split_subjects(sample_subjects)

    all_split_subjects = set(manifest.train_subjects).union(manifest.validation_subjects).union(manifest.test_subjects)
    assert all_split_subjects == set(sample_subjects), "Split union does not match input cohort!"
    assert len(manifest.train_subjects) + len(manifest.validation_subjects) + len(manifest.test_subjects) == len(sample_subjects)


def test_duplicate_subject_rejection():
    splitter = PatientLevelSplitter(seed=42)
    duplicate_list = ["BraTS2021_00001", "BraTS2021_00002", "BraTS2021_00001"]
    with pytest.raises(ValueError, match="duplicates"):
        splitter.split_subjects(duplicate_list)


def test_empty_subject_rejection():
    splitter = PatientLevelSplitter(seed=42)
    with pytest.raises(ValueError, match="empty"):
        splitter.split_subjects([])


def test_split_determinism(sample_subjects):
    splitter_a = PatientLevelSplitter(seed=123)
    splitter_b = PatientLevelSplitter(seed=123)

    manifest_a = splitter_a.split_subjects(sample_subjects)
    manifest_b = splitter_b.split_subjects(sample_subjects)

    assert manifest_a.train_subjects == manifest_b.train_subjects
    assert manifest_a.validation_subjects == manifest_b.validation_subjects
    assert manifest_a.test_subjects == manifest_b.test_subjects
    assert manifest_a.split_hash == manifest_b.split_hash


def test_different_seeds_produce_different_valid_splits(sample_subjects):
    splitter_1 = PatientLevelSplitter(seed=42)
    splitter_2 = PatientLevelSplitter(seed=999)

    m1 = splitter_1.split_subjects(sample_subjects)
    m2 = splitter_2.split_subjects(sample_subjects)

    # Both must be internally disjoint
    assert m1.disjoint_verified is True
    assert m2.disjoint_verified is True

    # Hashes must differ for distinct seed orderings on 20 subjects
    assert m1.split_hash != m2.split_hash


def test_mini_cohort_edge_case():
    mini_cohort = ["BraTS2021_00001", "BraTS2021_00002", "BraTS2021_00003", "BraTS2021_00004"]
    splitter = PatientLevelSplitter(seed=42, allow_small_dataset=True)
    manifest = splitter.split_subjects(mini_cohort)

    assert manifest.disjoint_verified is True
    assert manifest.counts["train"] == 2
    assert manifest.counts["validation"] == 1
    assert manifest.counts["test"] == 1
    assert manifest.warning is not None
    assert "INSUFFICIENT_DATASET_SIZE_FOR_MEANINGFUL_EXPERIMENT" in manifest.warning
