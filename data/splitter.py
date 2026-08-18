"""
Module: data.splitter

Purpose:
Deterministic Patient-Level Dataset Splitting Engine for FedMed v2.0.
Guarantees 100% subject-level isolation across TRAIN, VALIDATION, and TEST sets with zero slice/voxel data leakage.
Outputs reports/dataset_split.json manifest with SHA-256 verification hashes.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


class DatasetSplitManifest(BaseModel):
    """Pydantic representation of the 3-way subject split."""
    dataset_mode: str
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    total_subjects: int
    train_subjects: List[str]
    validation_subjects: List[str]
    test_subjects: List[str]
    counts: Dict[str, int]
    disjoint_verified: bool
    split_hash: str
    warning: Optional[str] = None


class PatientLevelSplitter:
    """
    Deterministic subject-level dataset splitter.
    Never splits voxels, slices, or patches of the same patient between splits.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        allow_small_dataset: bool = False,
    ):
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0!"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.allow_small_dataset = allow_small_dataset

    @staticmethod
    def verify_disjoint(
        train: List[str],
        val: List[str],
        test: List[str],
        all_subjects: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Formally verifies mathematical disjointness across all three splits.
        """
        set_train = set(train)
        set_val = set(val)
        set_test = set(test)
        set_all = set(all_subjects)
        errors: List[str] = []

        # 1. Check pairwise disjointness
        inter_tv = set_train.intersection(set_val)
        if inter_tv:
            errors.append(f"Data leakage detected between TRAIN and VALIDATION: {inter_tv}")

        inter_tt = set_train.intersection(set_test)
        if inter_tt:
            errors.append(f"Data leakage detected between TRAIN and TEST: {inter_tt}")

        inter_vt = set_val.intersection(set_test)
        if inter_vt:
            errors.append(f"Data leakage detected between VALIDATION and TEST: {inter_vt}")

        # 2. Check union equality
        union_all = set_train.union(set_val).union(set_test)
        if union_all != set_all:
            missing = set_all.difference(union_all)
            extra = union_all.difference(set_all)
            if missing:
                errors.append(f"Subjects missing from split union: {missing}")
            if extra:
                errors.append(f"Unexpected subjects in split union: {extra}")

        # 3. Check for duplicates in original input
        if len(all_subjects) != len(set_all):
            errors.append("Duplicate subject IDs found in input subject list!")

        return len(errors) == 0, errors

    def split_subjects(
        self,
        subject_ids: List[str],
        dataset_mode: str = "DEVELOPMENT_SYNTHETIC",
    ) -> DatasetSplitManifest:
        """
        Deterministically splits subject IDs into TRAIN, VALIDATION, and TEST sets.
        """
        # Clean and sort uniquely first for determinism
        unique_sorted_ids = sorted(list(set(subject_ids)))
        total = len(unique_sorted_ids)

        if len(subject_ids) != total:
            raise ValueError("Input subject_ids contains duplicates!")
        if total == 0:
            raise ValueError("Cannot split empty subject list!")

        warning_msg = None
        if total < 10:
            warning_msg = "WARNING: INSUFFICIENT_DATASET_SIZE_FOR_MEANINGFUL_EXPERIMENT"
            if not self.allow_small_dataset:
                logger.warning(
                    f"{warning_msg} (Found {total} subjects). "
                    "Use allow_small_dataset=True for development testing only."
                )

        # Deterministic shuffle using explicit NumPy Generator
        rng = np.random.default_rng(self.seed)
        shuffled = list(unique_sorted_ids)
        rng.shuffle(shuffled)

        if total <= 4:
            # Special deterministic partition for mini development cohorts:
            # E.g. for 4 subjects: 2 train, 1 validation, 1 test
            n_train = max(1, total - 2)
            n_val = 1
            n_test = max(1, total - n_train - n_val)
        else:
            n_train = int(np.round(total * self.train_ratio))
            n_val = int(np.round(total * self.val_ratio))
            # Ensure test gets the remainder
            n_test = total - n_train - n_val

            # Ensure every partition has at least 1 subject if total >= 3
            if n_train == 0 and total >= 1:
                n_train = 1
            if n_val == 0 and total >= 2:
                n_val = 1
            if n_test == 0 and total >= 3:
                n_test = 1
            n_train = total - n_val - n_test

        train_ids = sorted(shuffled[:n_train])
        val_ids = sorted(shuffled[n_train : n_train + n_val])
        test_ids = sorted(shuffled[n_train + n_val :])

        is_disjoint, errs = self.verify_disjoint(train_ids, val_ids, test_ids, unique_sorted_ids)
        if not is_disjoint:
            raise RuntimeError(f"Split integrity verification failed:\n" + "\n".join(errs))

        # Compute deterministic SHA-256 fingerprint of the split
        split_repr = f"seed={self.seed}|train={','.join(train_ids)}|val={','.join(val_ids)}|test={','.join(test_ids)}"
        split_hash = hashlib.sha256(split_repr.encode("utf-8")).hexdigest()

        manifest = DatasetSplitManifest(
            dataset_mode=dataset_mode,
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            total_subjects=total,
            train_subjects=train_ids,
            validation_subjects=val_ids,
            test_subjects=test_ids,
            counts={
                "train": len(train_ids),
                "validation": len(val_ids),
                "test": len(test_ids),
            },
            disjoint_verified=is_disjoint,
            split_hash=split_hash,
            warning=warning_msg,
        )

        return manifest


def generate_and_save_split(
    manifest_path: Union[str, Path] = "reports/brats_dataset_manifest.json",
    seed: int = 42,
    allow_small_dataset: bool = True,
) -> DatasetSplitManifest:
    """
    Loads dataset manifest, executes deterministic splitting, and saves reports/dataset_split.json.
    """
    m_path = Path(manifest_path)
    if not m_path.is_absolute():
        m_path = PROJECT_ROOT / m_path

    if not m_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found at {m_path}. Run validate_brats_dataset.py first.")

    with open(m_path, "r") as f:
        data_manifest = json.load(f)

    subject_ids = [s["subject_id"] for s in data_manifest["subjects"] if s.get("is_valid", True)]
    mode = data_manifest.get("dataset_mode", "DEVELOPMENT_SYNTHETIC")

    splitter = PatientLevelSplitter(seed=seed, allow_small_dataset=allow_small_dataset)
    split_manifest = splitter.split_subjects(subject_ids=subject_ids, dataset_mode=mode)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "dataset_split.json"
    with open(out_file, "w") as f:
        json.dump(split_manifest.model_dump(), f, indent=2)

    logger.info(f"Saved dataset split manifest to {out_file}")
    return split_manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deterministic Subject-Level Splitter")
    parser.add_argument("--manifest", type=str, default="reports/brats_dataset_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-small-dataset", action="store_true", default=True)
    args = parser.parse_args()

    split_res = generate_and_save_split(
        manifest_path=args.manifest,
        seed=args.seed,
        allow_small_dataset=args.allow_small_dataset,
    )
    print("=" * 60)
    print("📋 DATASET PATIENT-LEVEL SPLIT SUMMARY")
    print("=" * 60)
    print(f"Dataset Mode: {split_res.dataset_mode}")
    print(f"Seed:         {split_res.seed}")
    print(f"Total:        {split_res.total_subjects}")
    print(f"TRAIN:        {split_res.counts['train']} subjects -> {split_res.train_subjects}")
    print(f"VALIDATION:   {split_res.counts['validation']} subjects -> {split_res.validation_subjects}")
    print(f"TEST:         {split_res.counts['test']} subjects -> {split_res.test_subjects}")
    print(f"Disjoint:     {split_res.disjoint_verified}")
    print(f"Split Hash:   {split_res.split_hash[:16]}...")
    if split_res.warning:
        print(f"Warning:      {split_res.warning}")
    print("=" * 60)
