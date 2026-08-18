"""
Module: data.real_brats_pipeline

Purpose:
Real BraTS Ingestion, Canonical Validation, Classification, Splitting, and Hospital Partitioning Engine.
Supports both BraTS 2021 and BraTS-GLI 2024 cohorts seamlessly via data/canonical_adapter.py.
"""

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from data.canonical_adapter import (
    BRATS_2021_IDENTITY,
    BRATS_GLI_2024_IDENTITY,
    DatasetIdentity,
    DatasetVersion,
    LabelCanonicalizer,
    ModalityCanonicalizer,
)

logger = logging.getLogger("real_brats_pipeline")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class RealBratsValidator:
    """
    Validates physical BraTS NIfTI image integrity, modal consistency, and label semantics.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def discover_subject_directories(self) -> List[Path]:
        """Discovers all subject directories (supports direct or nested training_data1_v2/ layouts)."""
        subjs = []
        if not self.data_dir.exists():
            return []

        for p in sorted(list(self.data_dir.iterdir())):
            if p.is_dir() and not p.name.startswith("."):
                if p.name.startswith("BraTS") or p.name.startswith("BraTS-GLI"):
                    subjs.append(p)
                elif p.name.startswith("training_data"):
                    for sub in sorted(list(p.iterdir())):
                        if sub.is_dir() and (sub.name.startswith("BraTS") or sub.name.startswith("BraTS-GLI")):
                            subjs.append(sub)

        return subjs

    def validate_subject(self, subject_dir: Path) -> Dict[str, Any]:
        subject_id = subject_dir.name
        version, mod_paths, missing = ModalityCanonicalizer.discover_subject_modalities(subject_dir)

        if missing:
            return {
                "subject_id": subject_id,
                "dataset_version": version.value,
                "valid": False,
                "reason": f"Missing required modalities: {', '.join(missing)}",
                "spatial_shape": None,
                "labels": [],
                "label_summary": {},
                "file_hashes": {},
            }

        try:
            shapes = []
            affines = []
            spacings = []
            file_hashes = {}

            for mod_key in ("t1", "t1ce", "t2", "flair", "seg"):
                p = mod_paths[mod_key]
                file_hashes[mod_key] = compute_sha256(p)
                img = nib.load(str(p))
                hdr = img.header
                shapes.append(tuple(img.shape))
                affines.append(img.affine)
                spacings.append(tuple(hdr.get_zooms()[:3]))

                # Check finite values
                arr = img.get_fdata(dtype=np.float32)
                if not np.isfinite(arr).all():
                    return {
                        "subject_id": subject_id,
                        "dataset_version": version.value,
                        "valid": False,
                        "reason": f"Non-finite values (NaN/Inf) detected in modality {mod_key}",
                        "spatial_shape": None,
                        "labels": [],
                        "label_summary": {},
                        "file_hashes": file_hashes,
                    }

                if mod_key == "seg":
                    seg_array = arr

            # Verify shape consistency across modalities
            if len(set(shapes)) != 1:
                return {
                    "subject_id": subject_id,
                    "dataset_version": version.value,
                    "valid": False,
                    "reason": f"Mismatched shapes across modalities: {shapes}",
                    "spatial_shape": shapes[0],
                    "labels": [],
                    "label_summary": {},
                    "file_hashes": file_hashes,
                }

            # Verify affine consistency
            ref_affine = affines[0]
            for idx, aff in enumerate(affines[1:], 1):
                if not np.allclose(ref_affine, aff, atol=1e-3):
                    return {
                        "subject_id": subject_id,
                        "dataset_version": version.value,
                        "valid": False,
                        "reason": "Affine matrices do not match across modalities",
                        "spatial_shape": shapes[0],
                        "labels": [],
                        "label_summary": {},
                        "file_hashes": file_hashes,
                    }

            # Verify label semantics
            targets, label_summary = LabelCanonicalizer.build_canonical_targets(seg_array, version=version)
            raw_labels = label_summary["raw_labels_present"]
            allowed_labels = {0, 1, 2, 3, 4} if version == DatasetVersion.BRATS_GLI_2024 else {0, 1, 2, 4}

            if not set(raw_labels).issubset(allowed_labels):
                invalid = set(raw_labels) - allowed_labels
                return {
                    "subject_id": subject_id,
                    "dataset_version": version.value,
                    "valid": False,
                    "reason": f"Invalid segmentation label values found: {invalid}",
                    "spatial_shape": list(shapes[0]),
                    "labels": raw_labels,
                    "label_summary": label_summary,
                    "file_hashes": file_hashes,
                }

            return {
                "subject_id": subject_id,
                "dataset_version": version.value,
                "valid": True,
                "reason": "OK",
                "spatial_shape": list(shapes[0]),
                "voxel_spacing": [float(x) for x in spacings[0]],
                "labels": raw_labels,
                "label_summary": label_summary,
                "file_hashes": file_hashes,
            }

        except Exception as e:
            return {
                "subject_id": subject_id,
                "dataset_version": version.value,
                "valid": False,
                "reason": f"Exception during validation: {str(e)}",
                "spatial_shape": None,
                "labels": [],
                "label_summary": {},
                "file_hashes": file_hashes,
            }


def discover_and_classify_dataset(data_dir: Path) -> Dict[str, Any]:
    validator = RealBratsValidator(data_dir)
    subject_dirs = validator.discover_subject_directories()

    manifest_entries = []
    valid_subjects = []
    invalid_subjects = []
    detected_version = DatasetVersion.UNKNOWN

    for s_dir in subject_dirs:
        v_res = validator.validate_subject(s_dir)
        manifest_entries.append(v_res)
        if v_res["valid"]:
            valid_subjects.append(v_res["subject_id"])
            detected_version = DatasetVersion(v_res["dataset_version"])
        else:
            invalid_subjects.append(v_res)

    is_real = len(valid_subjects) > 4
    if len(valid_subjects) <= 4:
        mode = "DEVELOPMENT_SYNTHETIC"
    elif detected_version == DatasetVersion.BRATS_GLI_2024:
        mode = "FULL_REAL_BRATS_GLI_2024"
    else:
        mode = "FULL_REAL_BRATS"

    return {
        "dataset_name": "BraTS-GLI" if detected_version == DatasetVersion.BRATS_GLI_2024 else "BraTS2021",
        "dataset_version": "2024" if detected_version == DatasetVersion.BRATS_GLI_2024 else "2021",
        "dataset_mode": mode,
        "is_real_brats": is_real,
        "total_subjects_found": len(subject_dirs),
        "valid_subjects_count": len(valid_subjects),
        "invalid_subjects_count": len(invalid_subjects),
        "valid_subjects": valid_subjects,
        "invalid_subjects": invalid_subjects,
    }


def create_deterministic_split(manifest: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    valid_subjects = sorted(manifest["valid_subjects"])
    rng = random.Random(seed)
    shuffled = list(valid_subjects)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    if n_total <= 4:
        # Development 4-case canonical split
        train_set = sorted(shuffled[:2])
        val_set = sorted(shuffled[2:3])
        test_set = sorted(shuffled[3:4])
    else:
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        train_set = sorted(shuffled[:n_train])
        val_set = sorted(shuffled[n_train : n_train + n_val])
        test_set = sorted(shuffled[n_train + n_val :])

    split_doc = {
        "dataset_name": manifest["dataset_name"],
        "dataset_version": manifest["dataset_version"],
        "seed": seed,
        "counts": {
            "total": n_total,
            "train": len(train_set),
            "validation": len(val_set),
            "test": len(test_set),
        },
        "leakage_verification": {
            "is_disjoint": (
                len(set(train_set) & set(val_set)) == 0
                and len(set(train_set) & set(test_set)) == 0
                and len(set(val_set) & set(test_set)) == 0
            ),
            "train_val_overlap": len(set(train_set) & set(val_set)),
            "train_test_overlap": len(set(train_set) & set(test_set)),
            "val_test_overlap": len(set(val_set) & set(test_set)),
        },
        "train_subjects": train_set,
        "validation_subjects": val_set,
        "test_subjects": test_set,
    }
    split_canonical_json = json.dumps(split_doc, sort_keys=True)
    split_doc["split_hash"] = hashlib.sha256(split_canonical_json.encode("utf-8")).hexdigest()
    return split_doc


def partition_hospitals(
    split: Dict[str, Any],
    hospitals: Tuple[str, ...] = ("hospital_alpha", "hospital_beta", "hospital_gamma", "hospital_delta"),
) -> Dict[str, Any]:
    train_subjects = split["train_subjects"]
    partitions = {h: [] for h in hospitals}

    for idx, subj in enumerate(train_subjects):
        target_h = hospitals[idx % len(hospitals)]
        partitions[target_h].append(subj)

    return {
        "dataset_name": split["dataset_name"],
        "dataset_version": split["dataset_version"],
        "hospital_count": len(hospitals),
        "partitions": {
            h: {
                "count": len(subjs),
                "status": "ACTIVE" if len(subjs) > 0 else "EMPTY_NO_TRAINING_DATA",
                "subjects": subjs,
            }
            for h, subjs in partitions.items()
        },
    }


def run_real_brats_ingestion_pipeline(
    data_dir: Path,
    output_reports_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    output_reports_dir.mkdir(parents=True, exist_ok=True)
    manifest = discover_and_classify_dataset(data_dir)
    split = create_deterministic_split(manifest, seed=seed)
    partitions = partition_hospitals(split)

    with open(output_reports_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(output_reports_dir / "dataset_split.json", "w") as f:
        json.dump(split, f, indent=2)

    with open(output_reports_dir / "hospital_partitions.json", "w") as f:
        json.dump(partitions, f, indent=2)

    return {
        "manifest": manifest,
        "split": split,
        "partitions": partitions,
    }


def run_dataset_ingestion(
    data_dir: Path,
    output_reports_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    return run_real_brats_ingestion_pipeline(data_dir, output_reports_dir, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS Dataset Validator and Partitioner")
    parser.add_argument("--data-dir", type=str, default="data/raw/BraTS2024", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="reports/real_brats2024", help="Output reports directory")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    run_real_brats_ingestion_pipeline(
        data_dir=Path(args.data_dir),
        output_reports_dir=Path(args.output_dir),
        seed=args.seed,
    )
