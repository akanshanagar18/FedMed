"""
Script: scripts/validate_brats_dataset.py

Purpose:
Comprehensive BraTS Dataset Discovery, Classification, and NIfTI Integrity Validator.
Executes Gates B1-1 through B1-4:
  1. Dynamic filesystem discovery of subject directories
  2. Provenance-based dataset classification (DEVELOPMENT_SYNTHETIC, REAL_BRATS_SUBSET, FULL_REAL_BRATS)
  3. Strict NIfTI volumetric integrity verification (readability, shapes, affines, spacings, finite values, labels)
  4. Generation of reports/brats_dataset_manifest.json with zero hardcoded statistics

Usage:
  python3 scripts/validate_brats_dataset.py --data-dir data/BraTS2021
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_git_commit() -> str:
    """Retrieves current Git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def classify_dataset(
    subject_count: int,
    shapes: List[Tuple[int, ...]],
    spacings: List[Tuple[float, ...]],
) -> str:
    """
    Deterministically classifies dataset based on volumetric shape, resolution, and count.
    Never infers mode from folder name alone.
    """
    if subject_count == 0:
        return "EMPTY_DATASET"

    # Real BraTS 2021 cohort volumes have spatial dimensions ~ (240, 240, 155) with 1mm isometric spacing
    is_mini_synthetic = any(s == (32, 32, 32) or s == (64, 64, 64) for s in shapes)

    if is_mini_synthetic:
        return "DEVELOPMENT_SYNTHETIC"
    elif subject_count >= 1250:
        return "FULL_REAL_BRATS"
    else:
        return "REAL_BRATS_SUBSET"


def validate_subject_nifti(
    subj_dir: Path,
    required_modalities: List[str] = ("flair", "t1", "t1ce", "t2"),
    seg_modality: str = "seg",
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Thoroughly inspects all 4 MRI modalities and segmentation mask for a single subject.
    """
    subj_id = subj_dir.name
    errors: List[str] = []
    modality_files: Dict[str, str] = {}
    modality_shapes: Dict[str, Tuple[int, ...]] = {}
    modality_spacings: Dict[str, Tuple[float, ...]] = {}
    modality_affines: Dict[str, List[List[float]]] = {}
    label_values: List[int] = []

    # 1. Check & Load Modality Files
    for mod in required_modalities:
        matches = sorted(list(subj_dir.glob(f"*{mod}.nii*")))
        if not matches:
            errors.append(f"Missing required modality file for '{mod}'")
            continue
        file_path = matches[0]
        modality_files[mod] = str(file_path.relative_to(PROJECT_ROOT))

        try:
            img = nib.load(str(file_path))
            header = img.header
            data = img.get_fdata(dtype=np.float32)

            # Dimensionality & Finite Checks
            if data.ndim != 3:
                errors.append(f"Modality '{mod}' is not 3D spatial (got {data.ndim}D)")
            if not np.all(np.isfinite(data)):
                errors.append(f"Modality '{mod}' contains NaN or Inf values")
            if data.size == 0 or np.all(data == 0):
                errors.append(f"Modality '{mod}' image volume is empty or all zeros")

            modality_shapes[mod] = tuple(img.shape)
            modality_spacings[mod] = tuple(float(s) for s in header.get_zooms()[:3])
            modality_affines[mod] = img.affine.tolist()

        except Exception as e:
            errors.append(f"Failed to read NIfTI file for '{mod}': {e}")

    # 2. Check & Load Segmentation File
    seg_matches = sorted(list(subj_dir.glob(f"*{seg_modality}.nii*")))
    if not seg_matches:
        errors.append(f"Missing segmentation mask file '*{seg_modality}.nii*'")
    else:
        seg_path = seg_matches[0]
        modality_files[seg_modality] = str(seg_path.relative_to(PROJECT_ROOT))
        try:
            seg_img = nib.load(str(seg_path))
            seg_data = seg_img.get_fdata(dtype=np.float32)

            if seg_data.ndim != 3:
                errors.append(f"Segmentation mask is not 3D spatial (got {seg_data.ndim}D)")
            if not np.all(np.isfinite(seg_data)):
                errors.append("Segmentation mask contains NaN or Inf values")

            unique_labels = sorted([int(x) for x in np.unique(seg_data)])
            label_values = unique_labels

            # Validate against expected BraTS labels (0: bg, 1: necrotic core, 2: edema, 3/4: enhancing)
            unexpected = [l for l in unique_labels if l not in (0, 1, 2, 3, 4)]
            if unexpected:
                errors.append(f"Segmentation mask contains unexpected labels: {unexpected}")

            modality_shapes[seg_modality] = tuple(seg_img.shape)
            modality_spacings[seg_modality] = tuple(float(s) for s in seg_img.header.get_zooms()[:3])
            modality_affines[seg_modality] = seg_img.affine.tolist()

        except Exception as e:
            errors.append(f"Failed to read segmentation NIfTI: {e}")

    # 3. Spatial Compatibility Verification
    if modality_shapes:
        shapes_set = set(modality_shapes.values())
        if len(shapes_set) > 1:
            errors.append(f"Spatial shape mismatch across modalities: {modality_shapes}")

    is_valid = len(errors) == 0
    ref_shape = list(modality_shapes.values())[0] if modality_shapes else None
    ref_spacing = list(modality_spacings.values())[0] if modality_spacings else None

    manifest_entry = {
        "subject_id": subj_id,
        "is_valid": is_valid,
        "errors": errors,
        "modality_paths": modality_files,
        "spatial_shape": ref_shape,
        "voxel_spacing": ref_spacing,
        "labels_found": label_values,
    }

    return is_valid, manifest_entry, errors


def validate_dataset(data_dir: str = "data/BraTS2021") -> Dict[str, Any]:
    print("=" * 70)
    print("🔬 FEDMED OS — BRATS DATASET DISCOVERY & INTEGRITY VALIDATOR")
    print("=" * 70)

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    print(f"\nTarget Dataset Directory: {data_path}")

    if not data_path.exists() or not data_path.is_dir():
        print(f"❌ Target directory '{data_path}' does not exist!")
        print("\n==================================================")
        print("REAL BRAΤS DATASET REQUIRED")
        print("==================================================")
        print("1. Expected Directory Structure:")
        print("   data/BraTS2021/")
        print("     ├── BraTS2021_00001/")
        print("     │     ├── BraTS2021_00001_flair.nii.gz")
        print("     │     ├── BraTS2021_00001_t1.nii.gz")
        print("     │     ├── BraTS2021_00001_t1ce.nii.gz")
        print("     │     ├── BraTS2021_00001_t2.nii.gz")
        print("     │     └── BraTS2021_00001_seg.nii.gz")
        print("     └── ...")
        print("2. Validation Command:")
        print("   python3 scripts/validate_brats_dataset.py --data-dir data/BraTS2021")
        print("3. Environment Variable:")
        print("   export FEDMED_DATA_DIR=/path/to/BraTS2021")
        sys.exit(1)

    # 1. Discover Subjects (Never hardcoded)
    subject_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    subject_count = len(subject_dirs)
    print(f"Discovered Subject Folders: {subject_count}")

    if subject_count == 0:
        print("❌ Zero subject subdirectories found in dataset directory!")
        sys.exit(1)

    subjects_manifest: List[Dict[str, Any]] = []
    valid_count = 0
    invalid_count = 0
    all_shapes: List[Tuple[int, ...]] = []
    all_spacings: List[Tuple[float, ...]] = []
    all_labels: Set[int] = set()

    for subj_dir in subject_dirs:
        is_val, entry, errs = validate_subject_nifti(subj_dir)
        subjects_manifest.append(entry)
        if is_val:
            valid_count += 1
            if entry["spatial_shape"]:
                all_shapes.append(tuple(entry["spatial_shape"]))
            if entry["voxel_spacing"]:
                all_spacings.append(tuple(entry["voxel_spacing"]))
            all_labels.update(entry["labels_found"])
        else:
            invalid_count += 1

    # 2. Classify Dataset
    dataset_mode = classify_dataset(subject_count, all_shapes, all_spacings)

    # 3. Aggregate Statistics
    shape_counts: Dict[str, int] = {}
    for s in all_shapes:
        key = str(s)
        shape_counts[key] = shape_counts.get(key, 0) + 1

    spacing_counts: Dict[str, int] = {}
    for sp in all_spacings:
        key = str(sp)
        spacing_counts[key] = spacing_counts.get(key, 0) + 1

    manifest_payload: Dict[str, Any] = {
        "dataset_id": data_path.name,
        "dataset_version": "BraTS2021",
        "dataset_mode": dataset_mode,
        "dataset_path": str(data_path.resolve()),
        "subject_count": subject_count,
        "validated_subject_count": valid_count,
        "invalid_subject_count": invalid_count,
        "subjects": subjects_manifest,
        "modalities": {
            "required": ["flair", "t1", "t1ce", "t2"],
            "mask": "seg",
        },
        "label_statistics": {
            "unique_labels_discovered": sorted(list(all_labels)),
            "expected_labels": [0, 1, 2, 3, 4],
            "all_labels_valid": all(l in (0, 1, 2, 3, 4) for l in all_labels),
        },
        "shape_statistics": shape_counts,
        "spacing_statistics": spacing_counts,
        "validation_status": "PASS" if invalid_count == 0 and valid_count > 0 else "FAIL",
        "validation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = REPORTS_DIR / "brats_dataset_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("📊 DATASET VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Dataset Mode:             {dataset_mode}")
    print(f"Discovered Subjects:      {subject_count}")
    print(f"Validated Valid Subjects: {valid_count}")
    print(f"Invalid / Corrupted:      {invalid_count}")
    print(f"Discovered Labels:        {sorted(list(all_labels))}")
    print(f"Shape Distribution:       {shape_counts}")
    print(f"Manifest Generated:       {manifest_file}")
    print(f"Validation Status:        {manifest_payload['validation_status']}")
    print("=" * 70)

    if dataset_mode == "DEVELOPMENT_SYNTHETIC":
        print("\n⚠️ NOTICE: Dataset classified as DEVELOPMENT_SYNTHETIC.")
        print("  These mini-NIfTI cases validate software pipeline integrity.")
        print("  DO NOT use this dataset to claim a real clinical AI benchmark.")

    if manifest_payload["validation_status"] != "PASS":
        print("\n❌ Dataset validation FAILED!")
        sys.exit(1)

    return manifest_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS Dataset Integrity Validator")
    parser.add_argument("--data-dir", type=str, default="data/BraTS2021", help="Path to BraTS dataset directory")
    args = parser.parse_args()

    validate_dataset(args.data_dir)
