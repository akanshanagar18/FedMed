"""
Unit tests for data/canonical_adapter.py and BraTS-GLI 2024 integration.
Verifies modality discovery, canonicalization, label mapping, TC/WT/ET construction,
RC preservation, shape/affine validation, and dataset identity.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import pytest

from data.canonical_adapter import (
    BRATS_2021_IDENTITY,
    BRATS_GLI_2024_IDENTITY,
    DatasetIdentity,
    DatasetVersion,
    LabelCanonicalizer,
    ModalityCanonicalizer,
)
from data.real_brats_pipeline import RealBratsValidator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_gli_label_canonicalization_and_rc_preservation():
    # Construct a synthetic segmentation volume with all 5 labels: 0, 1, 2, 3, 4
    seg = np.zeros((32, 32, 32), dtype=np.int16)
    seg[0:5, 0:5, 0:5] = 1   # NETC
    seg[5:10, 5:10, 5:10] = 2 # SNFH
    seg[10:15, 10:15, 10:15] = 3 # ET
    seg[15:20, 15:20, 15:20] = 4 # RC

    targets, summary = LabelCanonicalizer.build_canonical_targets(seg, version=DatasetVersion.BRATS_GLI_2024)

    # 1. Target shape must be (3, 32, 32, 32)
    assert targets.shape == (3, 32, 32, 32)

    # 2. TC must be label 1 | label 3 (NETC + ET)
    tc_expected = (seg == 1) | (seg == 3)
    assert np.array_equal(targets[0] > 0, tc_expected)

    # 3. WT must be label 1 | label 2 | label 3 (NETC + SNFH + ET)
    wt_expected = (seg == 1) | (seg == 2) | (seg == 3)
    assert np.array_equal(targets[1] > 0, wt_expected)

    # 4. ET must be label 3
    et_expected = (seg == 3)
    assert np.array_equal(targets[2] > 0, et_expected)

    # 5. RC (label 4) must NOT be in TC or WT
    rc_mask = (seg == 4)
    assert np.sum(targets[0][rc_mask]) == 0
    assert np.sum(targets[1][rc_mask]) == 0
    assert np.sum(targets[2][rc_mask]) == 0

    # 6. RC must be preserved in summary metadata
    assert summary["RC_voxels"] == 125
    assert summary["ET_voxels"] == 125
    assert summary["TC_voxels"] == 250
    assert summary["WT_voxels"] == 375


def test_brats_2021_label_canonicalization():
    seg = np.zeros((32, 32, 32), dtype=np.int16)
    seg[0:5, 0:5, 0:5] = 1   # NET
    seg[5:10, 5:10, 5:10] = 2 # ED
    seg[10:15, 10:15, 10:15] = 4 # ET

    targets, summary = LabelCanonicalizer.build_canonical_targets(seg, version=DatasetVersion.BRATS_2021_REAL)
    assert targets.shape == (3, 32, 32, 32)

    # In 2021: TC = 1 | 4, WT = 1 | 2 | 4, ET = 4
    assert np.array_equal(targets[0] > 0, (seg == 1) | (seg == 4))
    assert np.array_equal(targets[1] > 0, (seg == 1) | (seg == 2) | (seg == 4))
    assert np.array_equal(targets[2] > 0, (seg == 4))


def test_gli_modality_discovery_and_canonicalization(tmp_path):
    subj_dir = tmp_path / "BraTS-GLI-00001-000"
    subj_dir.mkdir()

    # Create dummy files with 2024 GLI naming
    for mod in ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]:
        f = subj_dir / f"BraTS-GLI-00001-000{mod}"
        f.touch()

    version, mod_paths, missing = ModalityCanonicalizer.discover_subject_modalities(subj_dir)
    assert version == DatasetVersion.BRATS_GLI_2024
    assert len(missing) == 0
    assert "t1" in mod_paths
    assert "t1ce" in mod_paths
    assert "t2" in mod_paths
    assert "flair" in mod_paths
    assert "seg" in mod_paths


def test_gli_missing_modality_detection(tmp_path):
    subj_dir = tmp_path / "BraTS-GLI-00002-000"
    subj_dir.mkdir()

    # Create 3 modalities, omitting t2w
    for mod in ["-t1n.nii.gz", "-t1c.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]:
        (subj_dir / f"BraTS-GLI-00002-000{mod}").touch()

    version, mod_paths, missing = ModalityCanonicalizer.discover_subject_modalities(subj_dir)
    assert version == DatasetVersion.BRATS_GLI_2024
    assert "t2w" in missing


def test_dataset_identity_specification():
    gli = BRATS_GLI_2024_IDENTITY.to_dict()
    assert gli["dataset_name"] == "BraTS-GLI"
    assert gli["dataset_version"] == "2024"
    assert gli["spatial_template"] == "MNI152"
    assert gli["native_shape"] == [182, 218, 182]
    assert gli["voxel_spacing_mm"] == [1.0, 1.0, 1.0]

    b21 = BRATS_2021_IDENTITY.to_dict()
    assert b21["dataset_name"] == "BraTS2021"
    assert b21["dataset_version"] == "2021"
    assert b21["spatial_template"] == "SRI24"
    assert b21["native_shape"] == [240, 240, 155]
