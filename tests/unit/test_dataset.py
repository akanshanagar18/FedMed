"""
Unit tests for data.datasets.brats BraTSDataset.
"""

from pathlib import Path
import pytest
from data.datasets.brats import BraTSDataset, generate_synthetic_brats_nifti


def test_synthetic_brats_nifti_generation(tmp_path):
    """Verify synthetic NIfTI files generation."""
    data_dir = generate_synthetic_brats_nifti(tmp_path, num_subjects=2, spatial_shape=(16, 16, 16))
    assert data_dir.exists()
    subjects = list(data_dir.glob("BraTS2021_*"))
    assert len(subjects) == 2

    # Check 4 modalities + seg mask exist for subject 1
    subj1 = subjects[0]
    for mod in ["t1", "t1ce", "t2", "flair", "seg"]:
        matches = list(subj1.glob(f"*{mod}.nii.gz"))
        assert len(matches) == 1


def test_brats_dataset_initialization(tmp_path):
    """Verify BraTSDataset subject indexing and split ratios."""
    generate_synthetic_brats_nifti(tmp_path, num_subjects=4, spatial_shape=(16, 16, 16))

    ds = BraTSDataset(
        data_dir=tmp_path,
        cache_type="dataset",
        val_split=0.25,
        allow_synthetic_fallback=False,
    )

    assert len(ds.patient_metadata) == 4
    assert len(ds.train_files) == 3
    assert len(ds.val_files) == 1

    train_monai = ds.get_train_dataset()
    val_monai = ds.get_val_dataset()
    assert len(train_monai) == 3
    assert len(val_monai) == 1
