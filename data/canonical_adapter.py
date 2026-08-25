"""
Module: data.canonical_adapter

Purpose:
Canonical Dataset and Label Adapter for FedMed v2.0.
Provides unified, non-destructive canonicalization for both BraTS 2021 and BraTS-GLI 2024 cohorts:
1. Maps dataset-specific raw filenames (t1n/t1c/t2w/t2f vs t1/t1ce/t2/flair) into canonical 4-channel inputs.
2. Converts dataset-specific ground-truth label semantics into canonical 3-channel composite targets (TC, WT, ET),
   while preserving Resection Cavity (RC) in metadata and raw provenance.
"""

from enum import Enum
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger("canonical_adapter")


class DatasetVersion(str, Enum):
    BRATS_2021_SYNTHETIC = "BRATS_2021_SYNTHETIC"
    BRATS_2021_REAL = "BRATS_2021_REAL"
    BRATS_GLI_2024 = "BRATS_GLI_2024"
    UNKNOWN = "UNKNOWN"


class DatasetIdentity:
    def __init__(
        self,
        dataset_name: str,
        dataset_version: str,
        dataset_task: str,
        spatial_template: str,
        native_shape: Tuple[int, ...],
        voxel_spacing_mm: Tuple[float, ...],
    ):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.dataset_task = dataset_task
        self.spatial_template = spatial_template
        self.native_shape = native_shape
        self.voxel_spacing_mm = voxel_spacing_mm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_task": self.dataset_task,
            "spatial_template": self.spatial_template,
            "native_shape": list(self.native_shape),
            "voxel_spacing_mm": list(self.voxel_spacing_mm),
        }


BRATS_2021_IDENTITY = DatasetIdentity(
    dataset_name="BraTS2021",
    dataset_version="2021",
    dataset_task="adult_glioma_segmentation",
    spatial_template="SRI24",
    native_shape=(240, 240, 155),
    voxel_spacing_mm=(1.0, 1.0, 1.0),
)

BRATS_GLI_2024_IDENTITY = DatasetIdentity(
    dataset_name="BraTS-GLI",
    dataset_version="2024",
    dataset_task="adult_glioma_post_treatment",
    spatial_template="MNI152",
    native_shape=(182, 218, 182),
    voxel_spacing_mm=(1.0, 1.0, 1.0),
)


class ModalityCanonicalizer:
    """
    Maps heterogeneous dataset file naming conventions into canonical 4-channel representation:
    Channel 0 = T1 (Native)
    Channel 1 = T1CE (Post-contrast)
    Channel 2 = T2 (T2-weighted)
    Channel 3 = FLAIR (T2-FLAIR)
    Mask = SEG
    """

    CANONICAL_MODALITIES = ("t1", "t1ce", "t2", "flair")

    @staticmethod
    def discover_subject_modalities(subject_dir: Path) -> Tuple[DatasetVersion, Dict[str, Path], List[str]]:
        subject_id = subject_dir.name
        mod_paths: Dict[str, Path] = {}
        missing: List[str] = []

        # 1. Check BraTS-GLI 2024 Pattern (*-t1n.nii*, *-t1c.nii*, *-t2w.nii*, *-t2f.nii*, *-seg.nii*)
        t1n_files = sorted(list(subject_dir.glob("*-t1n.nii*")) + list(subject_dir.glob("*_t1n.nii*")))
        t1c_files = sorted(list(subject_dir.glob("*-t1c.nii*")) + list(subject_dir.glob("*_t1c.nii*")))
        t2w_files = sorted(list(subject_dir.glob("*-t2w.nii*")) + list(subject_dir.glob("*_t2w.nii*")))
        t2f_files = sorted(list(subject_dir.glob("*-t2f.nii*")) + list(subject_dir.glob("*_t2f.nii*")))
        seg_gli_files = sorted(list(subject_dir.glob("*-seg.nii*")) + list(subject_dir.glob("*_seg.nii*")))

        if t1n_files or t1c_files or t2w_files or t2f_files:
            version = DatasetVersion.BRATS_GLI_2024
            if t1n_files:
                mod_paths["t1"] = t1n_files[0]
            else:
                missing.append("t1n")

            if t1c_files:
                mod_paths["t1ce"] = t1c_files[0]
            else:
                missing.append("t1c")

            if t2w_files:
                mod_paths["t2"] = t2w_files[0]
            else:
                missing.append("t2w")

            if t2f_files:
                mod_paths["flair"] = t2f_files[0]
            else:
                missing.append("t2f")

            if seg_gli_files:
                mod_paths["seg"] = seg_gli_files[0]
            else:
                missing.append("seg")

            return version, mod_paths, missing

        # 2. Check BraTS 2021 Pattern (*_t1.nii*, *_t1ce.nii*, *_t2.nii*, *_flair.nii*, *_seg.nii*)
        t1_files = sorted(list(subject_dir.glob("*_t1.nii*")) + list(subject_dir.glob("*-t1.nii*")))
        t1ce_files = sorted(list(subject_dir.glob("*_t1ce.nii*")) + list(subject_dir.glob("*-t1ce.nii*")))
        t2_files = sorted(list(subject_dir.glob("*_t2.nii*")) + list(subject_dir.glob("*-t2.nii*")))
        flair_files = sorted(list(subject_dir.glob("*_flair.nii*")) + list(subject_dir.glob("*-flair.nii*")))
        seg_21_files = sorted(list(subject_dir.glob("*_seg.nii*")) + list(subject_dir.glob("*-seg.nii*")))

        if t1_files or t1ce_files or t2_files or flair_files:
            # Check if synthetic mini-cases
            version = DatasetVersion.BRATS_2021_REAL
            if "0000" in subject_id and int(subject_id.split("_")[-1]) <= 4:
                version = DatasetVersion.BRATS_2021_SYNTHETIC

            if t1_files:
                mod_paths["t1"] = t1_files[0]
            else:
                missing.append("t1")

            if t1ce_files:
                mod_paths["t1ce"] = t1ce_files[0]
            else:
                missing.append("t1ce")

            if t2_files:
                mod_paths["t2"] = t2_files[0]
            else:
                missing.append("t2")

            if flair_files:
                mod_paths["flair"] = flair_files[0]
            else:
                missing.append("flair")

            if seg_21_files:
                mod_paths["seg"] = seg_21_files[0]
            else:
                missing.append("seg")

            return version, mod_paths, missing

        return DatasetVersion.UNKNOWN, {}, ["unrecognized_naming_convention"]


class LabelCanonicalizer:
    """
    Canonicalizes raw ground truth segmentation labels into standard multi-channel binary targets (TC, WT, ET)
    according to dataset-specific schema.
    """

    @staticmethod
    def build_canonical_targets(
        seg_array: np.ndarray,
        version: DatasetVersion = DatasetVersion.BRATS_GLI_2024,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Builds 3-channel composite binary targets:
        Channel 0: TC (Tumor Core)
        Channel 1: WT (Whole Tumor)
        Channel 2: ET (Enhancing Tumor)

        Returns:
            targets: (3, H, W, D) binary float32 tensor
            voxel_counts: dictionary of raw and composite voxel counts (including RC)
        """
        seg = seg_array.astype(np.int16)
        h, w, d = seg.shape[-3:]

        # Raw label counts
        unique, counts = np.unique(seg, return_counts=True)
        raw_counts = dict(zip([int(x) for x in unique], [int(x) for x in counts]))

        tc_mask = np.zeros((h, w, d), dtype=bool)
        wt_mask = np.zeros((h, w, d), dtype=bool)
        et_mask = np.zeros((h, w, d), dtype=bool)
        rc_voxels = 0

        if version == DatasetVersion.BRATS_GLI_2024:
            # BraTS-GLI 2024 Official Schema:
            # 0: BG, 1: NETC, 2: SNFH, 3: ET, 4: RC
            # Composite targets:
            # TC = label 1 | label 3
            # WT = label 1 | label 2 | label 3
            # ET = label 3
            # RC = label 4 (NOT included in TC or WT)
            tc_mask = (seg == 1) | (seg == 3)
            wt_mask = (seg == 1) | (seg == 2) | (seg == 3)
            et_mask = (seg == 3)
            rc_voxels = raw_counts.get(4, 0)
        else:
            # BraTS 2021 Schema:
            # 0: BG, 1: NET, 2: ED, 4: ET
            # Composite targets:
            # TC = label 1 | label 4
            # WT = label 1 | label 2 | label 4
            # ET = label 4
            tc_mask = (seg == 1) | (seg == 4)
            wt_mask = (seg == 1) | (seg == 2) | (seg == 4)
            et_mask = (seg == 4)
            rc_voxels = 0

        targets = np.stack([tc_mask, wt_mask, et_mask], axis=0).astype(np.float32)

        voxel_summary = {
            "raw_labels_present": list(raw_counts.keys()),
            "raw_counts": raw_counts,
            "TC_voxels": int(np.sum(tc_mask)),
            "WT_voxels": int(np.sum(wt_mask)),
            "ET_voxels": int(np.sum(et_mask)),
            "RC_voxels": rc_voxels,
        }

        return targets, voxel_summary
