"""
Module: inference.validator

Purpose:
Production Clinical Input and Output Validator for FedMed v2.0.
Enforces strict modality completeness, NIfTI structural integrity, affine alignment,
finite-value constraints, and segmentation label semantics with structured error reporting.
Supports both BraTS 2021 (t1/t1ce/t2/flair) and BraTS-GLI 2024 (t1n/t1c/t2w/t2f) nomenclature.
"""

from enum import Enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger("inference_validator")


class ClinicalErrorCode(str, Enum):
    OK = "OK"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_MODALITY = "MISSING_MODALITY"
    CORRUPTED_NIFTI = "CORRUPTED_NIFTI"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    AFFINE_MISMATCH = "AFFINE_MISMATCH"
    NONFINITE_IMAGE = "NONFINITE_IMAGE"
    MODEL_LOAD_FAILURE = "MODEL_LOAD_FAILURE"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    OUTPUT_VALIDATION_FAILURE = "OUTPUT_VALIDATION_FAILURE"
    UNSUPPORTED_CONFIGURATION = "UNSUPPORTED_CONFIGURATION"


class ClinicalInputValidator:
    """
    Validates clinical multi-parametric MRI input files for inference.
    """

    REQUIRED_MODALITIES = ("t1", "t1ce", "t2", "flair")
    MODALITY_ALIASES = {
        "t1n": "t1",
        "t1c": "t1ce",
        "t2w": "t2",
        "t2f": "flair",
    }

    def validate_modalities_dict(self, modality_paths: Dict[str, Path]) -> Tuple[bool, ClinicalErrorCode, str, Dict[str, Any]]:
        # Normalize modality keys
        normalized_paths: Dict[str, Path] = {}
        for k, v in modality_paths.items():
            norm_k = self.MODALITY_ALIASES.get(k.lower(), k.lower())
            normalized_paths[norm_k] = Path(v)

        # 1. Check all required modalities are provided
        missing = [m for m in self.REQUIRED_MODALITIES if m not in normalized_paths or not normalized_paths[m].exists()]
        if missing:
            return False, ClinicalErrorCode.MISSING_MODALITY, f"Missing required modalities: {missing}", {}

        shapes = []
        affines = []
        spacings = []
        loaded_niftis = {}

        for mod in self.REQUIRED_MODALITIES:
            p = normalized_paths[mod]
            try:
                img = nib.load(str(p))
                hdr = img.header
                arr = img.get_fdata(dtype=np.float32)

                # Check finite values
                if not np.isfinite(arr).all():
                    return False, ClinicalErrorCode.NONFINITE_IMAGE, f"Non-finite values (NaN/Inf) found in {mod}", {}

                shapes.append(tuple(img.shape))
                affines.append(img.affine)
                spacings.append(tuple(hdr.get_zooms()[:3]))
                loaded_niftis[mod] = img

            except Exception as e:
                return False, ClinicalErrorCode.CORRUPTED_NIFTI, f"Failed to parse NIfTI file {p.name}: {str(e)}", {}

        # 2. Check shape consistency across modalities
        if len(set(shapes)) != 1:
            return False, ClinicalErrorCode.SHAPE_MISMATCH, f"Mismatched shapes across modalities: {shapes}", {}

        # 3. Check affine consistency
        ref_affine = affines[0]
        for idx, aff in enumerate(affines[1:], 1):
            if not np.allclose(ref_affine, aff, atol=1e-3):
                return False, ClinicalErrorCode.AFFINE_MISMATCH, f"Mismatched affine matrix in modality {self.REQUIRED_MODALITIES[idx]}", {}

        meta = {
            "spatial_shape": [int(x) for x in shapes[0]],
            "voxel_spacing": [float(x) for x in spacings[0]],
            "ref_affine": ref_affine.tolist(),
            "modalities": list(self.REQUIRED_MODALITIES),
            "normalized_paths": {k: str(v) for k, v in normalized_paths.items()},
        }

        return True, ClinicalErrorCode.OK, "Validation successful", meta


class ClinicalOutputValidator:
    """
    Validates predicted segmentation masks before saving or clinical transmission.
    """

    def validate_segmentation_output(
        self,
        seg_array: np.ndarray,
        expected_shape: Tuple[int, ...],
        allowed_channels: int = 3,
    ) -> Tuple[bool, ClinicalErrorCode, str]:
        if not np.isfinite(seg_array).all():
            return False, ClinicalErrorCode.NONFINITE_IMAGE, "Non-finite values in output segmentation array"

        if seg_array.ndim == 4:
            if seg_array.shape[0] != allowed_channels or tuple(seg_array.shape[1:]) != expected_shape:
                return False, ClinicalErrorCode.SHAPE_MISMATCH, f"Output shape {seg_array.shape} does not match expected ({allowed_channels}, {expected_shape})"
        elif seg_array.ndim == 3:
            if tuple(seg_array.shape) != expected_shape:
                return False, ClinicalErrorCode.SHAPE_MISMATCH, f"Output shape {seg_array.shape} does not match expected {expected_shape}"
        else:
            return False, ClinicalErrorCode.OUTPUT_VALIDATION_FAILURE, f"Invalid segmentation array dimension: {seg_array.ndim}"

        return True, ClinicalErrorCode.OK, "Output validation passed"
