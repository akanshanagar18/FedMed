"""
Unit tests for inference/validator.py and inference/pipeline.py.
Verifies clinical input validation, structured failure modes, output validation,
and inference pipeline execution.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
import torch

from inference.pipeline import ClinicalInferenceEngine
from inference.validator import ClinicalErrorCode, ClinicalInputValidator, ClinicalOutputValidator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_clinical_input_validator_valid_and_missing_modalities():
    validator = ClinicalInputValidator()
    subj_dir = PROJECT_ROOT / "data" / "BraTS2021" / "BraTS2021_00002"

    mod_paths = {
        m: sorted(list(subj_dir.glob(f"*{m}.nii*")))[0]
        for m in ("t1", "t1ce", "t2", "flair")
    }

    # 1. Valid Input
    is_valid, code, msg, meta = validator.validate_modalities_dict(mod_paths)
    assert is_valid is True
    assert code == ClinicalErrorCode.OK
    assert meta["spatial_shape"] == [32, 32, 32]

    # 2. Missing Modality Failure
    incomplete_paths = {k: v for k, v in mod_paths.items() if k != "flair"}
    is_valid_inc, code_inc, msg_inc, _ = validator.validate_modalities_dict(incomplete_paths)
    assert is_valid_inc is False
    assert code_inc == ClinicalErrorCode.MISSING_MODALITY


def test_clinical_output_validator():
    validator = ClinicalOutputValidator()

    # Valid output (3, 32, 32, 32)
    valid_seg = np.zeros((3, 32, 32, 32), dtype=np.float32)
    is_valid, code, msg = validator.validate_segmentation_output(valid_seg, expected_shape=(32, 32, 32), allowed_channels=3)
    assert is_valid is True
    assert code == ClinicalErrorCode.OK

    # Shape mismatch failure
    invalid_seg = np.zeros((2, 32, 32, 32), dtype=np.float32)
    is_valid_inv, code_inv, _ = validator.validate_segmentation_output(invalid_seg, expected_shape=(32, 32, 32), allowed_channels=3)
    assert is_valid_inv is False
    assert code_inv == ClinicalErrorCode.SHAPE_MISMATCH

    # Non-finite values failure
    nan_seg = np.zeros((3, 32, 32, 32), dtype=np.float32)
    nan_seg[0, 0, 0, 0] = np.nan
    is_valid_nan, code_nan, _ = validator.validate_segmentation_output(nan_seg, expected_shape=(32, 32, 32), allowed_channels=3)
    assert is_valid_nan is False
    assert code_nan == ClinicalErrorCode.NONFINITE_IMAGE


def test_clinical_inference_engine_execution():
    ckpt_path = PROJECT_ROOT / "checkpoints" / "fedavg" / "global_best.pt"
    engine = ClinicalInferenceEngine(checkpoint_path=ckpt_path)

    subj_dir = PROJECT_ROOT / "data" / "BraTS2021" / "BraTS2021_00002"
    mod_paths = {
        m: sorted(list(subj_dir.glob(f"*{m}.nii*")))[0]
        for m in ("t1", "t1ce", "t2", "flair")
    }

    res = engine.predict_subject(mod_paths)
    assert res["status"] == "SUCCESS"
    assert res["error_code"] == "OK"
    assert "timing_ms" in res
    assert res["timing_ms"]["total_inference_time_ms"] > 0.0
    assert "segmented_voxels" in res
