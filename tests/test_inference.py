"""
FedMed - Inference Smoke Tests

Tests the important inference utilities without
requiring model retraining.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from configs.inference_config import (
    NUM_CLASSES,
    ROI_SIZE,
    SW_BATCH_SIZE,
    SW_OVERLAP,
)

from src.inference.evaluation import (
    calculate_dice,
    calculate_iou,
)

from src.inference.output import (
    get_case_name,
    get_prediction_path,
    save_segmentation_nifti,
)


def test_inference_configuration():
    """
    Verify the main inference configuration values.
    """

    assert ROI_SIZE == (
        96,
        96,
        96,
    )

    assert SW_BATCH_SIZE == 1

    assert SW_OVERLAP == 0.25

    assert NUM_CLASSES == 4


def test_case_name_extraction():
    """
    Verify NIfTI case-name extraction.
    """

    assert (
        get_case_name(
            "BRATS_001.nii.gz"
        )
        == "BRATS_001"
    )

    assert (
        get_case_name(
            "BRATS_002.nii"
        )
        == "BRATS_002"
    )


def test_prediction_path_generation():
    """
    Verify automatic prediction path generation.
    """

    path = get_prediction_path(
        "inference_data/BRATS_001.nii.gz"
    )

    assert path.name == (
        "BRATS_001_prediction.nii.gz"
    )

    assert (
        path.parent.name
        == "predictions"
    )


def test_dice_perfect_overlap():
    """
    Dice should be 1.0 for identical masks.
    """

    prediction = np.array(
        [1, 1, 0, 0]
    )

    target = np.array(
        [1, 1, 0, 0]
    )

    score = calculate_dice(
        prediction,
        target,
    )

    assert score == 1.0


def test_dice_partial_overlap():
    """
    Dice should be 0.5 for this known overlap.
    """

    prediction = np.array(
        [1, 1, 0, 0]
    )

    target = np.array(
        [1, 0, 1, 0]
    )

    score = calculate_dice(
        prediction,
        target,
    )

    assert np.isclose(
        score,
        0.5,
    )


def test_iou_partial_overlap():
    """
    IoU should be 1/3 for this known overlap.
    """

    prediction = np.array(
        [1, 1, 0, 0]
    )

    target = np.array(
        [1, 0, 1, 0]
    )

    score = calculate_iou(
        prediction,
        target,
    )

    assert np.isclose(
        score,
        1.0 / 3.0,
    )


def test_empty_masks():
    """
    Two empty masks should receive perfect overlap.
    """

    prediction = np.zeros(
        10,
        dtype=np.uint8,
    )

    target = np.zeros(
        10,
        dtype=np.uint8,
    )

    assert (
        calculate_dice(
            prediction,
            target,
        )
        == 1.0
    )

    assert (
        calculate_iou(
            prediction,
            target,
        )
        == 1.0
    )


def test_save_segmentation_nifti(
    tmp_path,
):
    """
    Verify that a segmentation can be saved as NIfTI
    while preserving spatial metadata.
    """

    reference_path = (
        tmp_path
        / "reference.nii.gz"
    )

    output_path = (
        tmp_path
        / "prediction.nii.gz"
    )

    # --------------------------------------------------------
    # Create a small synthetic reference volume
    # --------------------------------------------------------

    reference_data = np.zeros(
        (
            20,
            20,
            20,
        ),
        dtype=np.float32,
    )

    affine = np.eye(
        4
    )

    reference_image = (
        nib.Nifti1Image(
            reference_data,
            affine,
        )
    )

    nib.save(
        reference_image,
        str(reference_path),
    )

    # --------------------------------------------------------
    # Create synthetic segmentation
    # --------------------------------------------------------

    segmentation = torch.zeros(
        (
            20,
            20,
            20,
        ),
        dtype=torch.uint8,
    )

    segmentation[
        5:10,
        5:10,
        5:10,
    ] = 1

    # --------------------------------------------------------
    # Save segmentation
    # --------------------------------------------------------

    saved_path = (
        save_segmentation_nifti(
            segmentation=segmentation,
            reference_image_path=reference_path,
            output_path=output_path,
        )
    )

    # --------------------------------------------------------
    # Verify file
    # --------------------------------------------------------

    assert saved_path.exists()

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    prediction_image = nib.load(
        str(saved_path)
    )

    prediction_data = (
        prediction_image.get_fdata()
    )

    # --------------------------------------------------------
    # Verify shape
    # --------------------------------------------------------

    assert prediction_image.shape == (
        20,
        20,
        20,
    )

    # --------------------------------------------------------
    # Verify datatype
    # --------------------------------------------------------

    assert (
        prediction_image.get_data_dtype()
        == np.dtype("uint8")
    )

    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    assert set(
        np.unique(
            prediction_data
        )
    ).issubset(
        {
            0.0,
            1.0,
            2.0,
            3.0,
        }
    )

    # --------------------------------------------------------
    # Verify affine
    # --------------------------------------------------------

    assert np.allclose(
        prediction_image.affine,
        affine,
    )