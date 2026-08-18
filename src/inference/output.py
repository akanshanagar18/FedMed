"""
FedMed - NIfTI Prediction Output

Handles prediction output paths and saves segmentation
predictions as NIfTI files while preserving the spatial
metadata of the original MRI.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from configs.inference_config import (
    PREDICTION_DIR,
    PREDICTION_SUFFIX,
)


def get_case_name(
    image_path,
):
    """
    Extract the case name from an MRI filename.
    """

    image_path = Path(
        image_path
    )

    filename = image_path.name

    if filename.endswith(
        ".nii.gz"
    ):
        return filename[:-7]

    if filename.endswith(
        ".nii"
    ):
        return filename[:-4]

    return image_path.stem


def get_prediction_path(
    image_path,
    output_dir=PREDICTION_DIR,
):
    """
    Create the standard prediction output path.
    """

    image_path = Path(
        image_path
    )

    output_dir = Path(
        output_dir
    )

    case_name = get_case_name(
        image_path
    )

    return (
        output_dir
        / f"{case_name}{PREDICTION_SUFFIX}"
    )


def save_segmentation_nifti(
    segmentation,
    reference_image_path,
    output_path,
):
    """
    Save a segmentation mask as a NIfTI file.

    The affine transformation and spatial metadata from
    the original MRI are preserved.
    """

    reference_image_path = Path(
        reference_image_path
    )

    output_path = Path(
        output_path
    )

    if not reference_image_path.exists():
        raise FileNotFoundError(
            f"Reference MRI not found: "
            f"{reference_image_path}"
        )

    reference_image = nib.load(
        str(reference_image_path)
    )

    if isinstance(
        segmentation,
        torch.Tensor,
    ):
        segmentation = (
            segmentation
            .detach()
            .cpu()
            .numpy()
        )

    segmentation = np.asarray(
        segmentation
    )

    if segmentation.ndim != 3:
        raise ValueError(
            "Segmentation must be a 3D volume. "
            f"Received shape: {segmentation.shape}"
        )

    reference_shape = (
        reference_image.shape[:3]
    )

    if tuple(segmentation.shape) != tuple(
        reference_shape
    ):
        raise ValueError(
            "Segmentation spatial dimensions do not "
            "match the reference MRI. "
            f"Segmentation: {segmentation.shape}, "
            f"Reference: {reference_shape}"
        )

    segmentation = segmentation.astype(
        np.uint8
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_header = (
        reference_image.header.copy()
    )

    output_header.set_data_dtype(
        np.uint8
    )

    prediction_image = nib.Nifti1Image(
        segmentation,
        affine=reference_image.affine,
        header=output_header,
    )

    nib.save(
        prediction_image,
        str(output_path),
    )

    return output_path