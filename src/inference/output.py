"""
FedMed - NIfTI Prediction Output

Saves segmentation predictions as NIfTI files while
preserving the spatial metadata of the original MRI.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def save_segmentation_nifti(
    segmentation,
    reference_image_path,
    output_path,
):
    """
    Save a segmentation mask as a NIfTI file.

    The affine transformation and spatial metadata from the
    original MRI are preserved.

    Parameters
    ----------
    segmentation : torch.Tensor or numpy.ndarray
        3D segmentation mask with shape [H, W, D].

    reference_image_path : str or Path
        Path to the original MRI NIfTI file.

    output_path : str or Path
        Destination path for the prediction NIfTI file.

    Returns
    -------
    Path
        Path to the saved prediction.
    """

    reference_image_path = Path(
        reference_image_path
    )

    output_path = Path(
        output_path
    )

    # --------------------------------------------------------
    # Check reference image
    # --------------------------------------------------------

    if not reference_image_path.exists():
        raise FileNotFoundError(
            f"Reference MRI not found: "
            f"{reference_image_path}"
        )

    # --------------------------------------------------------
    # Load original MRI
    # --------------------------------------------------------

    reference_image = nib.load(
        str(reference_image_path)
    )

    # --------------------------------------------------------
    # Convert segmentation to NumPy
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validate segmentation dimensions
    # --------------------------------------------------------

    if segmentation.ndim != 3:
        raise ValueError(
            "Segmentation must be a 3D volume. "
            f"Received shape: {segmentation.shape}"
        )

    # --------------------------------------------------------
    # Validate spatial dimensions
    # --------------------------------------------------------

    reference_shape = reference_image.shape[:3]

    if tuple(segmentation.shape) != tuple(
        reference_shape
    ):
        raise ValueError(
            "Segmentation spatial dimensions do not "
            "match the reference MRI. "
            f"Segmentation: {segmentation.shape}, "
            f"Reference: {reference_shape}"
        )

    # --------------------------------------------------------
    # Convert segmentation labels to uint8
    #
    # Expected labels:
    # 0 = background
    # 1 = class 1
    # 2 = class 2
    # 3 = class 3
    # --------------------------------------------------------

    segmentation = segmentation.astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Copy original NIfTI header
    # --------------------------------------------------------

    output_header = (
        reference_image.header.copy()
    )

    # Explicitly set the output datatype to uint8.
    #
    # This is important because the prediction is a
    # discrete segmentation mask rather than continuous
    # MRI intensity data.
    output_header.set_data_dtype(
        np.uint8
    )

    # --------------------------------------------------------
    # Create prediction NIfTI
    #
    # The original affine is preserved so that the
    # segmentation remains spatially aligned with
    # the source MRI.
    # --------------------------------------------------------

    prediction_image = nib.Nifti1Image(
        segmentation,
        affine=reference_image.affine,
        header=output_header,
    )

    # --------------------------------------------------------
    # Save prediction
    # --------------------------------------------------------

    nib.save(
        prediction_image,
        str(output_path),
    )

    return output_path