"""
FedMed - Brain Tumour Segmentation Visualization

Creates visualizations of MRI slices and their predicted
segmentation masks.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def get_middle_slice(
    image,
    axis=2,
):
    """
    Get the middle slice of a 3D volume.

    Parameters
    ----------
    image : numpy.ndarray
        3D image volume.

    axis : int
        Axis along which the slice is selected.

    Returns
    -------
    numpy.ndarray
        2D image slice.
    """

    if image.ndim != 3:
        raise ValueError(
            "Expected a 3D volume. "
            f"Received shape: {image.shape}"
        )

    slice_index = (
        image.shape[axis] // 2
    )

    if axis == 0:
        return image[slice_index, :, :]

    if axis == 1:
        return image[:, slice_index, :]

    if axis == 2:
        return image[:, :, slice_index]

    raise ValueError(
        "Axis must be 0, 1, or 2."
    )


def load_nifti_data(
    image_path,
):
    """
    Load image data from a NIfTI file.

    Parameters
    ----------
    image_path : str or Path
        NIfTI file path.

    Returns
    -------
    numpy.ndarray
        NIfTI image data.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {image_path}"
        )

    image = nib.load(
        str(image_path)
    )

    return image.get_fdata()


def create_segmentation_visualization(
    image_path,
    prediction_path,
    output_path,
    slice_axis=2,
):
    """
    Create a visualization containing an MRI slice
    and the corresponding predicted segmentation.

    Parameters
    ----------
    image_path : str or Path
        Original MRI NIfTI file.

    prediction_path : str or Path
        Predicted segmentation NIfTI file.

    output_path : str or Path
        Destination PNG file.

    slice_axis : int
        Axis used to select the representative slice.

    Returns
    -------
    Path
        Saved visualization path.
    """

    image_path = Path(
        image_path
    )

    prediction_path = Path(
        prediction_path
    )

    output_path = Path(
        output_path
    )

    # --------------------------------------------------------
    # Load MRI
    # --------------------------------------------------------

    mri = load_nifti_data(
        image_path
    )

    # --------------------------------------------------------
    # Load segmentation
    # --------------------------------------------------------

    segmentation = load_nifti_data(
        prediction_path
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if mri.ndim != 4:
        raise ValueError(
            "Expected a 4D MRI volume with four modalities. "
            f"Received shape: {mri.shape}"
        )

    if segmentation.ndim != 3:
        raise ValueError(
            "Expected a 3D segmentation volume. "
            f"Received shape: {segmentation.shape}"
        )

    if tuple(mri.shape[:3]) != tuple(
        segmentation.shape
    ):
        raise ValueError(
            "MRI and segmentation spatial dimensions "
            "do not match. "
            f"MRI: {mri.shape[:3]}, "
            f"Segmentation: {segmentation.shape}"
        )

    # --------------------------------------------------------
    # Use first MRI modality for visualization
    # --------------------------------------------------------

    mri_volume = mri[..., 0]

    # --------------------------------------------------------
    # Select corresponding slices
    # --------------------------------------------------------

    mri_slice = get_middle_slice(
        mri_volume,
        axis=slice_axis,
    )

    segmentation_slice = get_middle_slice(
        segmentation,
        axis=slice_axis,
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    # --------------------------------------------------------
    # MRI
    # --------------------------------------------------------

    axes[0].imshow(
        np.rot90(mri_slice),
        cmap="gray",
    )

    axes[0].set_title(
        "MRI"
    )

    axes[0].axis(
        "off"
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    axes[1].imshow(
        np.rot90(segmentation_slice),
        interpolation="nearest",
    )

    axes[1].set_title(
        "Predicted Segmentation"
    )

    axes[1].axis(
        "off"
    )

    figure.suptitle(
        "FedMed Brain Tumour Segmentation"
    )

    figure.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path