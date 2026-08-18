"""
FedMed - MRI Input Validation

Validates medical image files before inference.
"""

from pathlib import Path

import nibabel as nib


def validate_mri_file(
    image_path,
):
    """
    Validate a BraTS MRI NIfTI file.

    Parameters
    ----------
    image_path : str or Path
        Path to the MRI file.

    Returns
    -------
    dict
        Information about the validated image.

    Raises
    ------
    FileNotFoundError
        If the image does not exist.

    ValueError
        If the image format or dimensions are invalid.
    """

    image_path = Path(image_path)

    # --------------------------------------------------------
    # Check that the file exists
    # --------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(
            f"MRI file not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Path is not a file: {image_path}"
        )

    # --------------------------------------------------------
    # Check file extension
    # --------------------------------------------------------

    valid_extension = (
        image_path.name.endswith(".nii")
        or image_path.name.endswith(".nii.gz")
    )

    if not valid_extension:
        raise ValueError(
            "MRI file must have a .nii or .nii.gz extension."
        )

    # --------------------------------------------------------
    # Try loading the NIfTI image
    # --------------------------------------------------------

    try:
        image = nib.load(
            str(image_path)
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to load NIfTI image: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Check image dimensions
    # --------------------------------------------------------

    shape = image.shape

    if len(shape) != 4:
        raise ValueError(
            "Expected a 4-dimensional MRI volume "
            "with four modalities. "
            f"Received shape: {shape}"
        )

    # --------------------------------------------------------
    # Check number of MRI modalities
    # --------------------------------------------------------

    num_modalities = shape[-1]

    if num_modalities != 4:
        raise ValueError(
            "Expected 4 MRI modalities. "
            f"Received {num_modalities}."
        )

    # --------------------------------------------------------
    # Return useful information
    # --------------------------------------------------------

    result = {
        "path": str(image_path),
        "shape": tuple(shape),
        "num_modalities": num_modalities,
        "valid": True,
    }

    return result