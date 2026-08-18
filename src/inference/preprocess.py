"""
FedMed - MRI Inference Preprocessing

Prepares a BraTS MRI volume for inference.
"""

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    NormalizeIntensity,
)


def get_inference_transform():
    """
    Create the preprocessing pipeline used
    for MRI inference.

    Returns
    -------
    Compose
        MONAI preprocessing pipeline.
    """

    transform = Compose(
        [
            # ------------------------------------------------
            # Load NIfTI image
            # ------------------------------------------------

            LoadImage(
                image_only=True
            ),

            # ------------------------------------------------
            # Arrange MRI modalities as channels
            # ------------------------------------------------

            EnsureChannelFirst(
                channel_dim=-1
            ),

            # ------------------------------------------------
            # Standardize orientation
            # ------------------------------------------------

            Orientation(
                axcodes="RAS"
            ),

            # ------------------------------------------------
            # Normalize MRI intensities
            # ------------------------------------------------

            NormalizeIntensity(
                nonzero=True,
                channel_wise=True,
            ),
        ]
    )

    return transform