"""
FedMed - MONAI Medical Image Transforms

Defines preprocessing and augmentation pipelines for
the MSD Task01 Brain Tumour dataset.
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)


def get_train_transforms(patch_size=(96, 96, 96)):
    """
    Create the training preprocessing pipeline.

    Args:
        patch_size: 3D spatial size used for training patches.

    Returns:
        MONAI Compose transform.
    """

    return Compose(
        [
            # Load NIfTI files
            LoadImaged(
                keys=["image", "label"]
            ),

            # Image has shape:
            # H x W x D x 4
            #
            # Convert to:
            # 4 x H x W x D
            EnsureChannelFirstd(
                keys=["image"],
                channel_dim=-1,
            ),

            # Label has no channel dimension.
            EnsureChannelFirstd(
                keys=["label"],
                channel_dim="no_channel",
            ),

            # Standardize orientation
            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
            ),

            # Normalize each MRI modality
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True,
            ),

            # Sample a patch containing positive
            # or negative regions.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),

            # Random left/right type augmentation
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),

            # Random 90-degree rotation
            RandRotate90d(
                keys=["image", "label"],
                prob=0.5,
                max_k=3,
            ),

            # Convert to PyTorch-compatible tensors
            EnsureTyped(
                keys=["image", "label"],
            ),
        ]
    )


def get_validation_transforms(patch_size=(96, 96, 96)):
    """
    Create validation preprocessing pipeline.

    Validation should avoid random augmentation.
    """

    return Compose(
        [
            LoadImaged(
                keys=["image", "label"]
            ),

            EnsureChannelFirstd(
                keys=["image"],
                channel_dim=-1,
            ),

            EnsureChannelFirstd(
                keys=["label"],
                channel_dim="no_channel",
            ),

            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
            ),

            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True,
            ),

            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),

            EnsureTyped(
                keys=["image", "label"],
            ),
        ]
    )