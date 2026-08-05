"""
MONAI Transform Pipeline
------------------------
This module defines preprocessing and augmentation
for the BraTS MRI dataset.
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    ToTensord,
)


def get_train_transforms():
    """
    Preprocessing + augmentation for training.
    """

    train_transforms = Compose([

        # Load MRI and mask
        LoadImaged(keys=["image", "mask"]),

        # Ensure channel dimension exists
        EnsureChannelFirstd(keys=["image", "mask"]),

        # Normalize MRI intensities
        NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=True
        ),

        # Random crop
        RandSpatialCropd(
            keys=["image", "mask"],
            roi_size=(128, 128, 128),
            random_size=False
        ),

        # Random flip
        RandFlipd(
            keys=["image", "mask"],
            prob=0.5,
            spatial_axis=0
        ),

        # Random rotation
        RandRotate90d(
            keys=["image", "mask"],
            prob=0.5,
            max_k=3
        ),

        # Convert to tensors
        ToTensord(keys=["image", "mask"])

    ])

    return train_transforms


def get_validation_transforms():
    """
    Validation preprocessing.
    """

    val_transforms = Compose([

        LoadImaged(keys=["image", "mask"]),

        EnsureChannelFirstd(keys=["image", "mask"]),

        NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=True
        ),

        ToTensord(keys=["image", "mask"])

    ])

    return val_transforms