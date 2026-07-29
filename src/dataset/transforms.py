"""
Image preprocessing transforms
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)


def get_train_transforms():

    return Compose([

        LoadImaged(keys=["image", "mask"]),

        EnsureChannelFirstd(keys=["image", "mask"]),

        NormalizeIntensityd(keys="image"),

        RandFlipd(
            keys=["image", "mask"],
            prob=0.5,
            spatial_axis=0
        ),

        RandRotate90d(
            keys=["image", "mask"],
            prob=0.5
        ),

        ToTensord(keys=["image", "mask"])

    ])


def get_validation_transforms():

    return Compose([

        LoadImaged(keys=["image", "mask"]),

        EnsureChannelFirstd(keys=["image", "mask"]),

        NormalizeIntensityd(keys="image"),

        ToTensord(keys=["image", "mask"])

    ])