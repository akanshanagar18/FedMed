"""
Module: data.datasets.transforms

Purpose:
Production MONAI medical image preprocessing and augmentation transform sequences.
Supports training, validation, and inference modes with strict spatial & intensity normalization.
"""

from typing import List, Tuple, Union
from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    SpatialPadd,
    Spacingd,
)



def get_brats_transforms(
    mode: str = "train",
    image_size: Union[Tuple[int, int, int], List[int]] = (128, 128, 128),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    keys: Tuple[str, str] = ("image", "label"),
) -> Compose:
    """
    Returns MONAI Compose transform pipeline for BraTS 3D MRI segmentation.
    
    Args:
        mode: "train", "val", or "infer"
        image_size: Spatial target size (D, H, W)
        pixdim: Resampling voxel dimensions
        keys: Dictionary keys for image and mask tensors
    """
    img_key, label_key = keys
    roi_size = tuple(image_size)

    if mode == "train":
        return Compose([
            LoadImaged(keys=[img_key, label_key], allow_missing_keys=True),
            EnsureChannelFirstd(keys=[img_key, label_key], allow_missing_keys=True),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=label_key, allow_missing_keys=True),
            Orientationd(keys=[img_key, label_key], axcodes="RAS", allow_missing_keys=True),
            Spacingd(
                keys=[img_key, label_key],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            NormalizeIntensityd(keys=img_key, nonzero=True, channel_wise=True),
            CropForegroundd(keys=[img_key, label_key], source_key=img_key, allow_missing_keys=True),
            SpatialPadd(keys=[img_key, label_key], spatial_size=roi_size, allow_missing_keys=True),
            RandSpatialCropd(
                keys=[img_key, label_key],
                roi_size=roi_size,
                random_size=False,
                allow_missing_keys=True,
            ),
            RandFlipd(keys=[img_key, label_key], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=[img_key, label_key], prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandFlipd(keys=[img_key, label_key], prob=0.5, spatial_axis=2, allow_missing_keys=True),
            RandRotate90d(keys=[img_key, label_key], prob=0.5, max_k=3, allow_missing_keys=True),
            RandScaleIntensityd(keys=img_key, factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=img_key, offsets=0.1, prob=0.5),
            EnsureTyped(keys=[img_key, label_key], data_type="tensor", allow_missing_keys=True),
        ])
    else:  # "val" or "infer"
        return Compose([
            LoadImaged(keys=[img_key, label_key], allow_missing_keys=True),
            EnsureChannelFirstd(keys=[img_key, label_key], allow_missing_keys=True),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=label_key, allow_missing_keys=True),
            Orientationd(keys=[img_key, label_key], axcodes="RAS", allow_missing_keys=True),
            Spacingd(
                keys=[img_key, label_key],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            NormalizeIntensityd(keys=img_key, nonzero=True, channel_wise=True),
            CropForegroundd(keys=[img_key, label_key], source_key=img_key, allow_missing_keys=True),
            SpatialPadd(keys=[img_key, label_key], spatial_size=roi_size, allow_missing_keys=True),
            EnsureTyped(keys=[img_key, label_key], data_type="tensor", allow_missing_keys=True),
        ])

