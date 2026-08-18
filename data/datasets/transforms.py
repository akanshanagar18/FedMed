"""
Module: data.datasets.transforms

Purpose:
Production MONAI medical image preprocessing and augmentation transform sequences.
Supports training, validation, and inference modes with strict spatial & intensity normalization
and unified canonical label conversion across BraTS 2021 and BraTS-GLI 2024.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    Spacingd,
)

from data.canonical_adapter import DatasetVersion, LabelCanonicalizer


class ConvertToCanonicalBratsClassesd(MapTransform):
    """
    Converts integer ground-truth segmentation masks into canonical 3-channel composite targets (TC, WT, ET)
    supporting both BraTS 2021 and BraTS-GLI 2024 schemas.
    """

    def __init__(
        self,
        keys: Union[str, List[str]],
        version: DatasetVersion = DatasetVersion.BRATS_2021_REAL,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.version = version

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            seg = d[key]
            if isinstance(seg, torch.Tensor):
                seg_np = seg.cpu().numpy()
            else:
                seg_np = np.asarray(seg)

            if seg_np.ndim == 4 and seg_np.shape[0] == 1:
                seg_np = seg_np.squeeze(0)

            targets, _ = LabelCanonicalizer.build_canonical_targets(seg_np, version=self.version)

            if isinstance(d[key], torch.Tensor):
                d[key] = torch.from_numpy(targets).to(d[key].device)
            else:
                d[key] = targets
        return d


def get_brats_transforms(
    mode: str = "train",
    image_size: Union[Tuple[int, int, int], List[int]] = (128, 128, 128),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    keys: Tuple[str, str] = ("image", "label"),
    dataset_version: DatasetVersion = DatasetVersion.BRATS_2021_REAL,
) -> Compose:
    """
    Returns MONAI Compose transform pipeline for BraTS 3D MRI segmentation.
    
    Args:
        mode: "train", "val", or "infer"
        image_size: Spatial target size (D, H, W)
        pixdim: Resampling voxel dimensions
        keys: Dictionary keys for image and mask tensors
        dataset_version: BraTS dataset version (BraTS 2021 vs BraTS-GLI 2024)
    """
    img_key, label_key = keys
    roi_size = tuple(image_size)

    if mode == "train":
        return Compose([
            LoadImaged(keys=[img_key, label_key], allow_missing_keys=True),
            EnsureChannelFirstd(keys=[img_key, label_key], allow_missing_keys=True),
            Orientationd(keys=[img_key, label_key], axcodes="RAS", allow_missing_keys=True),
            Spacingd(
                keys=[img_key, label_key],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            NormalizeIntensityd(keys=img_key, nonzero=True, channel_wise=True),
            CropForegroundd(keys=[img_key, label_key], source_key=img_key, allow_missing_keys=True),
            Resized(keys=[img_key, label_key], spatial_size=roi_size, mode=("bilinear", "nearest"), allow_missing_keys=True),
            ConvertToCanonicalBratsClassesd(keys=label_key, version=dataset_version, allow_missing_keys=True),
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
            Orientationd(keys=[img_key, label_key], axcodes="RAS", allow_missing_keys=True),
            Spacingd(
                keys=[img_key, label_key],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            NormalizeIntensityd(keys=img_key, nonzero=True, channel_wise=True),
            CropForegroundd(keys=[img_key, label_key], source_key=img_key, allow_missing_keys=True),
            Resized(keys=[img_key, label_key], spatial_size=roi_size, mode=("bilinear", "nearest"), allow_missing_keys=True),
            ConvertToCanonicalBratsClassesd(keys=label_key, version=dataset_version, allow_missing_keys=True),
            EnsureTyped(keys=[img_key, label_key], data_type="tensor", allow_missing_keys=True),
        ])
