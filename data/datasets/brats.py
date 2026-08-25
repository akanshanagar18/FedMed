"""
Module: data.datasets.brats

Purpose:
Production MONAI BraTS (Brain Tumor Segmentation) Dataset Loader for BraTS 2021 / 2023.
Auto-discovers subjects and 4-channel MRI NIfTI modalities (T1, T1ce, T2, FLAIR) + SEG mask.
Includes fallback synthetic NIfTI file generation when dataset files are absent.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset as PyTorchDataset, DataLoader


from data.datasets.cache import build_monai_dataset
from data.datasets.metadata import PatientMetadataIndexer
from data.datasets.transforms import get_brats_transforms
from data.datasets.validators import DatasetValidator

logger = logging.getLogger(__name__)


def generate_synthetic_brats_nifti(
    target_dir: Union[str, Path],
    num_subjects: int = 4,
    spatial_shape: Tuple[int, int, int] = (32, 32, 32),
) -> Path:
    """
    Generates synthetic NIfTI files matching BraTS structure for local execution and unit tests.
    """
    import nibabel as nib

    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modalities = ["t1", "t1ce", "t2", "flair", "seg"]

    for i in range(1, num_subjects + 1):
        subj_name = f"BraTS2021_{i:05d}"
        subj_dir = out_dir / subj_name
        subj_dir.mkdir(exist_ok=True)

        for mod in modalities:
            file_path = subj_dir / f"{subj_name}_{mod}.nii.gz"
            if not file_path.exists():
                if mod == "seg":
                    # Generate realistic BraTS 2021/2023 label values: 0 (bg), 1 (TC), 2 (ED), 3 (ET)
                    data = np.random.choice([0, 1, 2, 3], size=spatial_shape, p=[0.7, 0.1, 0.1, 0.1]).astype(np.int16)
                else:
                    data = np.random.randn(*spatial_shape).astype(np.float32)
                affine = np.eye(4)
                img = nib.Nifti1Image(data, affine)
                nib.save(img, str(file_path))


    return out_dir


class BraTSDataset:
    """
    Production BraTS Dataset Manager & Loader.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/BraTS2021",
        modalities: Optional[List[str]] = None,
        image_size: Tuple[int, int, int] = (128, 128, 128),
        cache_type: str = "persistent",
        cache_dir: Optional[Union[str, Path]] = ".cache/monai",
        num_workers: int = 0,
        val_split: float = 0.2,
        seed: int = 42,
        allow_synthetic_fallback: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.modalities = modalities or ["t1", "t1ce", "t2", "flair"]
        self.image_size = image_size
        self.cache_type = cache_type
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        # Validate dataset directory
        report = DatasetValidator.validate_dataset_directory(self.data_dir, self.modalities)

        if not report.is_valid and allow_synthetic_fallback:
            logger.info(f"Dataset directory '{self.data_dir}' invalid or empty. Generating synthetic BraTS NIfTI files for demonstration...")
            generate_synthetic_brats_nifti(self.data_dir, num_subjects=4)
            report = DatasetValidator.validate_dataset_directory(self.data_dir, self.modalities)

        self.validation_report = report
        self.patient_metadata = PatientMetadataIndexer.index_directory(self.data_dir, self.modalities)

        # Build MONAI dict entries: {"image": [t1_path, t1ce_path, t2_path, flair_path], "label": seg_path}
        self.data_list: List[Dict[str, Any]] = []
        for meta in self.patient_metadata:
            mod_files = [meta.modality_paths[m] for m in self.modalities if m in meta.modality_paths]
            if len(mod_files) == len(self.modalities) and meta.mask_path:
                self.data_list.append({
                    "image": mod_files,
                    "label": meta.mask_path,
                    "patient_id": meta.patient_id,
                })

        # Split train / val
        np.random.seed(self.seed)
        indices = np.arange(len(self.data_list))
        np.random.shuffle(indices)
        val_size = int(len(self.data_list) * self.val_split)

        self.val_indices = indices[:val_size] if val_size > 0 else indices[:1]
        self.train_indices = indices[val_size:] if val_size > 0 else indices

        self.train_files = [self.data_list[i] for i in self.train_indices]
        self.val_files = [self.data_list[i] for i in self.val_indices]

    def get_train_dataset(self) -> PyTorchDataset:
        """Returns MONAI training dataset with augmentations and caching."""
        transforms = get_brats_transforms(mode="train", image_size=self.image_size)
        return build_monai_dataset(
            data_list=self.train_files,
            transforms=transforms,
            cache_type=self.cache_type,
            cache_dir=self.cache_dir,
            num_workers=self.num_workers,
        )

    def get_val_dataset(self) -> PyTorchDataset:
        """Returns MONAI validation dataset with spatial normalization."""
        transforms = get_brats_transforms(mode="val", image_size=self.image_size)
        return build_monai_dataset(
            data_list=self.val_files,
            transforms=transforms,
            cache_type=self.cache_type,
            cache_dir=self.cache_dir,
            num_workers=self.num_workers,
        )

    def get_dataloader(self, split: str = "train", batch_size: int = 2) -> DataLoader:
        """Returns PyTorch DataLoader for requested split."""
        ds = self.get_train_dataset() if split == "train" else self.get_val_dataset()
        return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))

