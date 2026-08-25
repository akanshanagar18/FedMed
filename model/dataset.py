"""
BraTS Dataset Loader
--------------------
Loads MRI image paths and corresponding segmentation masks.
Compatible with PyTorch DataLoader.
"""

from pathlib import Path
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """
    Dataset class for BraTS MRI images.
    """

    def __init__(self, image_paths, mask_paths, transforms=None):
        """
        Args:
            image_paths (list): List of MRI image file paths.
            mask_paths (list): List of segmentation mask file paths.
            transforms (callable): MONAI transforms.
        """

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        """
        Returns total number of samples.
        """

        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Returns one training sample.
        """

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        sample = {
            "image": str(image_path),
            "mask": str(mask_path)
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample