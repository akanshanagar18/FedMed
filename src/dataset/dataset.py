"""
BraTS Dataset Loader
"""

from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """
    Dataset class for BraTS MRI images.
    """

    def __init__(self, image_paths=None, mask_paths=None, transforms=None):
        self.image_paths = image_paths or []
        self.mask_paths = mask_paths or []
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Returns one MRI scan and its mask.
        Actual loading logic will be added later.
        """

        image = None
        mask = None

        return image, mask