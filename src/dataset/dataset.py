"""
BraTS Dataset Loader
"""

from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """
    Dataset class for BraTS MRI images.
    """

    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        sample = {
            "image": image_path,
            "mask": mask_path
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample