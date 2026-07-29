"""
Image preprocessing transforms
"""

from monai.transforms import Compose


def get_train_transforms():
    """
    Returns preprocessing pipeline for training.
    """

    transforms = Compose([])

    return transforms


def get_validation_transforms():
    """
    Returns preprocessing pipeline for validation.
    """

    transforms = Compose([])

    return transforms