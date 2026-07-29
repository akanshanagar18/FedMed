"""
Post-processing utilities
"""

import torch


def apply_threshold(prediction, threshold=0.5):

    """
    Convert probability map into binary mask.
    """

    return (prediction > threshold).float()


def get_prediction_mask(prediction):

    """
    Return predicted segmentation mask.
    """

    return torch.argmax(prediction, dim=1)