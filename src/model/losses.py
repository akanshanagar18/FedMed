"""
FedMed - Segmentation Loss Functions

Loss functions used for multi-class brain tumour
segmentation.
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss


class BrainTumourLoss(nn.Module):
    """
    Combined Dice + Cross Entropy loss.

    Dice loss helps measure overlap between the predicted
    tumour regions and the ground-truth segmentation.

    Cross Entropy helps the model learn the individual
    segmentation classes.
    """

    def __init__(self):

        super().__init__()

        self.loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
        )

    def forward(self, prediction, target):

        return self.loss(
            prediction,
            target
        )