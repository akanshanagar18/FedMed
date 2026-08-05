"""
3D U-Net Model
"""

import torch.nn as nn
from monai.networks.nets import UNet

from configs.config import (
    IN_CHANNELS,
    OUT_CHANNELS,
    CHANNELS,
    STRIDES,
    NUM_RES_UNITS,
)


class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = UNet(

            spatial_dims=3,

            in_channels=IN_CHANNELS,

            out_channels=OUT_CHANNELS,

            channels=CHANNELS,

            strides=STRIDES,

            num_res_units=NUM_RES_UNITS,

        )

    def forward(self, x):

        return self.model(x)