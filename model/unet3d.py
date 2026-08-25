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

    def __init__(self, in_channels: int = IN_CHANNELS, out_channels: int = OUT_CHANNELS):

        super().__init__()

        self.model = UNet(

            spatial_dims=3,

            in_channels=in_channels,

            out_channels=out_channels,

            channels=CHANNELS,

            strides=STRIDES,

            num_res_units=NUM_RES_UNITS,

        )


    def forward(self, x):

        return self.model(x)