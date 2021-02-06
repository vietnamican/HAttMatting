import torch
from torch import nn

from .base import Base, BaseSequential, ConvBatchNormRelu


class Discriminator(Base):
    def __init__(self):
        super().__init__()
        base_depth = 16
        kernel_size = 3
        padding = 1
        conv1 = nn.Sequential(
            ConvBatchNormRelu(
                4, base_depth, kernel_size=kernel_size, padding=padding),
            ConvBatchNormRelu(base_depth, base_depth,
                              kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2)
        )
        conv2 = nn.Sequential(
            ConvBatchNormRelu(base_depth, base_depth*2,
                              kernel_size=kernel_size, padding=padding),
            ConvBatchNormRelu(base_depth*2, base_depth*2,
                              kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2)
        )
        conv3 = nn.Sequential(
            ConvBatchNormRelu(base_depth*2, base_depth*4,
                              kernel_size=kernel_size, padding=padding),
            ConvBatchNormRelu(base_depth*4, base_depth*4,
                              kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2)
        )
        conv4 = nn.Sequential(
            ConvBatchNormRelu(base_depth*4, base_depth*8,
                              kernel_size=kernel_size, padding=padding),
            ConvBatchNormRelu(base_depth*8, base_depth*8,
                              kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2)
        )
        conv5 = nn.Sequential(
            ConvBatchNormRelu(base_depth*8, base_depth*16,
                              kernel_size=kernel_size, padding=padding),
            ConvBatchNormRelu(base_depth*16, base_depth *
                              16, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2)
        )
        last_conv = nn.Conv2d(base_depth*16, 1, kernel_size=1)
        self.sequential = nn.Sequential(
            conv1, conv2, conv3, conv4, conv5, last_conv)

    def forward(
        self,
        x,
        alpha_matte,
    ):
        return self.sequential(torch.cat((x, alpha_matte), dim=1))
