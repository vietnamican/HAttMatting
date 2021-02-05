import torch
from torch import nn

from .base import Base, ConvBatchNormRelu


class Discriminator(Base):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBatchNormRelu(1, 8, kernel_size=3, padding=1),
            ConvBatchNormRelu(8, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            ConvBatchNormRelu(8, 16, kernel_size=3, padding=1),
            ConvBatchNormRelu(16, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            ConvBatchNormRelu(16, 32, kernel_size=3, padding=1),
            ConvBatchNormRelu(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            ConvBatchNormRelu(32, 64, kernel_size=3, padding=1),
            ConvBatchNormRelu(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            ConvBatchNormRelu(64, 128, kernel_size=3, padding=1),
            ConvBatchNormRelu(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.last_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(
        self,
        alpha_matte,
    ):
        return self.last_conv(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(alpha_matte))))))
