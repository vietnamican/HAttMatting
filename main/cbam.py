import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

import pytorch_lightning as pl

from. conv_batchnorm_relu import ConvBatchNormRelu


class MLP(pl.LightningModule):

    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio

        self.fc1 = nn.Linear(
            in_features=channels,
            out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            in_features=mid_channels,
            out_features=channels)

    def forward(self, x):
        return self.fc2(self.activ(self.fc1(x.view(x.size(0), -1))))


class ChannelGate(pl.LightningModule):

    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(ChannelGate, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mlp = MLP(
            channels=channels,
            reduction_ratio=reduction_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(
            torch.add(self.mlp(self.avg_pool(x)), self.mlp(self.max_pool(x))))
        return x * (att.unsqueeze(2).unsqueeze(3))


class SpatialGate(pl.LightningModule):

    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = ConvBatchNormRelu(2, 1, kernel_size=7,
                              stride=1, padding=3, bias=True, with_relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = torch.cat((x.max(dim=1)[0].unsqueeze(1), x.mean(dim=1).unsqueeze(1)), dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att


class CbamBlock(pl.LightningModule):

    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(CbamBlock, self).__init__()
        self.ch_gate = ChannelGate(
            channels=channels,
            reduction_ratio=reduction_ratio)
        self.sp_gate = SpatialGate()

    def forward(self, x):
        return self.sp_gate(self.ch_gate(x))
