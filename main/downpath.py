import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

from .conv_batchnorm_relu import ConvBatchNormRelu
from .cbam import CbamBlock


class DownPath(pl.LightningModule):
    def __init__(self, length, *args, **kwargs):
        super(DownPath, self).__init__()
        inplanes = int(args[0])
        outplanes = int(args[1])
        if length is None:
            length = 2
        if 'attention' not in kwargs:
            attention = False
        else:
            attention = kwargs['attention']
            kwargs.pop('attention', None)
        if length == 2:
            self.conv_path = nn.Sequential(
                ConvBatchNormRelu(inplanes, outplanes, *args[2:], **kwargs),
                ConvBatchNormRelu(outplanes, outplanes, *args[2:], **kwargs)
            )        
        elif length == 3:
            self.conv_path = nn.Sequential(
                ConvBatchNormRelu(inplanes, outplanes, *args[2:], **kwargs),
                ConvBatchNormRelu(outplanes, outplanes, *args[2:], **kwargs),
                ConvBatchNormRelu(outplanes, outplanes, *args[2:], **kwargs)
            )
        if attention:
            self.conv_path.add_module('attention', CbamBlock(outplanes))
        self.pool_path = nn.MaxPool2d(kernel_size=(
            2, 2), stride=(2, 2), return_indices=True)

    def forward(self, x):
        last_feature_map = self.conv_path(x)
        feature_map_pool, indices = self.pool_path(last_feature_map)
        return last_feature_map, last_feature_map.size(), feature_map_pool, indices
