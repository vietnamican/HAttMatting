import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

from .conv_batchnorm_relu import ConvBatchNormRelu


class UpPath(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(UpPath, self).__init__()
        self.conv = ConvBatchNormRelu(*args, **kwargs)
        self.unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x, after_pool_feature, indices, output_size, return_conv_result=False):
        # print("--------------------------------")
        # print(x.shape)
        # print(after_pool_feature.shape)
        # print(indices.shape)
        # print(output_size)
        # print("--------------------------------")
        if return_conv_result:
            conv_result = torch.add(self.conv(x), after_pool_feature)
            return self.unpool(conv_result, indices, output_size=output_size), conv_result
        return self.unpool(torch.add(self.conv(x), after_pool_feature), indices, output_size=output_size)
