import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

class ConvBatchNormRelu(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(ConvBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        outplanes = args[1]
        self.batchnorm = nn.BatchNorm2d(int(outplanes))
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))