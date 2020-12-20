import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

from .conv_batchnorm_relu import ConvBatchnormRelu

class ApperanceCuesFiltration(nn.Module):
    def __init__(self):
        super(ApperanceCuesFiltration, self).__init__()
        self.conv1 = ConvBatchnormRelu(256, 256, 3, padding=1, padding_mode='zeros')
        self.way1_conv1 = ConvBatchnormRelu(256, 256, (7, 1), padding=(3, 0), padding_mode='zeros')
        self.way1_conv2 = ConvBatchnormRelu(256, 256, (1, 7), padding=(0, 3), padding_mode='zeros')
        self.way2_conv1 = ConvBatchnormRelu(256, 256, (7, 1), padding=(3, 0), padding_mode='zeros')
        self.way2_conv2 = ConvBatchnormRelu(256, 256, (1, 7), padding=(0, 3), padding_mode='zeros')
        self.conv1x1 = ConvBatchnormRelu(512, 256, 1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        way1 = self.way1_conv1(x)
        way1 = self.way1_conv2(way1)
        way2 = self.way2_conv1(x)
        way2 = self.way2_conv2(way2)
        x = torch.cat([way1, way2], 1)
        x = self.conv1x1(x)
        return x
