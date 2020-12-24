import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from torchsummary import summary

from .conv_batchnorm_relu import ConvBatchnormRelu
from .channel_attention import SpatialAttention

class ApperanceCuesFiltration(nn.Module):
    def __init__(self):
        super(ApperanceCuesFiltration, self).__init__()
        self.conv1 = ConvBatchnormRelu(256, 256, 3, padding=1, padding_mode='zeros')
        self.way1_conv1 = ConvBatchnormRelu(256, 256, (7, 1), padding=(3, 0), padding_mode='zeros')
        self.way1_conv2 = ConvBatchnormRelu(256, 256, (1, 7), padding=(0, 3), padding_mode='zeros')
        self.way2_conv1 = ConvBatchnormRelu(256, 256, (7, 1), padding=(3, 0), padding_mode='zeros')
        self.way2_conv2 = ConvBatchnormRelu(256, 256, (1, 7), padding=(0, 3), padding_mode='zeros')
        self.conv1x1 = ConvBatchnormRelu(512, 64, 1, padding=0)
        self.spatial_attention = SpatialAttention(64)
        self.conv3x3_1 = ConvBatchnormRelu(64, 64, 3, padding=1, padding_mode='zeros')
        self.conv3x3_2 = ConvBatchnormRelu(320, 256, 3, padding=1, padding_mode='zeros')
        self.conv3x3_3 = ConvBatchnormRelu(256, 256, 3, padding=1, padding_mode='zeros')
        self.conv3x3_4 = ConvBatchnormRelu(256, 64, 1, padding=0, padding_mode='zeros')
        self.conv3x3_5 = ConvBatchnormRelu(64, 64, 1, padding=0, padding_mode='zeros')
        self.conv3x3_5 = ConvBatchnormRelu(64, 1, 1, padding=0, padding_mode='zeros')
    def forward(self, x, low_level_feature):
        x = self.conv1(x)
        low_level_concat = x
        way1 = self.way1_conv1(x)
        way1 = self.way1_conv2(way1)
        way2 = self.way2_conv1(x)
        way2 = self.way2_conv2(way2)
        x = torch.cat([way1, way2], 1)
        x = self.conv1x1(x)
        x = torch.sigmoid(x)
        x, _ = self.spatial_attention(x)
        x = x * low_level_feature
        x = self.conv3x3_1(x)
        x = torch.cat([x, low_level_concat], dim=1)
        x = self.conv3x3_2(x)
        x = torch.sigmoid(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
        x = self.conv3x3_3(x)
        x = self.conv3x3_4(x)
        x = self.conv3x3_5(x)
        return x
