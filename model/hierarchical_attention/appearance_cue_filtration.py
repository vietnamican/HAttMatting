import torch
from torch import nn

from ..base import Base, ConvBatchNormRelu


class AppearanceCueFiltration(Base):
    def __init__(self):
        super().__init__()
        self.high_conv3x3 = ConvBatchNormRelu(
            256, 256, kernel_size=3, padding=1)
        self.path_1 = nn.Sequential(
            nn.Conv2d(256, 128, (1, 7), padding=(0, 3)),
            nn.Conv2d(128, 1, (7, 1), padding=(3, 0)),
        )
        self.path_2 = nn.Sequential(
            nn.Conv2d(256, 128, (7, 1), padding=(3, 0)),
            nn.Conv2d(128, 1, (1, 7), padding=(0, 3)),
        )
        self.high_conv1x1 = ConvBatchNormRelu(2, 1, kernel_size=1)
        self.conv3x3 = ConvBatchNormRelu(64, 256, kernel_size=3, padding=1)
        self.last_conv3x3 = ConvBatchNormRelu(512, 1, kernel_size=3, padding=1)

    def forward(
        self,
        low_level_feature,
        high_level_feature,
    ):
        high_level_feature = self.high_conv3x3(high_level_feature)

        spatial_feature = torch.cat(
            (self.path_1(high_level_feature), self.path_2(high_level_feature)), dim=1)
        spatial_feature = nn.Sigmoid()(self.high_conv1x1(spatial_feature))

        feature = low_level_feature * spatial_feature
        feature = self.conv3x3(feature)
        feature = torch.cat((feature, high_level_feature), dim=1)

        return self.last_conv3x3(feature)
