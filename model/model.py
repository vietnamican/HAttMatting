from torch import nn

from .base import Base
from .feature_extraction import FeatureExtraction
from .aspp import ASPP
from .hierarchical_attention import HierarchicalAttention
from .discriminator import Discriminator


class Model(Base):
    def __init__(self):
        super().__init__()
        self.feature_extraction = FeatureExtraction()
        self.aspp = ASPP(1024, [12, 24, 36])
        self.hierarchical_attention = HierarchicalAttention()
        self.discriminator = Discriminator()

    def forward(
        self,
        x,
        alpha_matte_true
    ):
        low_level, high_level = self.feature_extraction(x)
        low_level_size = low_level.shape[-2:]
        high_level = self.aspp(high_level)
        high_level = nn.Upsample(low_level_size, mode='bilinear', align_corners=True)(high_level)

        alpha_matte_pred = self.hierarchical_attention(low_level, high_level)
        input_size = x.shape[-2:]
        alpha_matte_pred = nn.Upsample(input_size, mode='bilinear', align_corners=True)(alpha_matte_pred)
        out = self.discriminator(alpha_matte_pred)

        return alpha_matte_pred
