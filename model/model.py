from .base import Base
from .feature_extraction import FeatureExtraction
from .aspp import ASPP
from .hierarchical_attention import HierarchicalAttention
from .discriminator import Discriminator


class Model(Base):
    def __init__(self):
        super(Model, self).__init__()
        self.feature_extraction = FeatureExtraction()
        self.aspp = ASPP()
        self.hierarchical_attention = HierarchicalAttention()
        self.discriminator = Discriminator()
        self.upsample = lambda x: x

    def forward(
        self,
        x,
        alpha_matte_true
    ):
        low_level, high_level = self.feature_extraction(x)
        high_level = self.aspp(high_level)

        alpha_matte_pred = self.hierarchical_attention(low_level, high_level)

        self.upsample(alpha_matte_pred)

        out = self.discriminator(x, alpha_matte_true, alpha_matte_pred)

        return out
