from .base import Base
from .utils import build_backbone

class FeatureExtractor(Base):
    def __init__(self):
        super().__init__()
        self.backbone = build_backbone()

    def forward(self, x):
        x = self.backbone(x)
        return list(x.values())