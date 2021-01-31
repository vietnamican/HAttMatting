from .base import Base


class FeatureExtraction(Base):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x
    ):
        return x, x
