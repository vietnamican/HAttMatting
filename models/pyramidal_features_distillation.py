from torch import nn as nn

from .base import Base
from .config import config


class PyramidalFeaturesDistillation(Base):
    def __init__(self):
        super().__init__()
        r = config['r']
        w = config['base_width']
        self.attention_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(w, w//r),
            nn.ReLU(inplace=True),
            nn.Linear(w//r, w),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weight = self.attention_module(x)
