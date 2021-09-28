import torch
from torch import nn as nn

from .base import Base
from .utils import make_block, convbnrelu
from .config import config


class Discriminator(Base):
    def __init__(self):
        super().__init__()
        w = config['base_width']
        self.conv = nn.Sequential(
            make_block(1, w, stride=2),
            make_block(w, w, stride=2),
            make_block(w, w, stride=2)
        )
        self.aggregator = convbnrelu(w, 2, kernel_size=1, padding=0)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.aggregator(x)
        return self.avg(x)
