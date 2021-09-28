import torch
from torch import nn as nn
from torch.nn import functional as F

from .base import Base
from .utils import convbnrelu
from .config import config


class AppearanceCuesFiltration(Base):
    def __init__(self):
        super().__init__()
        w = config['base_width']
        self.h_lv_conv3x3 = convbnrelu(w, w)
        self.h_lv_top_path = nn.Sequential(
            convbnrelu(w, w, (1, 7), stride=1, padding=(0, 3)),
            convbnrelu(w, w, (7, 1), stride=1, padding=(3, 0))
        )
        self.h_lv_bot_path = nn.Sequential(
            convbnrelu(w, w, (7, 1), stride=1, padding=(3, 0)),
            convbnrelu(w, w, (1, 7), stride=1, padding=(0, 3))
        )
        self.h_lv_aggregator = convbnrelu(w*2, 1, 1, stride=1, padding=0)
        self.conv3x3_1 = convbnrelu(w*2, w)
        self.conv3x3_2 = convbnrelu(w*2, 1)

    def forward(self, l_lv, h_lv):
        h_lv = self.h_lv_conv3x3(h_lv)
        h_lv_top_feat = self.h_lv_top_path(h_lv)
        h_lv_bot_feat = self.h_lv_bot_path(h_lv)
        h_lv_concat = torch.cat([h_lv_top_feat, h_lv_bot_feat], dim=1)
        h_lv_concat = self.h_lv_aggregator(h_lv_concat)
        spatial_attention_map = torch.sigmoid(h_lv_concat)
        feat = l_lv * spatial_attention_map
        feat = self.conv3x3_1(feat)
        feat = torch.cat([feat, h_lv], dim=1)
        return self.conv3x3_2(feat)


