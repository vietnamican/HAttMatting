import torch
from torch.nn import functional as F

from .base import Base
from .feature_extractor import FeatureExtractor as FE
from .aspp import ASPP
from .appearance_cues_filtration import AppearanceCuesFiltration as ACF
from .pyramidal_features_distillation import PyramidalFeaturesDistillation as PFD
from .config import config

class Model(Base):
    def __init__(self):
        super().__init__()
        w = config['base_width']
        self.fe = FE()
        self.aspp = ASPP(in_channels=w*8, atrous_rates=[6, 12, 18], out_channels=w)
        self.acf = ACF()
        self.pfd = PFD()

    def forward(self, x):
        l_lv, h_lv = self.fe(x)
        h_lv = self.aspp(h_lv)
        h_lv = F.upsample(h_lv, scale_factor=4, mode='bilinear')
        logit = self.acf(l_lv, h_lv)
        logit = F.upsample(logit, scale_factor=2, mode='bilinear')
        return logit

