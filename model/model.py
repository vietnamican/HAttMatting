import torch
import torch.nn as nn
from torchsummary import summary
from .features_extractor import FeatureExtractor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features_extractor = FeatureExtractor()
    
    def forward(self, x):
        return self.features_extractor(x)