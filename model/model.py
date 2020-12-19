import torch
import torch.nn as nn
from torchsummary import summary
from .features_extractor import FeatureExtractor
from .aspp import ASPP
from .pyramidal_features_distillation import PyramidalFeaturesDistillation

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features_extractor = FeatureExtractor()
        self.aspp = ASPP(512, 16, nn.BatchNorm2d)
        self.pyramidal_features_distillation = PyramidalFeaturesDistillation()
    
    def forward(self, x):
        low_level_feature, high_level_feature = self.features_extractor(x)
        x = self.aspp(high_level_feature)
        x = self.pyramidal_features_distillation(x)
        return x

if __name__ == '__main__':
    model = Model()
    summary(model, (3, 320, 320))