import torchvision
from torch import nn

from .base import Base


class FeatureExtraction(Base):
    def __init__(self):
        super().__init__()
        module = torchvision.models.resnet.resnext50_32x4d(pretrained=True)
        self.conv1 = module.conv1 # divide by 2
        self.bn1 = module.bn1
        self.relu = module.relu
        self.maxpool = module.maxpool
        self.layer1 = module.layer1 # divide by 4
        self.layer2 = module.layer2 # divide by 8
        self.layer3 = module.layer3 # divide by 16
        self.layer4 = module.layer4 # divide by 32

    def forward(
        self,
        x
    ):
        low_level = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        high_level = self.layer3(self.layer2(self.layer1(low_level)))
        return low_level, high_level
