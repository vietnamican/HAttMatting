from collections import OrderedDict

from torch import nn as nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet34, BasicBlock
from torchvision.models._utils import IntermediateLayerGetter

from .config import config


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def make_block(inplanes, planes, stride=1):
    if not stride == 1:
        downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
        )
    else: 
        downsample = None
    return BasicBlock(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample)

def convbnrelu(inplanes, outplanes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding),
        nn.BatchNorm2d(outplanes),
        nn.ReLU(inplace=True)
    )

def build_backbone():
    w = config['base_width']
    model = nn.Sequential(OrderedDict([
        ('conv', convbnrelu(3, w)),
        ('layer1', make_block(inplanes=w, planes=w*2, stride=2)),
        ('layer2', make_block(inplanes=w*2, planes=w*4, stride=2)),
        ('layer3', make_block(inplanes=w*4, planes=w*8, stride=2)),
        ('layer4', make_block(inplanes=w*8, planes=w*8, stride=1)),
    ]))
    return IntermediateLayerGetter(model, {'layer1': 'layer1', 'layer4': 'layer4'})