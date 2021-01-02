import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class ConvBatchnormRelu(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(ConvBatchnormRelu, self).__init__()
        conv_mod = nn.Conv2d(*args, **kwargs)
        outplanes = args[1]
        self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(outplanes)), nn.ReLU(inplace=True))
        # self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
        # with_bn = kwargs['with_bn'] if 'with_bn' in kwargs else False
        # with_relu = kwargs['with_relu'] if 'with_relu' in kwargs else False
        # if with_bn:
        #     if with_relu:
        #         self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(outplanes)), nn.ReLU(inplace=True))
        #     else:
        #         self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(outplanes)))
        # else:
        #     if with_relu:
        #         self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
        #     else:
        #         self.cbr_unit = nn.Sequential(conv_mod)
    def forward(self, x):
        return self.cbr_unit(x)