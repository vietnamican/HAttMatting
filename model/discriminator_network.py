import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

from .conv_batchnorm_relu import ConvBatchnormRelu

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.resize = nn.Upsample()
        self.conv1 = ConvBatchnormRelu(1, 64, (5, 5), padding=(2, 2), padding_mode='zeros')
        self.conv2_1 = ConvBatchnormRelu(64, 64, (3, 3), padding=(1, 1), padding_mode='zeros')
        self.conv2_2 = ConvBatchnormRelu(64, 128, (3, 3), padding=(1, 1), padding_mode='zeros')
        self.conv2_3 = ConvBatchnormRelu(128, 128, (3,3), padding=(1, 1), padding_mode='zeros')
        self.conv3_1 = ConvBatchnormRelu(128, 256, (3,3), padding=(1, 1), padding_mode='zeros')
        self.conv3_2 = ConvBatchnormRelu(256, 256, (3,3), padding=(1, 1), padding_mode='zeros')
        self.conv4_1 = ConvBatchnormRelu(256, 512, (3,3), padding=(1, 1), padding_mode='zeros')
        self.conv4_2 = ConvBatchnormRelu(512, 512, (3,3), padding=(1, 1), padding_mode='zeros')
        self.upsample  = nn.Upsample(size=(20, 20), mode='bicubic')

    def forward(self, x):
        x = self.conv1(x)
        x = nn.MaxPool2d(2, 2, return_indices=False)(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = nn.MaxPool2d(2, 2, return_indices=False)(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = nn.MaxPool2d(2, 2, return_indices=False)(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = nn.MaxPool2d(2, 2, return_indices=False)(x)
        x = self.upsample(x)
        return x

if __name__ == "__main__":
    model = Discriminator()
    summary(model, (1, 320, 320))