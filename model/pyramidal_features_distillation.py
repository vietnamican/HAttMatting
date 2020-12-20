import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

from .channel_attention import ChannelWiseAttention


class PyramidalFeaturesDistillation(nn.Module):
    def __init__(self):
        super(PyramidalFeaturesDistillation, self).__init__()
        self.up = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.channel_wise_attention = ChannelWiseAttention(256)
        # self.inner_channel_attention = InnerChannelAttention()
        # self.inter_channel_attention = InterChannelAttention(256, 128)

    def forward(self, x):
        x = self.up(x)
        x, alpha = self.channel_wise_attention(x)
        # x = self.inner_channel_attention(x)
        # x = self.inter_channel_attention(x)
        return x


if __name__ == '__main__':
    model = PyramidalFeaturesDistillation()
    summary(model, (256, 40, 40))
