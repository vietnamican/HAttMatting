import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class Visualization(nn.Module):
    def __init__(self):
        super(Visualization, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1)
        return x