import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidalFeaturesDistillation(nn.Module):
    def __init__(self):
        super(PyramidalFeaturesDistillation, self).__init__()
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_mlp1 = nn.Conv1d(256, 128, 1)
        self.shared_mlp2 = nn.Conv1d(128, 256, 1)
    def forward(self, x):
        x = self.up(x)
        x = self.global_avg_pool(x)
        x = torch.squeeze(x, dim=-1)
        x = self.shared_mlp1(x)
        x = self.shared_mlp2(x)
        return x