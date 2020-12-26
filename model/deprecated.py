import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class InnerChannelAttention(nn.Module):
    def __init__(self):
        super(InnerChannelAttention, self).__init__()
        self.mlp1 = nn.Conv1d(1, 512, 1)
        self.mlp2 = nn.Conv1d(512, 256, 1)
        self.mlp3 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x_input = x
        print(x_input.shape)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))
        x = F.relu(self.bn3(self.mlp3(x)))
        x = x.transpose(2, 1)
        x = self.global_avg_pool(x)
        x = torch.sigmoid(x)
        # print(x.shape)
        scale = x.unsqueeze(-1).expand_as(x_input)
        print(scale.shape)
        x = x_input * scale
        print(x.shape)
        return x


class InterChannelAttention(nn.Module):
    def __init__(self, inplanes, encode_channels=128, pool_types=['avg', 'max']):
        super(InterChannelAttention, self).__init__()
        self.inplanes = inplanes
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inplanes, encode_channels),
            nn.ReLU(),
            nn.Linear(encode_channels, inplanes)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            print(channel_att_raw.shape)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            print(channel_att_raw.shape)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand_as(x)
        return x * scale


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.features_extractor = FeatureExtractor()
#         self.aspp = ASPP(512, 16, nn.BatchNorm2d)
#         self.pyramidal_features_distillation = PyramidalFeaturesDistillation()
#         self.visualization = Visualization()
#         self.apperance_cues_filtration = ApperanceCuesFiltration()
#         self.refinement_network = RefinementNetwork()
#         self.refinement_network.load_state_dict(torch.load('refinement_checkpoint.pth'))

#     def forward(self, x):
#         original_feature = x
#         low_level_feature, high_level_feature = self.features_extractor(x)
#         x = self.aspp(high_level_feature)
#         x = self.pyramidal_features_distillation(x)
#         visualize = self.visualization(x)
#         x = self.apperance_cues_filtration(x, low_level_feature)
#         x = torch.cat((original_feature, x), 1)
#         _, x = self.refinement_network(x)
#         return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encode_network = EncodeNetwork()
        # self.predict_alpha = PredictAlpha()
        # self.predict_trimap = PredictTrimap()
        # self.refinement_network = RefinementNetwork()
        # self.refinement_network = RefinementNetwork()
        # self.refinement_network.load_state_dict(torch.load('refinement_checkpoint.pth'))
    def forward(self, x):
        return self.encode_network(x)