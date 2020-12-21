import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class SpatialAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, inplanes, K=512):
        """
        :param feature_map: feature map in level L
        """
        super(SpatialAttention, self).__init__()
        C = inplanes
        self.register_parameter('Ws', nn.Parameter(
            nn.init.xavier_normal_(torch.empty(C, K))))
        self.register_parameter('Wi', nn.Parameter(
            nn.init.xavier_normal_(torch.empty(K, 1))))
        self.register_parameter('bs', nn.Parameter(
            nn.init.zeros_(torch.empty(K))))
        self.register_parameter('bi', nn.Parameter(
            nn.init.zeros_(torch.empty(1))))
        # self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=0)  # softmax layer to calculate weights

    def forward(self, feature_map):
        """
        Forward propagation.
        :param feature_map: feature map in level L(batch_size, C, H, W)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: alpha
        """
        V_map = feature_map.view(
            feature_map.shape[0], feature_map.shape[1], -1)
        V_map = V_map.permute(0, 2, 1)  # (batch_size,W*H,C)
        att = nn.Tanh()((torch.matmul(V_map, self.Ws) + self.bs))
        alpha = nn.Softmax(dim=0)(torch.matmul(att, self.Wi) + self.bi)
        alpha = alpha.squeeze(2)
        temp_feature_map = feature_map.view(
            feature_map.shape[0], feature_map.shape[1], -1)
        temp_alpha = alpha.unsqueeze(1)
        attention_weighted_encoding = torch.mul(temp_feature_map, temp_alpha)
        attention_weighted_encoding = attention_weighted_encoding.view(feature_map.shape)
        return attention_weighted_encoding, alpha


class ChannelWiseAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, feature_map, K=512):
        """
        :param feature_map: feature map in level L
        :param decoder_dim: size of decoder's RNN
        """
        super(ChannelWiseAttention, self).__init__()
        self.register_parameter('Wc', nn.Parameter(
            nn.init.xavier_normal_(torch.empty(1, K))))
        self.register_parameter('Wi_hat', nn.Parameter(
            nn.init.xavier_normal_(torch.empty(K, 1))))
        self.register_parameter('bc', nn.Parameter(
            nn.init.zeros_(torch.empty(K))))
        self.register_parameter('bi_hat', nn.Parameter(
            nn.init.zeros_(torch.empty(1))))
        self.Wc = nn.Parameter(torch.randn(1, K))
        self.Wi_hat = nn.Parameter(torch.randn(K, 1))
        self.bc = nn.Parameter(torch.randn(K))
        self.bi_hat = nn.Parameter(torch.randn(1))
        # self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=0)  # softmax layer to calculate weights

    def forward(self, feature_map):
        """
        Forward propagation.
        :param feature_map: feature map in level L(batch_size, C, H, W)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: alpha
        """
        V_map = feature_map.view(
            feature_map.shape[0], feature_map.shape[1], -1) .mean(dim=2)
        V_map = V_map.unsqueeze(2)  # (batch_size,C,1)
        # (batch_size,C,K)
        att = nn.Tanh()((torch.matmul(V_map, self.Wc) + self.bc))
        beta = nn.Softmax(dim=0)(torch.matmul(att, self.Wi_hat) + self.bi_hat)
        beta = beta.unsqueeze(2)
        attention_weighted_encoding = torch.mul(feature_map, beta)

        return attention_weighted_encoding, beta


if __name__ == "__main__":
    model = SpatialAttention(256)
    summary(model, (256, 160, 160))
