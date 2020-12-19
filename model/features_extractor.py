import torch
import torch.nn as nn
from torchsummary import summary



# print(model.conv1)
# summary(model,(3, 320, 320))

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = torch.hub.load('pytorch/vision', 'resnext50_32x4d', pretrained=True)
        self.conv1 = model.conv1
        self.forward_path = nn.Sequential(
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            # model.layer3,
            # model.layer4
        )
        # summary(self.conv4, (1024, 20, 20))
    def forward(self, x):
        x = self.conv1(x)
        x = self.forward_path(x)
        return x


if __name__ == "__main__":
    model = FeatureExtractor()
    # model = torch.hub.load('pytorch/vision', 'resnext50_32x4d', pretrained=True)
    # print(model)
    # for layer in model.modules():
    #     print(layer.__dir__())
    summary(model, (3, 320, 320), depth=5)