from tqdm import tqdm
from time import time

import torch
import torchvision
from torchsummary import summary
# from torchsummaryX import summary
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from model import Model

if __name__ == '__main__':
    model = Model('train_alpha')
    summary(model, (3, 322, 322))
    # inp = torch.zeros((2, 3, 320, 320))
    # out1, out2 = model(inp)
    # print(out1.shape, out2.shape)
    # summary(model, torch.zeros((2, 3, 320, 320)))