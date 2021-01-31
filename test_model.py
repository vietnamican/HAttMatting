from tqdm import tqdm
from time import time

import torch
import torchvision
from torchsummary import summary
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from model import Model

if __name__ == '__main__':
    model = Model()
    model.remove_prefix_state_dict({}, prefix=0.1)