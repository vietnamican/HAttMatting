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
    summary(model, (3, 800, 576), col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], depth=4)
    summary(model.hattmatting.discriminator, [(3, 800, 576), (1, 800, 576)], col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], depth=4)
