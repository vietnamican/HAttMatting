import torch
from pthflops import count_ops
from torchinfo import summary

from model import Model

model = Model()
x = torch.rand(2, 3, 320, 320)
model(x)
count_ops(model, x)

summary(model, (2, 3, 320, 320))