import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

from model import Model
from torchsummary import summary
from compose import compose

if __name__ == "__main__":

    model = Model('train_trimap')
    summary(model, (3, 320, 320), depth=5)

    model = compose()
    trainer = pl.Trainer(precision=16, gpus=1, benchmark=True, accumulate_grad_batches=4)
    trainer.fit(model)
