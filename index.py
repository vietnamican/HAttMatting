import math
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

from main import Model
from torchsummary import summary
from compose import compose


if __name__ == "__main__":

    model = compose()
    trainer = pl.Trainer(precision=16, tpu_cores=8, benchmark=True, progress_bar_refresh_rate=500, accumulate_grad_batches=4) # resume_from_checkpoint='lightning_logs/version_1/checkpoints/epoch=12-step=12278.ckpt'
    trainer.fit(model)
