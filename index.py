import math
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from main import Model
from torchsummary import summary
from compose import compose


if __name__ == "__main__":

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = compose()
    trainer = pl.Trainer(precision=16, gpus=1,
                         benchmark=True, accumulate_grad_batches=4,
                         callbacks=[checkpoint_callback, lr_monitor])\
    trainer.fit(model)
