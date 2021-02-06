import math
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from model import Model
from utils import parse_args
from data_human import HADataset

global args
args = parse_args()


if __name__ == "__main__":
    pl.seed_everything(42)
    checkpoint_callback = ModelCheckpoint(
        monitor='Validation Loss',
        dirpath='',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min',
    )
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = Model()
    # print(type(model.hattmatting.discriminator.parameters()))
    train_dataloader = DataLoader(
        HADataset('train'), batch_size=2, shuffle=True, pin_memory=True, num_workers=2)
    val_dataloader = DataLoader(HADataset(
        'valid'), batch_size=2, shuffle=False, pin_memory=True, num_workers=2)
    trainer = pl.Trainer(
        # precision=16,
        gpus=0,
        benchmark=True, accumulate_grad_batches=4,
        progress_bar_refresh_rate=200,
        callbacks=[checkpoint_callback],
        fast_dev_run=True
        #  resume_from_checkpoint
    )
    trainer.fit(model, train_dataloader, val_dataloader)
