import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import numpy as np

from data import MattingDataset, RepeatDataset
from model import Model
from models.config import config

pl.seed_everything(42)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model.migrate(state_dict, force=True)

def load_trainer(logdir, device, max_epochs, checkpoint=None):

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name=logdir,
    )
    lr_monitor = LearningRateMonitor(log_momentum=False)
    loss_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        save_top_k=5,
        mode='min',
        save_last=True
    )
    callbacks = [loss_callback, lr_monitor]
    resume_from_checkpoint = checkpoint
    if device == 'tpu':
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            tpu_cores=8,
            resume_from_checkpoint=resume_from_checkpoint
        )
    elif device == 'gpu':
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            gpus=[1],
            resume_from_checkpoint=resume_from_checkpoint
        )
    else:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            resume_from_checkpoint=resume_from_checkpoint
        )

    return trainer

def load_data():
    batch_size = config['batch_size']
    data_root = '../datasets/matting_human_half/'
    train_dataset = MattingDataset(data_root, set_type='train')
    val_dataset = MattingDataset(data_root, set_type='val')
    train_dataset = RepeatDataset(train_dataset, 2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
    return train_dataloader, val_dataloader

model = Model()
checkpoint_path = 'log_logs/version_1/checkpoints/last.ckpt'

epoch = config['epoch']
trainer = load_trainer('log_logs', 'gpu', epoch, checkpoint=checkpoint_path)
train_dataloader, val_dataloader = load_data()
trainer.fit(model, train_dataloader, val_dataloader)
