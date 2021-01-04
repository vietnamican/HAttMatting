import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader.DataLoader as DataLoader
import torchvision

import pytorch_lightning as pl

from .downpath import DownPath
from .conv_batchnorm_relu import ConvBatchNormRelu
from .uppath import UpPath


class Model(pl.LightningModule):
    def __init__(self, stage, lr=0.001, weight_decay=0, momentum=0.9):
        super(Model, self).__init__()
        self.stage = stage
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.down1 = DownPath(2, 3, 64, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down2 = DownPath(2, 64, 128, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down3 = DownPath(3, 128, 256, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down4 = DownPath(3, 256, 512, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down5 = DownPath(3, 512, 512, kernel_size=3,
                              stride=1, padding=1, bias=True)
        self.down = ConvBatchNormRelu(
            512, 512, kernel_size=3, padding=1, bias=True)

        self.trimap_5 = UpPath(512, 512, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.trimap_4 = UpPath(512, 512, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.trimap_3 = UpPath(512, 256, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.trimap_2 = UpPath(256, 128, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.trimap_1 = UpPath(128, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.trimap_conv1 = ConvBatchNormRelu(
            64, 64, kernel_size=5, padding=2, bias=True)
        self.trimap_conv2 = ConvBatchNormRelu(
            64, 3, kernel_size=5, padding=2, bias=True)

        self.alpha_5 = UpPath(1024, 512, kernel_size=1,
                              stride=1, padding=0, bias=True)
        self.alpha_4 = UpPath(1024, 512, kernel_size=5,
                              stride=1, padding=2, bias=True)
        self.alpha_3 = UpPath(1024, 256, kernel_size=5,
                              stride=1, padding=2, bias=True)
        self.alpha_2 = UpPath(512, 128, kernel_size=5,
                              stride=1, padding=2, bias=True)
        self.alpha_1 = UpPath(256, 64, kernel_size=5,
                              stride=1, padding=2, bias=True)
        self.alpha_conv1 = ConvBatchNormRelu(
            128, 64, kernel_size=5, padding=2, bias=True)
        self.alpha_conv2 = ConvBatchNormRelu(
            128, 1, kernel_size=5, padding=2, bias=True)

        self.refine_1 = ConvBatchNormRelu(
            7, 64, kernel_size=3, padding=1, bias=True)
        self.refine_2 = ConvBatchNormRelu(
            64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_3 = ConvBatchNormRelu(
            64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_pred = ConvBatchNormRelu(
            64, 1, kernel_size=3, padding=1, bias=True)
        if stage == 'train_alpha':
            self.freeze_trimap_path()

    def forward(self, x):
        l1, s1, x1, i1 = self.down1(x)
        l2, s2, x2, i2 = self.down2(x1)
        l3, s3, x3, i3 = self.down3(x2)
        l4, s4, x4, i4 = self.down4(x3)
        l5, s5, x5, i5 = self.down5(x4)
        x6 = self.down(x5)

        t5, x6t = self.trimap_5(x6, x5, i5, s5, return_conv_result=True)
        t5 = torch.add(t5, l5)
        t4 = torch.add(self.trimap_4(t5, x4, i4, s4), l4)
        t3 = torch.add(self.trimap_3(t4, x3, i3, s3), l3)
        t2 = torch.add(self.trimap_2(t3, x2, i2, s2), l2)
        t1 = torch.add(self.trimap_1(t2, x1, i1, s1), l1)
        raw_trimap = self.trimap_conv2(self.trimap_conv1(t1))

        if self.stage == 'train_alpha':
            d5 = torch.add(self.alpha_5(torch.cat((x6, x6t), 1), x5, i5, s5), l5)
            d4 = torch.add(self.alpha_4(torch.cat((d5, t5), 1), x4, i4, s4), l4)
            d3 = torch.add(self.alpha_3(torch.cat((d4, t4), 1), x3, i3, s3), l3)
            d2 = torch.add(self.alpha_2(torch.cat((d3, t3), 1), x2, i2, s2), l2)
            d1 = torch.add(self.alpha_1(torch.cat((d2, t2), 1), x1, i1, s1), l1)
            raw_alpha = self.alpha_conv2(
                torch.cat((self.alpha_conv1(torch.cat((d1, t1), 1)), t1), 1))
            pred_mattes = torch.sigmoid(raw_alpha)

            refine0 = torch.cat((x, raw_trimap, pred_mattes),  1)
            pred_refine = self.refine_pred(self.refine_3(
                self.refine_2(self.refine_1(refine0))))
            pred_alpha = torch.sigmoid(torch.add(raw_alpha, pred_refine))

            return raw_trimap, pred_alpha

        return raw_trimap, None
    def freeze_trimap_path(self):
        for name, p in self.named_parameters():
            print(name)
            if name.startswith('down') or name.startswith('trimap'):
                p.requires_grad = False
    def train_dataloader(self):
        return 
if __name__ == "__main__":

    model = Model()
    trainer = pl.Trainer(weighs_summary='full')
    # summary(model, (3, 800, 600))
