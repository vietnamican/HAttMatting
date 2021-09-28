import torch
from torch import nn as nn
from torch import optim as optim

from models.base import Base
from models.model import Model as Generator
from models.discriminator import Discriminator as Discriminator
from pytorch_ssim import SSIM
from models.config import config

class Model(Base):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion_ssim = SSIM()
        self.criterion_mse = nn.MSELoss()
        self.criterion_adv = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, mask = batch
        lambda_adv, lambda_mse, lambda_ssim = config['lambda_adv'], config['lambda_mse'], config['lambda_ssim']
        if optimizer_idx == 0:
            generate = self.generator(image)
            loss_ssim = self.criterion_ssim(generate, mask)
            loss_mse = self.criterion_mse(generate, mask)
            logit = self.discriminator(generate)
            label = torch.ones(logit.size(0), logit.size(2), logit.size(3),dtype=torch.int64, device=logit.device)
            loss_adv = self.criterion_adv(logit, label)
            loss = lambda_adv*loss_adv + lambda_mse*loss_mse + lambda_ssim*loss_ssim
        else:
            generate = self.generator(image)
            loss_ssim = self.criterion_ssim(generate, mask)
            loss_mse = self.criterion_mse(generate, mask)
            # for fake
            logit_fake = self.discriminator(generate)
            label_fake = torch.zeros(logit_fake.size(0), logit_fake.size(2), logit_fake.size(3), dtype=torch.int64, device=logit_fake.device)
            loss_adv_fake = self.criterion_adv(logit_fake, label_fake)
            # for real
            logit_real = self.discriminator(mask)
            label_real = torch.ones(logit_real.size(0), logit_real.size(2), logit_real.size(3), dtype=torch.int64, device=logit_real.device)
            loss_adv_real = self.criterion_adv(logit_real, label_real)
            loss_adv = (loss_adv_fake + loss_adv_real) / 2
            loss = lambda_adv*loss_adv + lambda_mse*loss_mse + lambda_ssim*loss_ssim
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        lambda_adv, lambda_mse, lambda_ssim = config['lambda_adv'], config['lambda_mse'], config['lambda_ssim']
        generate = self.generator(image)
        loss_ssim = self.criterion_ssim(generate, mask)
        loss_mse = self.criterion_mse(generate, mask)
        # for fake
        logit_fake = self.discriminator(generate)
        label_fake = torch.zeros(logit_fake.size(0), logit_fake.size(2), logit_fake.size(3), dtype=torch.int64, device=logit_fake.device)
        loss_adv_fake = self.criterion_adv(logit_fake, label_fake)
        # for real
        logit_real = self.discriminator(mask)
        label_real = torch.ones(logit_real.size(0), logit_real.size(2), logit_real.size(3), dtype=torch.int64, device=logit_real.device)
        loss_adv_real = self.criterion_adv(logit_real, label_real)
        loss_adv = (loss_adv_fake + loss_adv_real) / 2
        loss = lambda_adv*loss_adv + lambda_mse*loss_mse + lambda_ssim*loss_ssim
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        lr = config['lr']
        epoch = config['epoch']
        g_optmizer = optim.SGD(self.generator.parameters(), lr=lr, weight_decay=1e-5)
        g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optmizer, T_max=epoch)
        d_optmizer = optim.SGD(self.discriminator.parameters(), lr=lr, weight_decay=1e-5)
        d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optmizer, T_max=epoch)
        return [g_optmizer, d_optmizer], [g_scheduler, d_scheduler]