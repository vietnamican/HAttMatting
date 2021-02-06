from torch import nn
import torch

from .model import HAttMatting
from .base import Base


class Model(Base):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.lambda_adversarial_loss = 0.05
        self.lambda_recon_loss = 1
        self.lambda_ssim_loss = 0.01
        self.save_hyperparameters()
        self.hattmatting = HAttMatting()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss()

    def forward(self, x):
        return self.hattmatting(x)

    def dis_forward(self, image, alpha):
        return self.hattmatting.discriminator(image, alpha)

    def _gen_step(self, image, alpha):
        alpha_pred = self.hattmatting(image)
        print(alpha_pred.type(), alpha.type(), image.type())
        recon_loss = self.recon_criterion(alpha_pred, alpha.float())
        dis_logit = self.dis_forward(image, alpha_pred)
        adversarial_loss = self.adversarial_criterion(
            dis_logit, torch.ones_like(dis_logit))

        return recon_loss, adversarial_loss

    def _dis_step(self, image, alpha):
        alpha_pred = self.hattmatting(image)
        print('forward pred')
        fake_logit = self.dis_forward(image, alpha_pred)
        print('forward true')
        real_logit = self.dis_forward(image, alpha)
        print('pass forward true')
        fake_loss = self.adversarial_criterion(
            fake_logit, torch.zeros_like(fake_logit))
        real_loss = self.adversarial_criterion(
            real_logit, torch.ones_like(real_logit))

        return (fake_loss + real_loss) / 2

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, alpha, _, _ = batch
        image = image[..., 12:-12]
        alpha = alpha[..., 12:-12]
        alpha = alpha.unsqueeze(1).float()
        if optimizer_idx == 0:
            print("gen loss ssssssssssssssssssssssssssssssssss")
            recon_loss, adversarial_loss = self._gen_step(image, alpha)
            loss = self.lambda_recon_loss*recon_loss + \
                self.lambda_adversarial_loss*adversarial_loss
            self.log('Generator Loss', loss)
            return loss
        else:
            print("dis loss ssssssssssssssssssssssssssssssssss")
            adversarial_loss = self._dis_step(image, alpha)
            loss = self.lambda_adversarial_loss*adversarial_loss
            self.log('Discriminator Loss', loss)
            return loss

    def validation_step(self, batch, batch_idx):
        image, alpha, _, _ = batch
        image = image[..., 12:-12]
        alpha = alpha[..., 12:-12]
        alpha = alpha.unsqueeze(1).float()
        recon_loss, adversarial_loss = self._gen_step(image, alpha)
        loss = self.lambda_recon_loss*recon_loss + \
            self.lambda_adversarial_loss*adversarial_loss
        self.log('Validation Loss', loss)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        dis_opt = torch.optim.Adam(
            self.hattmatting.discriminator.parameters(), lr=lr)
        gen_opt = torch.optim.Adam(
            [
                {'params':self.hattmatting.feature_extraction.parameters()},
                {'params':self.hattmatting.aspp.parameters()},
                {'params':self.hattmatting.hierarchical_attention.parameters()}
            ],
            lr=lr)

        return [gen_opt, dis_opt]
