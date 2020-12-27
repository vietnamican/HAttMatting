import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn.functional as F
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ssim_loss(y_pred, y_true):
    if len(y_pred.shape) == 3:
        y_pred = y_pred.unsqueeze(0)
    if len(y_true.shape) == 3:
        y_true = y_true.unsqueeze(0)
    return 1 - ms_ssim(y_pred, y_true, data_range=1, size_average=True)


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        criterion = criterion.to(device)

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)

        criterion = criterion.to(device)

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


def trimap_prediction_loss(trimap_pred, trimap_true):
    trimap_true[trimap_true == 0] = 0
    trimap_true[trimap_true == 128] = 1
    trimap_true[trimap_true == 255] = 2
    criterion = SegmentationLosses(batch_average=True).build_loss('ce')
    return criterion(trimap_pred, trimap_true)


def alpha_prediction_loss(y_pred, y_true):
    diff = y_pred - y_true
    diff = diff
    return torch.sum(torch.sqrt(torch.pow(diff, 2) + epsilon_sqr)) / (y_true.numel() + epsilon)


def alpha_prediction_loss_with_trimap(y_pred, y_true, trimap):
    weighted = torch.zeros(trimap.shape, device=device)
    weighted[trimap == 128] = 1.
    diff = y_pred - y_true
    diff = diff * weighted
    alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
    return torch.sum(alpha_loss) / (weighted.sum() + 1.)


class LossFunction(object):
    def __init__(self, stage):
        self.stage = stage
        self.trimap_criterion = SegmentationLosses(
            batch_average=True).build_loss('ce')

    def __call__(self):
        if self.stage == 'train_trimap':
            return self.trimap_loss
        elif self.stage == 'train_alpha':
            return self.fusion_loss

    def trimap_loss(self, trimap_pred, trimap_true):
        mask = torch.zeros(trimap_true.shape, device=device)
        # mask[trimap_true == 0] = 0
        mask[trimap_true == 128] = 1
        mask[trimap_true == 255] = 2
        return self.trimap_criterion(trimap_pred, mask)

    def alpha_prediction_loss_with_trimap(self, y_pred, y_true, trimap):
        weighted = torch.zeros(trimap.shape, device=device)
        weighted[trimap == 128] = 1.
        diff = y_pred - y_true
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
        return torch.sum(alpha_loss) / (weighted.sum() + 1.)

    def fusion_loss(self, y_pred, y_true, trimap_pred, trimap_true):
        return self.alpha_prediction_loss_with_trimap(y_pred, y_true, trimap_true) + 0.01*self.trimap_loss(trimap_pred, trimap_true)
