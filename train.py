from tqdm import tqdm
from time import time

import torch
import torchvision
from torchsummary import summary
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_learning_rate, \
    alpha_prediction_loss, alpha_prediction_loss_with_trimap, ssim_loss, trimap_prediction_loss
from config import device, im_size, grad_clip, print_freq

from model import Model
from data_tfrecord_2 import HADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device {}'.format(device))


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    scaler = GradScaler()
    # loss_function = alpha_prediction_loss
    loss_function = alpha_prediction_loss_with_trimap

    # Batches
    for i, (img, alpha_label, trimap_label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device)  # [N, 4, 320, 320]
        alpha_label = alpha_label.type(
            torch.FloatTensor).to(device)  # [N, 320, 320]
        alpha_label = alpha_label.unsqueeze(1)
        trimap_label = trimap_label.to(device)
        # alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]
        with autocast():
            # Forward prop.
            trimap_out, alpha_out = model(img)  # [N, 3, 320, 320]
            # alpha_out = alpha_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]

            # Calculate loss
            # loss = criterion(alpha_out, alpha_label)
            alpha_loss = alpha_prediction_loss_with_trimap(
                alpha_out, alpha_label, trimap_label)
            trimap_loss = trimap_prediction_loss(trimap_out, trimap_label)
            loss = 0.5*alpha_loss + 0.5*trimap_loss

        # Back prop.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status

        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                         epoch, i, len(train_loader), loss=losses)
            logger.info(status)
    writer.add_scalar('Train_Loss', losses.avg, epoch)
    writer.add_scalar('Learning_Rate', get_learning_rate(optimizer), epoch)
    save_checkpoint(epoch, 0, model, optimizer, losses.avg, False)
    return losses.avg


if __name__ == '__main__':
    global args
    args = parse_args()
    checkpoint = args.checkpoint
    global writer
    writer = SummaryWriter(logdir=args.logdir)
    global start_epoch
    start_epoch = args.start_epoch
    if checkpoint is None:
        torch.random.manual_seed(7)
        torch.cuda.manual_seed(7)
        np.random.seed(7)
        model = Model(args.stage)
        optimizer = torch.optim.Adam(model.parameters())
    else:
        checkpoint = torch.load(checkpoint)
        model_state_dict = checkpoint['model_state_dict']
        model = Model(args.stage)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        if args.reset_optimizer:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters())
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            optimizer.load_state_dict(optimizer_state_dict)
        if 'epoch' in checkpoint and checkpoint['epoch'] is not None:
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 1
        if 'torch_seed' in checkpoint and checkpoint['torch_seed'] is not None:
            torch.random.set_rng_state(checkpoint['torch_seed'])
        else:
            torch.random.manual_seed(7)
        if 'torch_cuda_seed' in checkpoint and checkpoint['torch_cuda_seed'] is not None:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_seed'])
        else:
            torch.cuda.manual_seed(7)
        if 'np_seed' in checkpoint and checkpoint['np_seed'] is not None:
            np.random.set_state(checkpoint['np_seed'])
        else:
            np.random.seed(7)
    summary(model, (3, 320, 320), depth=6)
    train_loader = DataLoader(
        HADataset('train'), batch_size=8, shuffle=True, pin_memory=True, num_workers=8)
    # optimizer = optim.Adam(model.parameters())
    total_training_time = 0
    n_epochs = args.end_epoch
    logger = get_logger()
    for epoch in range(start_epoch, n_epochs + 1):
        start = time()
        train(train_loader, model, optimizer, epoch, logger)
        end = time()
        print('\nTraning process takes {} seconds'.format(end - start))
        total_training_time += end - start
        # test(test_loader, model)
    # print('Total traning process takes {} seconds'.format(total_training_time))

# if __name__ == '__main__':
#     model = Model()
#     summary(model, (3, 320, 320), depth=4)
