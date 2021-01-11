import math
import argparse

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device, fg_path_test, a_path_test, bg_path_test
from data_gen import data_transforms, fg_test_files, bg_test_files
from utils import compute_mse, compute_sad, AverageMeter, get_logger, compute_gradient_loss, compute_connectivity_error, ensure_folder, draw_str
from model import Model

device = 'cuda'
output_folder = 'trimap_human_2_32_train'

def val(val_loader, model):
    mse_losses = AverageMeter()
    sad_losses = AverageMeter()
    gradient_losses = AverageMeter()
    connectivity_losses = AverageMeter()

    model.eval()
    
    # Batches
    for i, (img, alpha_label, trimap_label, img_path) in enumerate(val_loader):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device)  # [N, 4, 320, 320]
        alpha_label = alpha_label.type(
            torch.FloatTensor).to(device)  # [N, 320, 320]
        alpha_label = alpha_label.unsqueeze(1)
        trimap_label = trimap_label.to(device)
        # Forward prop.
        trimap_out, alpha_out = model(img)  # [N, 3, 320, 320]
        trimap_out.squeeze(0)
        # alpha_out = alpha_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]
        trimap_out = trimap_out.argmax(dim=1)
        trimap_out = trimap_out.squeeze(0)
        trimap_out[trimap_out==1] = 128
        trimap_out[trimap_out==2] = 255
        trimap_out = np.array(trimap_out.cpu(), dtype=np.uint8)
        # print(trimap_out)
        # return trimap, alpha
        mse_loss = compute_mse(alpha_out, alpha_label, trimap_label)
        sad_loss = compute_sad(alpha_out, alpha_label)
        gradient_loss = compute_gradient_loss(alpha_out, alpha_label, trimap_label)
        connectivity_loss = compute_connectivity_error(alpha_out, alpha_label, trimap_label)
        print("sad:{} mse:{} gradient: {} connectivity: {}".format(sad_loss.item(), mse_loss.item(), gradient_loss, connectivity_loss))
        # f.write("sad:{} mse:{} gradient: {} connectivity: {}".format(sad_loss.item(), mse_loss.item(), gradient_loss, connectivity_loss) + "\n")

        alpha_out = (alpha_out.copy() * 255).astype(np.uint8)
        draw_str(alpha_out, (10, 20), "sad:{} mse:{} gradient: {} connectivity: {}".format(sad_loss.item(), mse_loss.item(), gradient_loss, connectivity_loss))
        cv.imwrite(os.path.join('images/test/out/', output_folder, img_path[0].split('/')[-1]), alpha_out)
        # print(os.path.join('images/test/out', output_folder, img_path[0].split('/')[-1]))
        # cv.imwrite(os.path.join('images/test/out', output_folder, img_path[0].split('/')[-1]), alpha_out)
    print("sad_avg:{} mse_avg:{} gradient_avg: {} connectivity_avg: {}".format(sad_losses.avg, mse_losses.avg, gradient_losses.avg, connectivity_losses.avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='checkpoint.txt')
    parser.add_argument('--checkpoint', type=str, default='BEST_checkkpoint.tar')
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()
    try:
        os.makedirs(ps.path.join('images', 'test', 'out', output_folder))
    except:
        pass

    checkpoint = args.checkpoint
    if args.device == 'cpu':
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(checkpoint)
    model_state_dict = checkpoint['state_dict']
    model = Model('train_alpha').to(device)
    model.load_state_dict(model_state_dict)
    val_loader  = DataLoader(HADataset('valid'), batch_size=1, shuffle=False, num_workers=2)
    val(val_loader, model)


    
