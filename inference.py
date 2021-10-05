import os
import cv2
import numpy as np

import albumentations as A
import albumentations.pytorch as AP
import glob
import torch

from models import Model


def make_sure_dir_exists(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def load_state_dict(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    state_dict = checkpoint['state_dict']
    model.migrate(state_dict, force=True, verbose=2)
    model.freeze_with_prefix('')
    model = model.to('cuda:1')

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

transformer = A.Compose(
    [
        # A.Resize(320, 320),
        A.Normalize(mean=mean, std=std),
        AP.ToTensorV2()
    ])

checkpoint_path = 'log_logs/version_0/checkpoints/last.ckpt'
model = Model()
load_state_dict(checkpoint_path, model)
model.eval()

image_dir = '/home/ubuntu/Workspace/datasets/matting_human_half/clip_img/1803151818/clip_00000000/'
outdir = 'output/outdir/'
images_path = glob.glob(image_dir+"*.jpg", recursive=True)
for image_path in images_path:
    image = cv2.imread(image_path)
    image = transformer(image=image)['image']
    image = image.unsqueeze(0).to('cuda:1')
    logit = model(image)
    logit = logit.squeeze(0).squeeze(0)
    mask = logit.cpu().numpy()
    mask *= 255
    mask = mask.astype(np.uint8)
    print(mask.shape)
    out_path = image_path.replace(image_dir, outdir)
    print(out_path)
    make_sure_dir_exists(out_path)
    cv2.imwrite(out_path, mask)
# for image_path in images_path:
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     image = transformer(image=image)['image']
#     mask = image[3]
#     mask *= 255
#     mask = mask.numpy().astype(np.uint8)
#     out_path = image_path.replace(image_dir, outdir)
#     print(out_path)
#     make_sure_dir_exists(out_path)
#     cv2.imwrite(out_path, mask)
