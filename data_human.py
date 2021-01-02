import math
import os
import random
from tqdm import tqdm

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tensorflow as tf

import tfrecord_creator
from config import im_size, unknown_code, fg_path, bg_path, a_path, num_valid
from utils import safe_crop, parse_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global args
args = parse_args()

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(
            brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def return_raw_image(dataset):
    dataset_raw = []
    for image_features in dataset:
        image_raw = image_features['image'].numpy()
        image = tf.image.decode_jpeg(image_raw)
        dataset_raw.append(image)

    return dataset_raw


bg_dataset = tfrecord_creator.read("bg", "./data/tfrecord/")
bg_dataset = list(bg_dataset)
print("___________________")
print(len(bg_dataset))
print("___________________")


def get_raw(type_of_dataset, count):
    if type_of_dataset == 'fg':
        temp = fg_dataset[count]['image']
        channels = 3
    elif type_of_dataset == 'bg':
        temp = bg_dataset[count]['image']
        channels = 3
    else:
        temp = a_dataset[count]['image']
        channels = 1
    temp = tf.image.decode_jpeg(temp, channels=channels)
    temp = np.asarray(temp).astype(np.float32)
    return temp


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()
with open('Combined_Dataset/Training_set/training_bg_names.txt') as f:
    bg_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_bg_names.txt') as f:
    bg_test_files = f.read().splitlines()


def composite4(fg, bg, a, w, h):
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    if bg.ndim == 2:
        bg = np.reshape(bg, (h, w, 1))
    bg = bg[y:y + h, x:x + w]
    bg = np.reshape(bg, (h, w, -1))
    fg = np.reshape(fg, (h, w, -1))
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg


def process(img_path, alpha_path, bcount):
    img_root_path = args.img_root_path
    img_path = os.path.join(img_root_path, img_path)
    alpha_path = os.path.join(img_root_path, alpha_path)
    im = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    a = cv.imread(alpha_path, cv.IMREAD_UNCHANGED)
    a = a[:, :, 3]
    h, w = im.shape[:2]
    bg = get_raw("bg", bcount)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio),
                                      math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4(im, bg, a, w, h)


def gen_trimap(alpha):
    if args.stage == 'train_alpha':
        k_size = 5
    else:
        k_size = random.choice(range(1, 12))
    iterations = np.random.randint(1, 20)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv.dilate(alpha, kernel, iterations)
    eroded = cv.erode(alpha, kernel, iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


class HADataset(Dataset):
    def __init__(self, split):
        super(HADataset, self).__init__()
        self.split = split

        filename = '{}_names.txt'.format(split)
        with open("img.txt", 'r') as f:
            self.imgs = f.read().splitlines()
        with open("alpha.txt", 'r') as f:
            self.alpha = f.read().splitlines()

        split_index = 9 * len(self.imgs) // 10
        if split == 'train':
            self.imgs = self.imgs[:split_index]
            self.alpha = self.alpha[:split_index]
        else:
            self.imgs = self.imgs[split_index:]
            self.alpha = self.alpha[split_index:]

        self.num_bgs = 43100

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        img_path = self.imgs[i]
        alpha_path = self.alpha[i]
        bcount = np.random.randint(self.num_bgs)
        # size 800x600

        img, alpha, _, _ = process(img_path, alpha_path, bcount)
        trimap = gen_trimap(alpha)

        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap).copy()
            alpha = np.fliplr(alpha)

        return self.transformer(img), alpha / 255.0, trimap, img_path

    def __len__(self):
        return len(self.imgs)


def gen_names():
    num_fgs = 431
    num_bgs = 43100
    num_bgs_per_fg = 100

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    valid_names = random.sample(names, num_valid)
    train_names = [n for n in names if n not in valid_names]

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == "__main__":
    dataset = HADataset(split='train')
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=16)
    for i, (image, alpha, trimap) in enumerate(tqdm(dataloader)):
        print(image.size(), alpha.size(), trimap.size())
