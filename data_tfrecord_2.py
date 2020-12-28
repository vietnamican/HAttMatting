import math
import os
import random

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
        transforms.ColorJitter(
            brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
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


fg_dataset = tfrecord_creator.read("fg", "./data/tfrecord/")
bg_dataset = tfrecord_creator.read("bg", "./data/tfrecord/")
a_dataset = tfrecord_creator.read("a",  "./data/tfrecord/")
fg_dataset = list(fg_dataset)
bg_dataset = list(bg_dataset)
a_dataset = list(a_dataset)
print("___________________")
print(len(fg_dataset))
print(len(bg_dataset))
print(len(a_dataset))
print("___________________")
# fg_raw = return_raw_image(fg_dataset)
# bg_raw = return_raw_image(bg_dataset)
# a_raw  = return_raw_image(a_dataset)


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
    # temp = transforms.ToTensor()(np.asarray(temp))
    # temp = torch.Tensor(temp, device=device, dtype=torch.float16)
    temp = np.asarray(temp)
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


def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('data/mask', name)
    alpha = cv.imread(filename, 0)
    return alpha


def get_alpha_test(name):
    fg_i = int(name.split("_")[0])
    name = fg_test_files[fg_i]
    filename = os.path.join('data/mask_test', name)
    alpha = cv.imread(filename, 0)
    return alpha


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    if bg.ndim == 2:
        bg = np.reshape(bg, (h, w, 1))
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    bg = np.reshape(bg, (h, w, -1))
    fg = np.reshape(fg, (h, w, -1))
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg


def process(fcount, bcount):
    im = get_raw("fg", fcount)
    a = get_raw("a", fcount)
    # a = a.view(a.shape[0], a.shape[1])
    a = np.reshape(a, (a.shape[0], a.shape[1]))
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


# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, different_sizes=[(320, 320), (480, 480), (640, 640)]):
    crop_size = random.choice(different_sizes)
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y, crop_size


class HADataset(Dataset):
    def __init__(self, split):
        super(HADataset, self).__init__()
        self.split = split

        filename = '{}_names.txt'.format(split)
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()

        # fgs = np.repeat(np.arange(num_fgs), args.batch_size * 8)
        # np.random.shuffle(fgs)
        # split_index = int(fgs.shape[0] * (1 - valid_ratio))
        # self.fgs = fgs
        # if split == 'train':
        #     self.fgs = fgs[:split_index]
        # else:
        #     self.fgs = fgs[split_index:]
        # self.fg_num = np.unique(self.fgs).shape[0]

        # print(self.fg_num)

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        name = self.names[i]
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        # fcount = self.fgs[i]
        # bcount = np.random.randint(num_bgs)
        img, alpha, _, _ = process(fcount, bcount)

        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        # crop_size = random.choice(different_sizes)

        trimap = gen_trimap(alpha)
        x, y, crop_size = random_choice(trimap, different_sizes)
        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)
        trimap = safe_crop(trimap, x, y, crop_size)

        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap).copy()
            alpha = np.fliplr(alpha)

        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        x = img

        # y = np.empty((2, im_size, im_size), dtype=np.float32)
        y = alpha / 255.
        # mask = np.equal(trimap, 128).astype(np.float32)
        # y[1, :, :] = mask

        return x, y, trimap

    def __len__(self):
        return len(self.names)


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
    gen_names()
