import os
import os.path as osp
import cv2
from tqdm import tqdm
from functools import partial
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP

mean=(0.485, 0.456, 0.406, 0)
std=(0.229, 0.224, 0.225, 1)

color_transformer = {
    'train': A.ColorJitter(brightness=0.35, contrast=0.5, saturation=0.5, hue=0.2, always_apply=False, p=0.7),
    'val': lambda image: dict(image=image)
}

transformer = {
    'train': A.Compose(
    [
        A.HorizontalFlip(p=0.5),  ## Becareful when using that, because the keypoint is flipped but the index is flipped too
        A.Affine(scale=(-0.25, 0.25), translate_percent=(-0.125, 0.125), rotate=(-40, 40), mode=4, always_apply=False, p=0.5),
        A.RandomSizedCrop(min_max_height=[320, 600], width=320, height=320, p=0.5),
        A.Resize(320, 320),
        A.Normalize(mean=mean, std=std),
        AP.ToTensorV2()
    ]),
    'val': A.Compose(
    [
        A.Resize(320, 320),
        A.Normalize(mean=mean, std=std),
        AP.ToTensorV2()
    ]),
}

def _transform(image, set_type='train'):
    image = transformer[set_type](image=image)['image']
    return image

def _color_transform(image, set_type='train'):
    image = color_transformer[set_type](image=image)['image']
    return image

class MattingDataset(Dataset):
    def __init__(self, data_root, set_type='train'):
        super().__init__()
        self.data_root = data_root
        self.set_type = set_type
        self.images_dir = 'clip_img'
        self.labels_dir = 'matting'
        self.images_root = osp.join(self.data_root, self.images_dir)
        self.labels_root = osp.join(self.data_root, self.labels_dir)
        self.transformer = partial(_transform, set_type=self.set_type)
        self.color_transformer = partial(_color_transform, set_type=self.set_type)
        self.load_annotations()
        split_index = -1024
        if self.set_type == 'train':
            self.images_path = self.images_path[:split_index]
            self.labels_path = self.labels_path[:split_index]
        elif self.set_type == 'val':
            self.images_path = self.images_path[split_index:]
            self.labels_path = self.labels_path[split_index:]

    def load_annotations(self):
        self.images_path = [os.path.join(r, f) for r, _, fs in os.walk(self.images_root) for f in fs if osp.splitext(f)[1] == '.jpg']
        self.images_path.sort()
        self.labels_path = [image_path.replace(self.images_dir, self.labels_dir).replace('jpg', 'png').replace('clip', 'matting') for image_path in self.images_path]

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        label_path = self.labels_path[idx]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            return self.__getitem__(random.randint(0, self.__len__()-1))
        label = label[:,:,3:4]
        image = self.color_transformer(image)
        image_rgba = np.concatenate([image, label], axis=-1)
        image_rgba= self.transformer(image_rgba)
        return image_rgba[:3], image_rgba[3:4]
    
    def __len__(self):
        return len(self.images_path)