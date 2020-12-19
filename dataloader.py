import os

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms

from PIL import Image
import numpy as np

image_dir = 'dogs-vs-cats/train'

images = os.listdir(image_dir)
# images = images[:100]

training_transformer = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

class HADataset(data.Dataset):
    def __init__(self, split):
        split_index = 4*len(images)//5
        if split == 'train':
            self.images = images[:split_index]
        else:
            self.images = images[split_index:]
        self.transformer = training_transformer
        print(type(self.transformer))
    def __getitem__(self, i):
        image = self.images[i]

        # label = 0 -> dog
        # label = 1 -> cat
        if 'cat' in image:
            label = 1
        else:
            label = 0
        
        image = Image.open(os.path.join(image_dir, image))
        # image = np.array(image)
        image = self.transformer(image)
        return torch.Tensor(image), label
    
    def __len__(self):
        return len(self.images)