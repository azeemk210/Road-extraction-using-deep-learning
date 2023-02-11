#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:42:28 2021

@author: azeem
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Road_dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
dataset= Road_dataset("train/data", "train/seg", transform=None)
for x,y in iter(dataset):
    print(x,y)
    break
   