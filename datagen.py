from __future__ import print_function

import torch
import torch.utils.data as data

import random
from PIL import Image
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = [
    'ListDataset'
]

class ListDataset(data.Dataset):
    def __init__(self, img_list, transform, size):
        self.transform = transform
        with open(img_list, 'r') as f:
            self.imgs = [x.strip() for x in f.readlines()]
        self.imgs = [Image.open(x) for x in self.imgs]

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError

        self.img_nums = len(self.imgs)
        self.sample_num = 200000

    def __getitem__(self, idx):
        idx = random.randint(0, self.img_nums-1)
        # img_path = self.imgs[idx]
        # img = Image.open(img_path)
        img = self.imgs[idx]

        nimg = self.random_crop(img, self.size)

        # nimg = nimg.resize(self.size)
        nimg = self.transform(nimg)
        return nimg, np.array([0])

    def random_crop(self, img, size):
        # ratio = random.uniform(0.8, 1.2)
        w, h = img.size
        # nw, nh = int(size * ratio), int(size * ratio)
        nw, nh = size[0], size[1]
        nx, ny = random.randint(0, w - nw - 1), random.randint(0, h - nh - 1)
        nimg = img.crop([nx, ny, nx+nw, ny+nh])
        return nimg

    def crop(self, img, size):
        w, h = img.size
        assert size[0] < w and size[1] < h

        nx = random.randint(0, w-size[0]-1)
        ny = random.randint(0, h-size[1]-1)

        nimg = img.crop([nx, ny, nx+size[0], ny+size[1]])

        return nimg

    def __len__(self):
        return self.sample_num
