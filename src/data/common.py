import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TVF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks


def random_flip(image, mask):
    random_horizontal_flip = random.random()
    random_vertical_flip = random.random()
    if random_horizontal_flip < 0.5:
        image = transforms.RandomHorizontalFlip(p=1.0)(image)
        mask = transforms.RandomHorizontalFlip(p=1.0)(mask)
    if random_vertical_flip < 0.5:
        image = transforms.RandomVerticalFlip(p=1.0)(image)
        mask = transforms.RandomVerticalFlip(p=1.0)(mask)
    return image, mask


def crop(image, mask, min_size=256, max_size=512, size=(512, 512)):
    width = random.randint(min_size, max_size)
    ratio = size[0] / size[1]
    height = int(ratio * width)
    i, j, h, w = transforms.RandomCrop.get_params(
        image,
        output_size=(height, width),
    )
    image = TVF.crop(image, i, j, h, w)
    mask = TVF.crop(mask, i, j, h, w)
    image = transforms.Resize(size)(image)
    mask = transforms.Resize(size)(mask)
    return image, mask


def random_rotate(image, location):
    x_min, y_min, x_max, y_max = location
    img_size = image.size[0]
    rotate = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, None]
    random_rotate = random.randint(0, 3)
    if rotate[random_rotate] != None:
        image = image.transpose(rotate[random_rotate])
        if random_rotate == 0:
            x_max, x_min, y_max, y_min = (
                y_max,
                y_min,
                img_size - x_max,
                img_size - x_min,
            )
        elif random_rotate == 1:
            x_max, x_min, y_max, y_min = (
                img_size - x_max,
                img_size - x_min,
                img_size - y_max,
                img_size - y_min,
            )
        elif random_rotate == 2:
            x_max, x_min, y_max, y_min = (
                img_size - y_max,
                img_size - y_min,
                x_max,
                x_min,
            )
    if x_max < x_min:
        x_max, x_min = x_min, x_max
    if y_max < y_min:
        y_max, y_min = y_min, y_max
    return image, [x_min, y_min, x_max, y_max]
