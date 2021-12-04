import json
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TVF
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from .common import crop, random_flip

VAL_FILES = (
    "BATCH_s05_img_11050.jpg",
    "BATCH_s05_img_11130.jpg",
    "BATCH_s05_img_11210.jpg",
    "camera_img_0.jpg",
    "camera_img_1.jpg",
    "camera_img_2.jpg",
    "camera_img_3.jpg",
    "DJI_0009.JPG",
    "DJI_0010.JPG",
    "GOPR0019.JPG",
    "GOPR0020.JPG",
    "GOPR0021.JPG",
    "GOPR0022.JPG",
    "GOPR0023.JPG",
    "GOPR0024.JPG",
    "GOPR0025.JPG",
    "GOPR0026.JPG",
    "GOPR0027.JPG",
    "GOPR0028.JPG",
    "GOPR0029.JPG",
    "GOPR0030.JPG",
    "GOPR0031.JPG",
    "GOPR0032.JPG",
    "GOPR0033.JPG",
    "GOPR0034.JPG",
    "GOPR0035.JPG",
    "GOPR0036.JPG",
    "GOPR0037.JPG",
    "GOPR0038.JPG",
    "GOPR0039.JPG",
    "GOPR0040.JPG",
    "GOPR0041.JPG",
    "GOPR0042.JPG",
    "GOPR0043.JPG",
    "GOPR0044.JPG",
    "GOPR0045.JPG",
    "GOPR0046.JPG",
    "GOPR0047.JPG",
    "GOPR0048.JPG",
    "GOPR0049.JPG",
    "GOPR0050.JPG",
    "GOPR0051.JPG",
    "GOPR0052.JPG",
    "GOPR0053.JPG",
    "GOPR0054.JPG",
    "GOPR0055.JPG",
    "GOPR0056.JPG",
    "photo_2020-10-23_14-02-00.jpg",
    "photo_2020-10-23_14-02-01.jpg",
    "photo_2020-10-23_14-02-02.jpg",
    "photo_2020-10-23_14-02-03.jpg",
    "photo_2020-10-23_14-02-04.jpg",
    "photo_2020-10-23_14-02-05.jpg",
)


class UAVVaste(Dataset):
    def __init__(
        self,
        annotations_file: Union[str, Path],
        img_dir: Union[str, Path],
        train: bool = False,
        val_files: Optional[List[str]] = None,
        transform=None,
        to_rotate: bool = False,
    ):
        if val_files is None:
            val_files = list(VAL_FILES)
        self.train = train
        self.coco_set = COCO(str(annotations_file))
        self.ids = list(
            map(
                lambda x: x[0],
                filter(
                    lambda x: x[1]["file_name"] not in val_files
                    if self.train
                    else x[1]["file_name"] in val_files,
                    self.coco_set.imgs.items(),
                ),
            )
        )
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.to_rotate = to_rotate

    def __len__(self):
        return len(self.ids)

    def view_item(self, idx):
        img = self[idx]["image"]
        mask = self[idx]["mask"]
        img = (
            img * torch.Tensor([0.229, 0.224, 0.225]).to(img.device).resize(3, 1, 1)
            + torch.Tensor(
                [
                    0.485,
                    0.456,
                    0.406,
                ]
            )
            .to(img.device)
            .resize(3, 1, 1)
        )
        img[img > 1] = 1
        img[img < 0] = 0
        return TVF.to_pil_image(
            draw_segmentation_masks(
                (img * 255).type(torch.ByteTensor),
                torch.Tensor(mask).bool(),
                alpha=0.8,
            )
        )

    def __getitem__(self, idx):
        index = self.ids[idx]
        img_info = self.coco_set.imgs[index]
        img_annotations = self.coco_set.imgToAnns[index]
        img_path = self.img_dir / img_info["file_name"]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        _, h, w = image.shape
        mask = torch.zeros((h, w), dtype=torch.bool)
        for ann in img_annotations:
            mask |= (
                transforms.Resize(
                    (h, w),
                    interpolation=transforms.InterpolationMode.NEAREST,
                )(torch.Tensor(self.coco_set.annToMask(ann)).unsqueeze(0))
                .squeeze(0)
                .bool()
            )
        mask = torch.stack(
            (~mask, mask),
            dim=0,
        ).float()

        if self.to_rotate:
            image, mask = random_flip(image, mask)
            image, mask = crop(image, mask)

        return {
            "image": image,
            "mask": mask,
        }
