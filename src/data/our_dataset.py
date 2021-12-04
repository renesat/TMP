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


class OurDataset(Dataset):
    def __init__(
        self,
        img_dir: Union[str, Path],
        train: bool = False,
        val_files: Optional[List[str]] = None,
        transform=None,
        to_rotate: bool = False,
    ):
        self.classes = ["net", "wood", "plastic", "metall"]
        # if val_files is None:
        #     val_files = list(VAL_FILES)
        self.img_dir = Path(img_dir)
        self.train = train
        self.ids = list(
            map(
                lambda x: "_".join(str(x.stem).split("_")[:-1]),
                self.img_dir.glob("*_image.*"),
            )
        )
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.to_rotate = to_rotate

    def __len__(self):
        return len(self.ids)

    def view_item(self, idx):
        item = self[idx]
        img = item["image"]
        mask = item["mask"]
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
                torch.Tensor(mask[1:]).bool(),
                alpha=1.0,
            )
        )

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.img_dir / f"{img_id}_image.JPG"
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        _, h, w = image.shape
        mask = torch.zeros((len(self.classes) + 1, h, w), dtype=torch.bool)
        mask[0] = torch.ones((h, w), dtype=torch.bool)
        for i, cls in enumerate(self.classes):
            mask_img_path = Path(self.img_dir / f"{img_id}_{cls}.png")
            if not mask_img_path.is_file():
                continue
            mask_image = Image.open(mask_img_path)
            item_mask = transforms.ToTensor()(mask_image)[3] > 0
            item_mask = (
                transforms.Resize(
                    (h, w),
                    interpolation=transforms.InterpolationMode.NEAREST,
                )(item_mask.unsqueeze(0))
                .squeeze(0)
                .bool()
            )
            mask[i + 1] = item_mask
            mask[0] = item_mask & ~item_mask
        mask = mask.float()

        if self.to_rotate:
            image, mask = random_flip(image, mask)
            image, mask = crop(image, mask)

        return {
            "image": image,
            "mask": mask,  # .argmax(dim=1),
        }
