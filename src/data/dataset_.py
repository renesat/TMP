import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TVF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.utils import draw_segmentation_masks


class TrashDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        train: bool = False,
        val_data=None,
        transform=None,
    ):
        if val_data is None:
            val_data = ["0089", "0063", "0049", "0102"]

        self.train = train
        self.val_data = val_data
        self.annotations = pd.read_csv(annotations_file, sep=",")
        self.annotations["filename"] = (
            self.annotations["INPUT:image"].dropna().apply(lambda x: x.split("/")[-1])
        )
        drop_raws = []
        for i in range(len(self.annotations)):
            video = self.annotations["filename"][i][:8]
            # print(video)
            if video in [f"DJI_{v}" for v in val_data]:
                if train:
                    drop_raws.append(i)
            else:
                if not train:
                    drop_raws.append(i)
        self.annotations = self.annotations.drop(list(set(drop_raws)))

        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return self.annotations.shape[0]

    def view_item(self, idx):
        img, mask = self[idx]
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

    @staticmethod
    def region_to_mask(region, image):
        _, height, width = image.shape
        polygon = []
        for poly in json.loads("[" + region.replace("\\", "") + "]"):
            for pair in poly["points"]:
                polygon.append((pair["top"] * height, pair["left"] * width))
        img = Image.new("L", image.shape[1:], 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img).T
        # mask = mask.reshape((3, image.shape[1], image))
        return mask

    def __getitem__(self, idx):
        img_path = self.img_dir / self.annotations["filename"].iloc[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        mask = self.region_to_mask(
            self.annotations["OUTPUT:path"].iloc[idx],
            image,
        )
        return image, mask
