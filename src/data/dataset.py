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

transform_2 = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize((CONFIG["img_size_h"], CONFIG["img_size_w"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


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


def crop(image, mask):
    width = random.randint(256, 512)
    height = int(width)
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(height, width))
    image = TVF.crop(image, i, j, h, w)
    mask = TVF.crop(mask, i, j, h, w)
    # print(mask.shape)
    image = transforms.Resize((512, 512))(image)
    mask = transforms.Resize((512, 512))(mask.unsqueeze(0)).squeeze(0)
    # print(type(image))
    # print(image.shape)
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


class TrashDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        train: bool = False,
        val_data=None,
        transform=None,
        to_rotate=False,
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
        self.to_rotate = to_rotate

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
        mask = torch.Tensor(mask)

        if self.to_rotate:
            image, mask = random_flip(image, mask)
            image, mask = crop(image, mask)

        return image, mask
