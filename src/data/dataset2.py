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


class TrashDataset2(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        train: bool = False,
        val_data=None,
        transform=None,
        to_rotate=False,
    ):
        self.N = 4
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
        self.annotations = self.annotations.sample(frac=1)

        self.img_dir = Path(img_dir)
        self.transform = transform
        self.to_rotate = to_rotate

    def __len__(self):
        return self.annotations.shape[0] // self.N

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

    @staticmethod
    def rle_to_mask(rle, image):
        _, height, width = image.shape
        shape = (height, width)
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

    def __getitem__(self, idx):
        indexes = np.random.choice(range(self.annotations.shape[0]), size=self.N)
        images = []
        for i in range(self.N):
            img_path = self.img_dir / self.annotations["filename"].iloc[indexes[i]]
            image = Image.open(img_path)
            images.append(image)
        image = Image.new("RGB", (512 + 512, 512 + 512))
        image.paste(images[0], (0, 0))
        image.paste(images[1], (0, 512))
        image.paste(images[2], (512, 0))
        image.paste(images[3], (512, 512))

        if self.transform:
            for i in range(self.N):
                images[i] = self.transform(images[i])
            image = self.transform(image)
        masks = []
        for i in range(self.N):
            mask = self.region_to_mask(
                self.annotations["OUTPUT:path"].iloc[indexes[i]],
                images[i],
            )
            masks.append(mask)
        mask = np.concatenate(
            (
                np.concatenate(
                    (masks[0], masks[1]),
                    axis=0,
                ),
                np.concatenate(
                    (masks[2], masks[3]),
                    axis=0,
                ),
            ),
            axis=1,
        )

        mask = torch.Tensor(mask)

        # image = transforms.Resize((512, 512))(image)
        # mask = transforms.Resize((512, 512))(mask.unsqueeze(0)).squeeze(0)

        if self.to_rotate:
            image, mask = random_flip(image, mask)
            image, mask = crop(image, mask)

        return image, mask
