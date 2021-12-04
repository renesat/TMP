from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TVF
from torchvision.utils import draw_segmentation_masks


class TrashSegmentation(pl.LightningModule):
    def __init__(
        self,
        pretrained: Union[bool, str, Path] = True,
        n_classes: int = 2,
        n_unfreez_backbone: int = 3,
        freez: bool = True,
    ):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=pretrained,
        )
        self.model.classifier[4] = nn.Conv2d(
            256,
            n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        if freez:
            for p in list(self.model.backbone.parameters())[:-n_unfreez_backbone]:
                p.requires_grad = False
        # if pretrained:
        #     self.model.aux_classifier[4] = nn.Conv2d(
        #         10,
        #         1,
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     )

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        img = batch["image"]
        mask = batch["mask"]
        out = self.forward(img)
        # out = nn.Sigmoid()(out["out"]).squeeze(1)
        loss = nn.CrossEntropyLoss()(out.float(), mask)

        self.log("train/loss", loss)

        out_mask = out.argmax(dim=1)

        with torch.no_grad():
            if batch_idx == 0:
                mean = torch.Tensor([0.229, 0.224, 0.225])
                mean = mean.to(img.device).resize(1, 3, 1, 1)
                var = torch.Tensor([0.485, 0.456, 0.406])
                var = var.to(img.device).resize(1, 3, 1, 1)

                img = img * var + mean
                img[img > 1] = 1
                img[img < 0] = 0
                self.logger.experiment.add_image(
                    "train/img1",
                    draw_segmentation_masks(
                        (img[0] * 255).type(torch.ByteTensor),
                        out[0][1:].bool(),
                        alpha=0.8,
                    ),
                    self.current_epoch,
                )
                self.logger.experiment.add_image(
                    "train/img2",
                    draw_segmentation_masks(
                        (img[1] * 255).type(torch.ByteTensor),
                        out[1][1:].bool(),
                        alpha=0.8,
                    ),
                    self.current_epoch,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["image"]
        mask = batch["mask"]
        out = self.forward(img)
        out_mask = out.argmax(dim=1)
        # out = nn.Sigmoid()(out["out"]).squeeze(1)
        loss = nn.CrossEntropyLoss()(out.float(), mask)

        # out_result = (out > 0.5).bool()
        # iou = (out_result & mask.bool()).sum(dim=(1, 2)) / (
        #     (out_result | mask.bool()).sum(dim=(1, 2)) + 1e-8
        # )
        # dsc = (
        #     2
        #     * (out_result & mask.bool()).sum(dim=(1, 2))
        #     / (out_result.sum(dim=(1, 2)) + mask.sum(dim=(1, 2)))
        # )

        self.log("val/loss", loss)

        if batch_idx == 0:
            mean = torch.Tensor([0.229, 0.224, 0.225])
            mean = mean.to(img.device).resize(1, 3, 1, 1)
            var = torch.Tensor([0.485, 0.456, 0.406])
            var = var.to(img.device).resize(1, 3, 1, 1)

            img = img * var + mean
            img[img > 1] = 1
            img[img < 0] = 0
            self.logger.experiment.add_image(
                "val/img1",
                draw_segmentation_masks(
                    (img[0] * 255).type(torch.ByteTensor),
                    out[0][1:].bool(),
                    alpha=0.8,
                ),
                self.current_epoch,
            )
            self.logger.experiment.add_image(
                "val/img2",
                draw_segmentation_masks(
                    (img[1] * 255).type(torch.ByteTensor),
                    out[1][1:].bool(),
                    alpha=0.8,
                ),
                self.current_epoch,
            )

        return {
            "loss": loss,
            # "iou": iou,
            # "dsc": dsc,
        }

    def validation_epoch_end(self, epoches_output):
        print(
            "val/loss",
            np.mean([x["loss"].detach().cpu().item() for x in epoches_output]),
        )
        # mIoU = []
        # mDSC = []
        # for item in epoches_output:
        #     mIoU.extend(list(item["iou"].detach().cpu().numpy()))
        #     mDSC.extend(list(item["dsc"].detach().cpu().numpy()))
        # mIoU = np.mean(mIoU)
        # mDSC = np.mean(mDSC)
        # self.log("val/IoU", mIoU)
        # self.log("val/DSC", mDSC)

        # print(f"mIoU = {mIoU}")
        # print(f"mDSC = {mDSC}")
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
