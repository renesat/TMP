import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
from torchvision.utils import draw_segmentation_masks

class Unet(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = nn.Sigmoid()(out).squeeze(1)
        loss = nn.BCELoss()(out.float(), y.float())

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = nn.Sigmoid()(out).squeeze(1)
        loss = nn.BCELoss()(out.float(), y.float())

        out_result = (out > 0.5).bool()
        iou = (out_result & y.bool()).sum(dim=(1, 2)) / (
            (out_result | y.bool()).sum(dim=(1, 2)) + 1e-8
        )
        dsc = 2 * (out_result & y.bool()).sum(dim=(1, 2)) / (out_result.sum() + y.sum())

        self.log("val/loss", loss)

        if batch_idx == 0:
            img = (
                x * torch.Tensor([0.229, 0.224, 0.225]).to(x.device).resize(1, 3, 1, 1)
                + torch.Tensor(
                    [
                        0.485,
                        0.456,
                        0.406,
                    ]
                ).to(x.device).resize(1, 3, 1, 1)
            )
            img[img > 1] = 1
            img[img < 0] = 0
            self.logger.experiment.add_image("val/img1", img[0], batch_idx)
            self.logger.experiment.add_image("val/img2", img[1], batch_idx)

        return {
            "loss": loss,
            "iou": iou,
            "dsc": dsc,
        }

    def validation_epoch_end(self, epoches_output):
        mIoU = []
        mDSC = []
        for item in epoches_output:
            mIoU.extend(list(item["iou"].detach().cpu().numpy()))
            mDSC.extend(list(item["dsc"].detach().cpu().numpy()))
        mIoU = np.mean(mIoU)
        mDSC = np.mean(mDSC)
        self.log("val/IoU", mIoU)
        self.log("val/DSC", mDSC)

        print(f"{mIoU=}")
        print(f"{mDSC=}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class UnetR34(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
        #self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #    in_channels=3, out_channels=1, init_features=32, pretrained=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = nn.Sigmoid()(out).squeeze(1)
        loss = nn.BCELoss()(out.float(), y.float())

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = nn.Sigmoid()(out).squeeze(1)
        loss = nn.BCELoss()(out.float(), y.float())

        out_result = (out > 0.5).bool()
        iou = (out_result & y.bool()).sum(dim=(1, 2)) / (
            (out_result | y.bool()).sum(dim=(1, 2)) + 1e-8
        )
        dsc = (
            2
            * (out_result & y.bool()).sum(dim=(1, 2))
            / (out_result.sum(dim=(1, 2)) + y.sum(dim=(1, 2)))
        )

        self.log("val/loss", loss)

        if batch_idx == 0:
            img = (
                x * torch.Tensor([0.229, 0.224, 0.225]).to(x.device).resize(1, 3, 1, 1)
                + torch.Tensor(
                    [
                        0.485,
                        0.456,
                        0.406,
                    ]
                )
                .to(x.device)
                .resize(1, 3, 1, 1)
            )
            img[img > 1] = 1
            img[img < 0] = 0
            self.logger.experiment.add_image(
                "val/img1",
                draw_segmentation_masks(
                    (img[0] * 255).type(torch.ByteTensor),
                    out_result[0],
                    alpha=0.8,
                ),
                batch_idx,
            )
            self.logger.experiment.add_image(
                "val/img2",
                draw_segmentation_masks(
                    (img[1] * 255).type(torch.ByteTensor),
                    out_result[1],
                    alpha=0.8,
                ),
                batch_idx,
            )

        return {
            "loss": loss,
            "iou": iou,
            "dsc": dsc,
        }

    def validation_epoch_end(self, epoches_output):
        mIoU = []
        mDSC = []
        for item in epoches_output:
            mIoU.extend(list(item["iou"].detach().cpu().numpy()))
            mDSC.extend(list(item["dsc"].detach().cpu().numpy()))
        mIoU = np.mean(mIoU)
        mDSC = np.mean(mDSC)
        self.log("val/IoU", mIoU)
        self.log("val/DSC", mDSC)

        print(f"{mIoU=}")
        print(f"{mDSC=}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
class UnetR18(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = smp.Unet("resnet18", encoder_weights="imagenet", activation=None)
        #self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #    in_channels=3, out_channels=1, init_features=32, pretrained=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = nn.Sigmoid()(out).squeeze(1)
        loss = nn.BCELoss()(out.float(), y.float())

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = nn.Sigmoid()(out).squeeze(1)
        loss = nn.BCELoss()(out.float(), y.float())

        out_result = (out > 0.5).bool()
        iou = (out_result & y.bool()).sum(dim=(1, 2)) / (
            (out_result | y.bool()).sum(dim=(1, 2)) + 1e-8
        )
        dsc = (
            2
            * (out_result & y.bool()).sum(dim=(1, 2))
            / (out_result.sum(dim=(1, 2)) + y.sum(dim=(1, 2)))
        )

        self.log("val/loss", loss)

        if batch_idx == 0:
            img = (
                x * torch.Tensor([0.229, 0.224, 0.225]).to(x.device).resize(1, 3, 1, 1)
                + torch.Tensor(
                    [
                        0.485,
                        0.456,
                        0.406,
                    ]
                )
                .to(x.device)
                .resize(1, 3, 1, 1)
            )
            img[img > 1] = 1
            img[img < 0] = 0
            self.logger.experiment.add_image(
                "val/img1",
                draw_segmentation_masks(
                    (img[0] * 255).type(torch.ByteTensor),
                    out_result[0],
                    alpha=0.8,
                ),
                self.current_epoch,
            )
            self.logger.experiment.add_image(
                "val/img2",
                draw_segmentation_masks(
                    (img[1] * 255).type(torch.ByteTensor),
                    out_result[1],
                    alpha=0.8,
                ),
                self.current_epoch,
            )

        return {
            "loss": loss,
            "iou": iou,
            "dsc": dsc,
        }

    def validation_epoch_end(self, epoches_output):
        mIoU = []
        mDSC = []
        for item in epoches_output:
            mIoU.extend(list(item["iou"].detach().cpu().numpy()))
            mDSC.extend(list(item["dsc"].detach().cpu().numpy()))
        mIoU = np.mean(mIoU)
        mDSC = np.mean(mDSC)
        self.log("val/IoU", mIoU)
        self.log("val/DSC", mDSC)

        print(f"{mIoU=}")
        print(f"{mDSC=}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
   
