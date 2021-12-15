import argparse
import logging
import os
import sys

import numpy as np
from pytorch_lightning import callbacks
import torch
import torchvision
from tqdm import tqdm
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from main import train

from models.pcrnet import PCRNet
from data.modelnet_loader_torch import ModelNetCls
from models import pcrnet
from src import ChamferDistance, FPSSampler, RandomSampler, SampleNet
from src import sputils
from src.pctransforms import OnUnitCube, PointcloudToTensor
from src.qdataset import QuaternionFixedDataset, QuaternionTransform, rad_to_deg

from data.protein_loader_torch import DudEDataset
from src.qdataset_for_two import QuaternionFixedTwoDataset


class PCRNetLightning(LightningModule):
    def __init__(self, bottleneck_size=1024, input_shape="bcn") -> None:
        super().__init__()
        self.model = PCRNet(bottleneck_size, input_shape)  # TODO: bncで入れてる
        self.loss_func = self.compute_loss

    def compute_loss(self, p0, p1, igt, twist, pre_normalized_quat):
        # https://arxiv.org/pdf/1805.06485.pdf QuaterNet quaternient regularization loss
        qnorm_loss = torch.mean((torch.sum(pre_normalized_quat ** 2, dim=1) - 1) ** 2)

        est_transform = QuaternionTransform(twist)
        gt_transform = QuaternionTransform.from_dict(igt)

        p1_est = est_transform.rotate(p0)

        cost_p0_p1, cost_p1_p0 = ChamferDistance()(p1, p1_est)
        cost_p0_p1 = torch.mean(cost_p0_p1)
        cost_p1_p0 = torch.mean(cost_p1_p0)

        chamfer_loss = cost_p0_p1 + cost_p1_p0

        rot_err, norm_err, trans_err = est_transform.compute_errors(gt_transform)

        # if self.LOSS_TYPE == 0:
        pcrnet_loss = 1.0 * norm_err + 1.0 * chamfer_loss

        # elif self.LOSS_TYPE == 1:
        #     pcrnet_loss = chamfer_loss

        rot_err = rad_to_deg(rot_err)

        pcrnet_loss_info = {
            "chamfer_loss": chamfer_loss,
            "qnorm_loss": qnorm_loss,
            "rot_err": rot_err,
            "norm_err": norm_err,
            "trans_err": trans_err,
            # "est_transform": est_transform,
        }

        return pcrnet_loss, pcrnet_loss_info

    def forward(self, x0, x1):
        return self.model(x0, x1)

    def training_step(self, batch, batch_index):
        p0, p1, igt = batch
        twist, pre_normalized_quat = self(p0, p1)
        loss, loss_info = self.loss_func(p0, p1, igt, twist, pre_normalized_quat)

        train_logs = {
            f"train/{key}": val for key, val in loss_info.items()
        }
        self.log_dict(train_logs, on_step=False, on_epoch=True)
        self.log("train/total_loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_index):
        p0, p1, igt = batch
        twist, pre_normalized_quat = self(p0, p1)
        loss, loss_info = self.loss_func(p0, p1, igt, twist, pre_normalized_quat)

        # train_logs = {
        #     f"val/{key}": val for key, val in loss_info.items()
        # }
        # self.log_dict(train_logs, on_step=False, on_epoch=True)
        # self.log("val/total_loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "loss_info": loss_info}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        #  TODO: loss info も追加する
        self.log("val/total_loss", avg_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        learnable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.Adam(learnable_params, lr=1e-3)


class DatasetLightning(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        traindata = DudEDataset(
            json_path="data/protein/dud.json",
            ply_path="data/protein/ply",
            does_transforms=True
        )
        testdata = DudEDataset(
            json_path="data/protein/dud.json",
            ply_path="data/protein/ply",
            does_transforms=True
        )
        train_repeats = max(int(5000 / len(traindata)), 1)

        self.trainset = QuaternionFixedTwoDataset(traindata, repeat=train_repeats, seed=0)
        self.testset = QuaternionFixedTwoDataset(testdata, repeat=1, seed=0)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False)


def main():
    data = DatasetLightning()
    checkpoint = ModelCheckpoint(
        monitor="val/total_loss",
        filename="log/lightning/pcrnet-{epoch:02d}",
        save_top_k=3,
        mode="min")

    model = PCRNetLightning(input_shape="bnc")

    trainer = Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint])

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
