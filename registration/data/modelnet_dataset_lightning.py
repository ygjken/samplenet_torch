import torch
import torchvision
from pytorch_lightning import LightningDataModule

from data.modelnet_dataset import ModelNetCls
from src.pctransforms import OnUnitCube, PointcloudToTensor
from src.qdataset import QuaternionFixedDataset


class ModelNetDatasetLightning(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        transforms = torchvision.transforms.Compose([PointcloudToTensor(), OnUnitCube()])
        traindata = ModelNetCls(
            1024,
            transforms=transforms,
            train=True,
            download=False,
            folder="modelnet40_ply_hdf5_2048",
        )
        testdata = ModelNetCls(
            1024,
            transforms=transforms,
            train=False,
            download=False,
            folder="modelnet40_ply_hdf5_2048",
        )

        train_repeats = max(int(5000 / len(traindata)), 1)

        self.trainset = QuaternionFixedDataset(traindata, repeat=train_repeats, seed=0,)
        self.testset = QuaternionFixedDataset(testdata, repeat=1, seed=0)

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
