import torch
from pytorch_lightning import LightningDataModule

from src.qdataset_for_two import QuaternionFixedTwoDataset
from data.dude_dataset import DudEDataset


class DudEDatasetLightning(LightningDataModule):
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
