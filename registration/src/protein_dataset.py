import open3d as o3d
import numpy as np
import os
import json

from torch.utils.data import Dataset


class DudEDataset(Dataset):
    """Torch向けDUD-Eデータセット
    """

    def __init__(self, json_path, ply_path, transforms) -> None:
        super().__init__()
        self.transforms = transforms

        json_file = open(json_path, 'r')
        self.name_sets = json.load(json_file)
        self.prefix = ply_path

    def __len__(self):
        return len(self.name_sets)

    def __getitem__(self, idx):
        pocket = o3d.io.read_point_cloud(os.path.join(self.prefix, self.name_sets[idx]['target']) + '.ply')
        ligand = o3d.io.read_point_cloud(os.path.join(self.prefix, self.name_sets[idx]['ligand']) + '.ply')
        pocket_len = len(np.asarray(pocket.points))
        ligand_len = len(np.asarray(ligand.points))
        # len_min = min(pocket_len, ligand_len)
        pocket = pocket.random_down_sample(1024.1 / pocket_len)  # 全ての点群の点数を1024に統一する
        ligand = ligand.random_down_sample(1024.1 / ligand_len)
        pocket = np.asarray(pocket.points)
        ligand = np.asarray(ligand.points)

        if self.transforms is not None:
            pocket = self.transforms(pocket)
            ligand = self.transforms(ligand)

        return ligand, pocket
