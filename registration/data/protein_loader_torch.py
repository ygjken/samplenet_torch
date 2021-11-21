import open3d as o3d
import numpy as np
import os
import json

from torch.utils.data import Dataset


class DudEDataset(Dataset):
    """Torch向けDUD-Eデータセット
    """

    def __init__(self, json_path, ply_path, transforms, include_shape=False) -> None:
        super().__init__()
        self.transforms = transforms

        self.include_shape = include_shape
        self.shape = []

        json_file = open(json_path, 'r')
        self.name_sets = json.load(json_file)
        self.prefix = ply_path

        self.pocket_list = []
        self.target_list = []
        for i in range(len(self.name_sets)):
            pocket = o3d.io.read_point_cloud(os.path.join(self.prefix, self.name_sets[i]["target"]) + '.ply')
            ligand = o3d.io.read_point_cloud(os.path.join(self.prefix, self.name_sets[i]["ligand"]) + '.ply')
            pocket_len = len(np.asarray(pocket.points))
            ligand_len = len(np.asarray(ligand.points))
            pocket = pocket.random_down_sample(1024.1 / pocket_len)  # 全ての点群の点数を1024に統一する
            ligand = ligand.random_down_sample(1024.1 / ligand_len)
            pocket = np.asarray(pocket.points)
            ligand = np.asarray(ligand.points)

            if self.transforms is not None:
                pocket = self.transforms(pocket)
                ligand = self.transforms(ligand)

            self.pocket_list.append(pocket)
            self.target_list.append(ligand)

            self.shape.append(len(ligand))

    def __len__(self):
        return len(self.name_sets)

    def __getitem__(self, idx):
        if not self.include_shape:
            return self.target_list[idx], self.pocket_list[idx], self.shape[idx]
        return self.target_list[idx], self.pocket_list[idx]


if __name__ == '__main__':
    d = DudEDataset("data/protein/dud.json", "data/protein/ply", None)
    for p, l in d:
        print(p.shape, l.shape)
