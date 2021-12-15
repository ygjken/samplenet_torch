import open3d as o3d
import numpy as np
import os
import json
import torch

from torch.utils.data import Dataset


class DudEDataset(Dataset):
    """Torch向けDUD-Eデータセット
    """

    def __init__(self, json_path, ply_path, does_transforms=True, include_shape=False) -> None:
        super().__init__()
        self.does_transforms = does_transforms

        self.include_shape = include_shape
        self.shapes = []

        json_file = open(json_path, 'r')
        self.name_sets = json.load(json_file)
        self.prefix = ply_path

        self.pocket_list = []
        self.ligand_list = []
        for i in range(len(self.name_sets)):
            pocket = o3d.io.read_point_cloud(os.path.join(self.prefix, self.name_sets[i]["target"]) + '.ply')
            ligand = o3d.io.read_point_cloud(os.path.join(self.prefix, self.name_sets[i]["ligand"]) + '.ply')
            pocket_len = len(np.asarray(pocket.points))
            ligand_len = len(np.asarray(ligand.points))
            pocket = pocket.random_down_sample(1024.1 / pocket_len)  # 全ての点群の点数を1024に統一する
            ligand = ligand.random_down_sample(1024.1 / ligand_len)
            pocket = np.asarray(pocket.points)
            ligand = np.asarray(ligand.points)

            if self.does_transforms:
                ligand, pocket = self.transforms(ligand, pocket)
            elif not self.does_transforms:
                pocket = torch.from_numpy(pocket.astype(np.float32)).clone()
                ligand = torch.from_numpy(ligand.astype(np.float32)).clone()

            self.pocket_list.append(pocket)
            self.ligand_list.append(ligand)

            self.shapes.append(self.name_sets[i]["target"] + '_' + self.name_sets[i]["ligand"])

    def __len__(self):
        return len(self.name_sets)

    def __getitem__(self, idx):
        if self.include_shape:
            return self.ligand_list[idx], self.pocket_list[idx], self.shapes[idx]
        return self.ligand_list[idx], self.pocket_list[idx]

    def transforms(self, src, tgt):
        src, tgt = self.__pointcloud_to_tensor(src, tgt)
        src, tgt = self.__on_unit_cube_for_two(src, tgt)
        return src, tgt

    def __pointcloud_to_tensor(self, src, tgt):
        return torch.from_numpy(src).float(), torch.from_numpy(tgt).float()

    def __on_unit_cube_for_two(self, source, target):
        all_points = torch.cat((source, target), 0)

        c = torch.max(all_points, dim=0)[0] - torch.min(all_points, dim=0)[0]
        s = torch.max(c)

        src = source / s
        tgt = target / s

        return src - src.mean(dim=0, keepdim=True), tgt - tgt.mean(dim=0, keepdim=True)


if __name__ == '__main__':
    d = DudEDataset("data/protein/dud.json", "data/protein/ply", None)
    for p, l in d:
        print(p.shape, l.shape)
