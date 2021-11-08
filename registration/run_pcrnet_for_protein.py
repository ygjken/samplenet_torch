import torch
from torch.utils.data import DataLoader
import open3d as o3d

from models import pcrnet
from src.protein_dataset import DudEDataset
from src.qdataset import QuaternionTransform
from torch2open3d import tensor2pc


model = pcrnet.PCRNet(input_shape="bnc")
model.load_state_dict(torch.load('log/baseline/PCRNet1024_model_best.pth', map_location="cpu"))
model.eval()

dataset = DudEDataset(json_path='data/protein/dud.json', ply_path='data/protein/ply', transforms=None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

i = 0
for ligand, pocket in dataloader:
    twist, _ = model(ligand.float(), pocket.float())

    est_transform = QuaternionTransform(twist)
    ligand_est = est_transform.rotate(ligand.float())

    source = tensor2pc(ligand[0])
    target = tensor2pc(pocket[0])
    est = tensor2pc(ligand_est[0])

    source.paint_uniform_color([0, 0.651, 0.929])  # yellow
    target.paint_uniform_color([1, 0.706, 0])  # blue
    est.paint_uniform_color([0, 0.651, 0.929])  # yellow
    q = source + target
    a = est + target

    o3d.io.write_point_cloud('log/pc/q.ply', q)
    o3d.io.write_point_cloud('log/pc/a.ply', a)

    if i == 0:
        break
    i += 1
