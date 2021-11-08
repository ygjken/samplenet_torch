import torch
from torch.utils.data import DataLoader
import open3d as o3d

from models import pcrnet
from src.qdataset import QuaternionTransform
from torch2open3d import tensor2pc
import torchvision

from data.modelnet_loader_torch import ModelNetCls
from src.pctransforms import OnUnitCube, PointcloudToTensor


model = pcrnet.PCRNet(input_shape="bnc")
model.load_state_dict(torch.load('log/baseline/PCRNet1024_model_best.pth', map_location="cpu"))
model.eval()

transforms = torchvision.transforms.Compose([PointcloudToTensor(), OnUnitCube()])
dataset = ModelNetCls(
    1024,
    transforms=transforms,
    train=False,
    download=False,
    cinfo=None,
    folder='car_hdf5_2048',
    include_shapes=True,
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

ligand = torch.unsqueeze(dataset[0][0], 0)
pocket = torch.unsqueeze(dataset[39][0], 0)

twist, _ = model(ligand, pocket)

est_transform = QuaternionTransform(twist)
ligand_est = est_transform.rotate(ligand)

source = tensor2pc(ligand[0])
target = tensor2pc(pocket[0])
est = tensor2pc(ligand_est[0])

source.paint_uniform_color([0, 0.651, 0.929])  # yellow
target.paint_uniform_color([1, 0.706, 0])  # blue
est.paint_uniform_color([0, 0.651, 0.929])  # yellow
q = source + target
a = est + target

o3d.io.write_point_cloud('log/pc/car/q.ply', q)
o3d.io.write_point_cloud('log/pc/car/a.ply', a)
