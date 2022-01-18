import torch
from torch.utils.data import DataLoader
import open3d as o3d
import torchvision
# from src.pctransforms import OnUnitCube, PointcloudToTensor
from src.pctransforms_for_two import OnUnitCubeForTwo, PointcloudToTensorForTwo, TransformsForTwo

from models import pcrnet
from data.dude_dataset import DudEDataset
from src.qdataset_for_two import QuaternionFixedTwoDataset, QuaternionTransform
from src.torch2open3d import tensor2pc


model = pcrnet.PCRNet(input_shape="bnc")
model.load_state_dict(
    torch.load('log/baseline/PCRNet1024_model_best.pth', map_location="cpu")
)
model.eval()

testdata = DudEDataset(json_path='data/protein/dud.json',
                       ply_path='data/protein/ply',
                       does_transforms=True,
                       include_shape=True,)
rotated_dataset = QuaternionFixedTwoDataset(
    testdata,
    repeat=1,
    seed=1,
    include_shapes=True
)
dataloader = DataLoader(rotated_dataset, batch_size=1, shuffle=False)

i = 0
for data_and_shape in dataloader:
    data = data_and_shape[0:3]
    shape = data_and_shape[3]

    ligand, pocket, igt = data

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

    o3d.io.write_point_cloud(f'log/protein_visible/{i}_ques.ply', q)
    o3d.io.write_point_cloud(f'log/protein_visible/{i}_ans.ply', a)

    i += 1
