import torch
import torchvision
from torch.utils.data import DataLoader
import open3d as o3d
import os

from pcrnet_lightning import PCRNetLightning
from data.modelnet_loader_torch import ModelNetCls
from src.qdataset import QuaternionTransform
from src.qdataset import QuaternionFixedDataset
from src.pctransforms import OnUnitCube, PointcloudToTensor
from src.torch2open3d import tensor2pc


if __name__ == "__main__":
    pretrain_checkpoint_name = "version_10/checkpoints/modelnet/pcrnet-epoch=498"
    pretrain_checkpoint_path = os.path.join("lightning_logs/", pretrain_checkpoint_name) + ".ckpt"
    result_dir_path = os.path.join("pc_logs/", pretrain_checkpoint_name)

    lightning_model = PCRNetLightning(input_shape="bnc")
    lightning_model = lightning_model.load_from_checkpoint(pretrain_checkpoint_path)
    model = lightning_model.model
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
    dataset = QuaternionFixedDataset(dataset, repeat=1, seed=0)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    source_tensor = torch.unsqueeze(dataset[0][0], 0)
    target_tensor = torch.unsqueeze(dataset[0][1], 0)

    twist, _ = model(source_tensor, target_tensor)

    est_transform = QuaternionTransform(twist)
    source_tensor_est = est_transform.rotate(source_tensor)
    source = tensor2pc(source_tensor[0])
    target = tensor2pc(target_tensor[0])
    est = tensor2pc(source_tensor_est[0])
    source.paint_uniform_color([0, 0.651, 0.929])  # yellow
    target.paint_uniform_color([1, 0.706, 0])  # blue
    est.paint_uniform_color([0, 0.651, 0.929])  # yellow
    q = source + target
    a = est + target

    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)
    o3d.io.write_point_cloud(os.path.join(result_dir_path, "question.ply"), q)
    o3d.io.write_point_cloud(os.path.join(result_dir_path, "answer.ply"), a)
