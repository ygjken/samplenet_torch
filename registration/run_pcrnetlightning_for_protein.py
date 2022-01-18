import torch
import torchvision
from torch.utils.data import DataLoader
import open3d as o3d
import os

from pcrnet_lightning import PCRNetLightning
from data.modelnet_dataset import ModelNetCls
from data.dude_dataset import DudEDataset
from src.qdataset import QuaternionTransform
from src.qdataset import QuaternionFixedDataset
from src.qdataset_for_two import QuaternionFixedTwoDataset
from src.pctransforms import OnUnitCube, PointcloudToTensor
from src.torch2open3d import tensor2pc


if __name__ == "__main__":
    # choose model
    pretrain_checkpoint_name = "version_12/checkpoints/protein/pcrnet-epoch=195"

    # load model
    pretrain_checkpoint_path = os.path.join("lightning_logs/", pretrain_checkpoint_name) + ".ckpt"
    result_dir_path = os.path.join("pc_logs/", pretrain_checkpoint_name)

    # define model
    lightning_model = PCRNetLightning(input_shape="bnc")
    lightning_model = lightning_model.load_from_checkpoint(pretrain_checkpoint_path)
    model = lightning_model.model
    model.eval()

    # load data
    transforms = torchvision.transforms.Compose([PointcloudToTensor(), OnUnitCube()])
    dataset = DudEDataset(
        json_path="data/protein/dud.json",
        ply_path="data/protein/ply",
        does_transforms=True
    )
    dataset = QuaternionFixedTwoDataset(dataset, repeat=1, seed=0)

    for idx in [60]:
        source_tensor = torch.unsqueeze(dataset[idx][0], 0)
        target_tensor = torch.unsqueeze(dataset[idx][1], 0)

        # model forward
        twist, _ = model(source_tensor, target_tensor)

        # transform point-cloud by result getted from model
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

        # save
        if not os.path.exists(result_dir_path):
            os.makedirs(result_dir_path)
        o3d.io.write_point_cloud(os.path.join(result_dir_path, f"{idx}_question.ply"), q)
        o3d.io.write_point_cloud(os.path.join(result_dir_path, f"{idx}_answer.ply"), a)
