import torch
import open3d as o3d
from utils.convertor import xyz2pcd, get_color

from model.samplenet import SampleNet, SampleNetDecoder
from utils.dataloader import ModelNetCls

# data
SAMPLED_SIZE = 256
modelnet40_train = ModelNetCls(
    num_points=1024,
    transforms=None,
    train=True
)

# model
downsampler = SampleNet(
    num_out_points=SAMPLED_SIZE,
    bottleneck_size=128,
    group_size=10,
    initial_temperature=0.1,
    complete_fps=True,
    input_shape="bnc",
    skip_projection=True
)

upsampler = SampleNetDecoder(
    num_sampled_points=SAMPLED_SIZE,
    bottleneck_size=128,
    num_reconstracted_points=1024,
)

downsampler.cuda()
upsampler.cuda()
downsampler.eval()
upsampler.eval()
downsampler.load_state_dict(torch.load('weight/encoder.pt'))

# forward propagation
pc_pl = torch.from_numpy(modelnet40_train[10][0])
print('Points:', modelnet40_train[10][0].shape)

pc_pl = torch.unsqueeze(pc_pl, 0).clone().detach().cuda()
simp_pc, proj_pc, _ = downsampler(pc_pl)
print('simp_pc', simp_pc.shape)
print('proj_pc', proj_pc.shape)

reconst_pc = upsampler(proj_pc)
print('reconst_pc', reconst_pc.shape)


# save as ply files
cloud_unsampled = xyz2pcd(pc_pl, 'bnc', batched=True)
cloud_unsampled.paint_uniform_color(get_color('blue'))
o3d.io.write_point_cloud('log/pc_out/cloud_unsampled.ply', cloud_unsampled)

cloud_sampled = xyz2pcd(simp_pc, 'bcn', batched=True)
cloud_sampled.paint_uniform_color(get_color('blue'))
o3d.io.write_point_cloud('log/pc_out/cloud_sampled.ply', cloud_sampled)

cloud_softproj = xyz2pcd(proj_pc, 'bcn', batched=True)
cloud_softproj.paint_uniform_color(get_color('blue'))
o3d.io.write_point_cloud('log/pc_out/cloud_softproj.ply', cloud_softproj)

cloud_reconst = xyz2pcd(reconst_pc, 'bnc', batched=True)
cloud_reconst.paint_uniform_color(get_color('blue'))
o3d.io.write_point_cloud('log/pc_out/cloud_reconst.ply', cloud_reconst)
