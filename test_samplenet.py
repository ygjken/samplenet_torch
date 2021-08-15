import torch
import numpy as np
from utils.convertor import xyz2pcd

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
pc_pl = torch.from_numpy(modelnet40_train[0][0])
print('Points:', modelnet40_train[0][0].shape)

pc_pl = torch.unsqueeze(pc_pl, 0).clone().detach().cuda()
simp_pc, proj_pc, _ = downsampler(pc_pl)
print('simp_pc', simp_pc.shape)
print('proj_pc', proj_pc.shape)

simp_pc = xyz2pcd(simp_pc[0])
print(simp_pc)
