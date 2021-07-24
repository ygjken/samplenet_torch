# SampleNet train for reconstruction
# data use ModelNet40

import torch
from samplenet import SampleNet, SampleNetDecoder
from dataloader import ModelNetCls
from torch.utils import data

modelnet40 = ModelNetCls(
    num_points=1024,
    transforms=None,
    train=True
)

train_loader = data.DataLoader(modelnet40, batch_size=32, shuffle=True)

encoder = SampleNet(
    num_out_points=64,
    bottleneck_size=128,
    group_size=10,
    initial_temperature=0.1,
    complete_fps=True,
    input_shape="bnc",
    skip_projection=True
)

decoder = SampleNetDecoder(
    num_sampled_points=64,
    bottleneck_size=128,
    num_reconstracted_points=1028,
)

encoder.cuda()
encoder.train()

for epoch in range(2):
    for pc in train_loader:
        pc_pl = torch.tensor(pc[0], dtype=torch.float32).cuda()
        encoder(pc_pl)

print("done")
