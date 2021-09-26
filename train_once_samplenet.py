# SampleNet train for reconstruction
# data use ModelNet40

from model.samplenet import SampleNet, SampleNetDecoder
from utils.dataloader import ModelNetCls
from torch.utils import data
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter

SAMPLED_SIZE = 256

writer = SummaryWriter("log/train_once_samplenet")

# data
modelnet40_train = ModelNetCls(
    num_points=1024,
    transforms=None,
    train=True
)


modelnet40_vali = ModelNetCls(
    num_points=1024,
    transforms=None,
    train=False
)

data_loaders = {'train':
                data.DataLoader(modelnet40_train, batch_size=32, shuffle=True),
                'val':
                data.DataLoader(modelnet40_vali, batch_size=16, shuffle=True)}

# models
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


# optimiter
downsampler_params = filter(
    lambda p: p.requires_grad, downsampler.parameters())
downsampler_optimizer = optim.Adam(downsampler_params, lr=1e-3)
upsampler_params = filter(lambda p: p.requires_grad, upsampler.parameters())
upsampler_optimizer = optim.Adam(upsampler_params, lr=1e-3)


# loss weight
alpha = 0.01
lmbda = 0.01

# train mode
downsampler.train()
upsampler.train()

# forward propagation
pc_pl = next(iter(data_loaders["train"]))[0].clone().detach().cuda()
print('Points:', pc_pl.shape)

# Forward Propagation
simp_pc, proj_pc, _ = downsampler(pc_pl)
pred = upsampler(proj_pc)

# Compute losses
simplification_loss = downsampler.get_simplification_loss(
    pc_pl, simp_pc, SAMPLED_SIZE
)
projection_loss = downsampler.get_projection_loss()
reconstrution_loss = upsampler.get_reconstruction_loss(
    pc_pl, pred)

loss = reconstrution_loss + \
    alpha * simplification_loss + lmbda * projection_loss

downsampler_optimizer.zero_grad()
upsampler_optimizer.zero_grad()
loss.backward()
downsampler_optimizer.step()
upsampler_optimizer.step()

writer.add_graph(downsampler, pc_pl)
writer.add_graph(upsampler, proj_pc)
print(loss)
