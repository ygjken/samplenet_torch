# SampleNet train for reconstruction
# data use ModelNet40

from model.samplenet import SampleNet, SampleNetDecoder
from utils.dataloader import ModelNetCls
from torch.utils import data
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter

SAMPLED_SIZE = 256

writer = SummaryWriter("log/down_and_up_sample")

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
encoder_params = filter(lambda p: p.requires_grad, downsampler.parameters())
encoder_optimizer = optim.Adam(encoder_params, lr=1e-3)
decoder_params = filter(lambda p: p.requires_grad, upsampler.parameters())
decoder_optimizer = optim.Adam(decoder_params, lr=1e-3)


# loss weight
alpha = 0.01
lmbda = 0.01


for epoch in range(300):

    for phase in ['train', 'val']:
        # Select mode
        if phase == 'train':
            downsampler.train()
            upsampler.train()
        else:
            downsampler.eval()
            upsampler.eval()

        for pc in data_loaders[phase]:
            pc_pl = pc[0].clone().detach().cuda()

            with torch.set_grad_enabled(phase == "train"):
                # Forward Propagation
                simp_pc, proj_pc, _ = downsampler(pc_pl)
                pred = upsampler(proj_pc)

                # Compute losses
                pc_pl = pc_pl.permute(0, 2, 1)
                simp_pc = simp_pc.permute(0, 2, 1)
                simplification_loss = downsampler.get_simplification_loss(
                    pc_pl, simp_pc, SAMPLED_SIZE
                )
                projection_loss = downsampler.get_projection_loss()

                pc_pl = pc_pl.permute(0, 2, 1)
                simp_pc = simp_pc.permute(0, 2, 1)
                reconstrution_loss = upsampler.get_reconstruction_loss(
                    pc_pl, pred)

                loss = reconstrution_loss + \
                    alpha * simplification_loss + lmbda * projection_loss

            if phase == 'train':
                # Backward + Optimize
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

        print(f"{epoch}, {phase}, {loss}", flush=True)
        writer.add_scalar(f'{phase}', loss, epoch)
        torch.save(downsampler.state_dict(), "weight/encoder.pt")
        torch.save(upsampler.state_dict(), "weight/decoder.pt")

writer.flush()
writer.close()
print("done!")
