# SampleNet train for reconstruction
# data use ModelNet40

from model.samplenet import SampleNet, SampleNetDecoder
from utils.dataloader import ModelNetCls
from torch.utils import data
from torch import optim
import torch

SAMPLED_SIZE = 256

# data
modelnet40 = ModelNetCls(
    num_points=1024,
    transforms=None,
    train=True
)

train_loader = data.DataLoader(modelnet40, batch_size=32, shuffle=True)

# models
encoder = SampleNet(
    num_out_points=SAMPLED_SIZE,
    bottleneck_size=128,
    group_size=10,
    initial_temperature=0.1,
    complete_fps=True,
    input_shape="bnc",
    skip_projection=True
)

decoder = SampleNetDecoder(
    num_sampled_points=SAMPLED_SIZE,
    bottleneck_size=128,
    num_reconstracted_points=1024,
)

encoder.cuda()
encoder.train()
decoder.cuda()
decoder.train()

# optimiter
encoder_params = filter(lambda p: p.requires_grad, encoder.parameters())
encoder_optimizer = optim.Adam(encoder_params, lr=1e-3)
decoder_params = filter(lambda p: p.requires_grad, decoder.parameters())
decoder_optimizer = optim.Adam(decoder_params, lr=1e-3)


# loss weight
alpha = 0.01
lmbda = 0.01


for epoch in range(300):
    for pc in train_loader:
        pc_pl = pc[0].clone().detach().cuda()

        # train
        simp_pc, proj_pc, _ = encoder(pc_pl)
        pred = decoder(proj_pc)

        # Compute losses
        simplification_loss = encoder.get_simplification_loss(
            pc_pl, simp_pc, 64
        )
        projection_loss = encoder.get_projection_loss()
        reconstrution_loss = decoder.get_reconstruction_loss(pc_pl, pred)

        loss = reconstrution_loss + \
            alpha * simplification_loss + lmbda * projection_loss

        # Backward + Optimize
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    print(f"epoch: {epoch}, {loss}", flush=True)
    torch.save(encoder.state_dict(), "weight/encoder.pt")
    torch.save(decoder.state_dict(), "weight/decoder.pt")

print("done!")
