# SampleNet train for reconstruction
# data use ModelNet40

from torch.autograd.grad_mode import set_grad_enabled
from samplenet import SampleNet, SampleNetDecoder
from dataloader import ModelNetCls
from torch.utils import data
from torch import optim
import torch

SAMPLED_SIZE = 256


class AutoDcoder(torch.nn.Module):
    def __init__(self, sampled_size, bottleneck_size):
        super().__init__()
        self.sampled_size = sampled_size
        self.bottleneck_size = bottleneck_size

        self.encoder = SampleNet(
            num_out_points=sampled_size,
            bottleneck_size=bottleneck_size,
            group_size=10,
            initial_temperature=0.1,
            complete_fps=True,
            input_shape="bnc",
            skip_projection=False
        )

        self.decoder = SampleNetDecoder(
            num_sampled_points=sampled_size,
            bottleneck_size=bottleneck_size,
            num_reconstracted_points=1028,
        )

    def forward(self, x):
        simp_pc, proj_pc, match_pc = self.encoder(x)
        pred_x = self.decoder(proj_pc)
        return simp_pc, proj_pc, match_pc, pred_x

    def get_simplification_loss(self, pc_pl, simp_pc):
        return self.encoder.get_simplification_loss(pc_pl, simp_pc, self.sampled_size)

    def get_projection_loss(self):
        return self.encoder.get_projection_loss()

    def get_reconstruction_loss(self, pc_pl, pred):
        return self.decoder.get_reconstruction_loss(pc_pl, pred)


if __name__ == "__main__":
    # data
    modelnet40 = ModelNetCls(
        num_points=1024,
        transforms=None,
        train=True
    )

    train_loader = data.DataLoader(
        modelnet40, batch_size=32, shuffle=True)

    model = AutoDcoder(sampled_size=256, bottleneck_size=128)
    model.cuda()
    model.train()

    # optimiter
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    # loss weight
    alpha = 0.01
    lmbda = 0.01

    for epoch in range(200):
        for pc in train_loader:
            pc_pl = pc[0].clone().detach().cuda()

            # train
            simp_pc, proj_pc, match_pc, pred_pc = model(pc_pl)

            # Compute losses
            simplification_loss = model.get_simplification_loss(
                pc_pl, simp_pc
            )
            projection_loss = model.get_projection_loss()
            reconstrution_loss = model.get_reconstruction_loss(pc_pl, pred_pc)

            loss = reconstrution_loss + \
                alpha * simplification_loss + lmbda * projection_loss

            # Backward + Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch}, {loss}", flush=True)
        torch.save(model.state_dict(), "weight/encoder.pt")

    print("done!")
