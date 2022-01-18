import torch
from pytorch_lightning import LightningModule
from models.pcrnet import PCRNet
from src.qdataset import QuaternionTransform, rad_to_deg
from src import ChamferDistance


class PCRNetLightning(LightningModule):
    def __init__(self, bottleneck_size=1024, input_shape="bnc") -> None:
        super().__init__()
        self.input_shape = input_shape
        self.model = PCRNet(bottleneck_size, input_shape)
        self.loss_func = self.compute_loss

    def compute_loss(self, p0, p1, igt, twist, pre_normalized_quat):
        # https://arxiv.org/pdf/1805.06485.pdf QuaterNet quaternient regularization loss
        qnorm_loss = torch.mean((torch.sum(pre_normalized_quat ** 2, dim=1) - 1) ** 2)

        est_transform = QuaternionTransform(twist)
        gt_transform = QuaternionTransform.from_dict(igt)

        p1_est = est_transform.rotate(p0)

        cost_p0_p1, cost_p1_p0 = ChamferDistance()(p1, p1_est)
        cost_p0_p1 = torch.mean(cost_p0_p1)
        cost_p1_p0 = torch.mean(cost_p1_p0)

        chamfer_loss = cost_p0_p1 + cost_p1_p0

        rot_err, norm_err, trans_err = est_transform.compute_errors(gt_transform)

        pcrnet_loss = 1.0 * norm_err + 1.0 * chamfer_loss
        # pcrnet_loss = chamfer_loss

        rot_err = rad_to_deg(rot_err)

        pcrnet_loss_info = {
            "chamfer_loss": chamfer_loss,
            "qnorm_loss": qnorm_loss,
            "rot_err": rot_err,
            "norm_err": norm_err,
            "trans_err": trans_err,
            # "est_transform": est_transform,
        }

        return pcrnet_loss, pcrnet_loss_info

    def forward(self, x0, x1):
        return self.model(x0, x1)

    def training_step(self, batch, batch_index):
        p0, p1, igt = batch
        twist, pre_normalized_quat = self(p0, p1)
        loss, loss_info = self.loss_func(p0, p1, igt, twist, pre_normalized_quat)

        train_logs = {
            f"train/{key}": val for key, val in loss_info.items()
        }
        self.log_dict(train_logs, on_step=False, on_epoch=True)
        self.log("train/total_loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_index):
        p0, p1, igt = batch
        twist, pre_normalized_quat = self(p0, p1)
        loss, loss_info = self.loss_func(p0, p1, igt, twist, pre_normalized_quat)

        return {"loss": loss, "loss_info": loss_info}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("valid/total_loss", avg_loss, on_step=False, on_epoch=True)

        # TODO: loss info も追加する
        avg_loss_info = {
            "chamfer_loss": torch.stack([x["loss_info"]["chamfer_loss"] for x in outputs]).mean(),
            "qnorm_loss": torch.stack([x["loss_info"]["qnorm_loss"] for x in outputs]).mean(),
            "rot_err": torch.stack([x["loss_info"]["rot_err"] for x in outputs]).mean(),
            "norm_err": torch.stack([x["loss_info"]["norm_err"] for x in outputs]).mean(),
            "trans_err": torch.stack([x["loss_info"]["trans_err"] for x in outputs]).mean()
        }
        for val, key in avg_loss_info.items():
            self.log(f"valid/{val}", key, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        learnable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.Adam(learnable_params, lr=1e-3)
