import torch.nn.functional as F

from metrics.evaluation_metrics import chamfer_distance_l1, chamfer_distance_l2
from model.Encoder_Component import *

class Encoder_Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.AE_encoder = PointTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.num_output = config.num_output
        self.num_channel = 3
        self.drop_path_rate = config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, (self.num_channel * self.num_output) // self.num_group, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = chamfer_distance_l1
        elif loss_type == 'cdl2':
            self.loss_func = chamfer_distance_l2
        elif loss_type == 'mse':
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, pts, hr_pt):
        x_vis, x_msk, mask, center, vis_pc, msk_pc = self.encode(pts, False)

        B, _, C = x_vis.shape  # B VIS C
        x_full = torch.cat([x_vis, x_msk], dim=1)
        rebuild_points = self.increase_dim(x_full.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)  # 38, 32, 3
        loss1 = self.loss_func(rebuild_points, hr_pt)
        return loss1

    def encode(self, pt, masked=False):
        B, _, N = pt.shape
        neighborhood, center = self.group_divider(pt)
        x_vis, mask, x_masked = self.AE_encoder(neighborhood, center)
        if masked:
            return x_vis, mask, center
        else:
            vis_pc, msk_pc = self.neighborhood(neighborhood, center, mask, x_vis)
            return x_vis, x_masked, mask, center, vis_pc, msk_pc

    def evaluate(self, x_vis, x_msk):
        B, _, C = x_vis.shape  # B VIS C
        x_full = torch.cat([x_vis, x_msk], dim=1)
        rebuild_points = self.increase_dim(x_full.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)  # 38, 32, 3
        return rebuild_points.reshape(-1, 3).unsqueeze(0)

    def neighborhood(self, neighborhood, center, mask, x_vis):
        B, M, N = x_vis.shape
        vis_point = neighborhood[~mask].reshape(B * M, -1, 3)
        full_vis = vis_point + center[~mask].unsqueeze(1)
        msk_point = neighborhood[mask].reshape(B * (self.num_group - M), -1, 3)
        full_msk = msk_point + center[mask].unsqueeze(1)

        full_vis = full_vis.reshape(B, -1, 3)
        full_msk = full_msk.reshape(B, -1, 3)

        return full_vis, full_msk
