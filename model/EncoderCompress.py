import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
import random

import torch.nn.functional as F

from metrics.evaluation_metrics import chamfer_distance_l2, chamfer_distance_l1
from model.Encoder_Component import Encoder, Group, TransformerEncoder

class PointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.trans_dim = config.trans_dim
        self.depth = config.encoder_depth
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.encoder_num_heads
        # embedding
        self.encoder_dims = config.trans_dim
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.mask_type = config.mask_type
        self.mask_ratio = config.mask_ratio
        self.group_size = config.group_size

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, neighborhood, center, mask):
        # generate mask
        B, _ = mask.size()
        _, _, C = neighborhood.shape
        vis = neighborhood.reshape(B, -1, self.group_size, C)
        group_input_tokens = self.encoder(vis)  # B G C

        batch_size, seq_len, L = group_input_tokens.size()
        vis_center = center[~mask].reshape(B, -1, C)
        p = self.pos_embed(vis_center)
        z = self.blocks(group_input_tokens, p)
        z = self.norm(z)
        return z.reshape(batch_size, -1, L)

class Encoder_Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.AE_encoder = PointTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.num_output = config.num_output
        self.mask_ratio = config.mask_ratio
        self.num_channel = 3
        self.drop_path_rate = config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, (self.num_channel * int(self.num_output * (1 - self.mask_ratio))) // self.num_group, 1)
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

    def forward(self, pts, neighborhood, center, mask, msk_pc):
        neighborhood = neighborhood.float()
        center = center.float()
        x_vis, vis_pc = self.encode(neighborhood, center, mask)

        B, _, C = x_vis.shape  # B VIS C

        rebuild_points = self.increase_dim(x_vis.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)  # 38, 32, 3

        full = torch.cat([rebuild_points, msk_pc], dim=1)
        loss1 = self.loss_func(full, pts)
        return loss1

    def encode(self, neighborhood, center, mask):
        neighborhood = neighborhood.float()
        center = center.float()
        x_vis = self.AE_encoder(neighborhood, center, mask)

        full_vis = self.neighborhood(neighborhood, center, mask, x_vis)
        return x_vis, full_vis


    def evaluate(self, x_vis, x_msk):
        B, _, C = x_vis.shape  # B VIS C
        x_full = torch.cat([x_vis, x_msk], dim=1)
        rebuild_points = self.increase_dim(x_full.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)  # 38, 32, 3
        return rebuild_points.reshape(-1, 3).unsqueeze(0)

    def neighborhood(self, neighborhood, center, mask, x_vis):
        B, M, N = x_vis.shape
        vis_point = neighborhood.reshape(B * M, -1, 3)
        full_vis = vis_point + center[~mask].unsqueeze(1)
        full_vis = full_vis.reshape(B, -1, 3)

        return full_vis


class Compress(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.mask_ratio = config.mask_ratio
        self.mask_type = config.mask_type

    def _mask_center_block(self, center, noaug=False):
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1,1,3) - points, p=2, dim=-1)
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        bool_masked_pos = torch.stack(mask_idx).to(center.device)
        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def compress(self, pt):
        neighborhood, center = self.group_divider(pt)
        if self.mask_type == 'rand':
            mask = self._mask_center_rand(center)  # B G
        else:
            mask = self._mask_center_block(center)
        vis_point = neighborhood[~mask]
        return vis_point.half(), center.half(), mask
