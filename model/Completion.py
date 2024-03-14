from knn_cuda import KNN
from metrics.evaluation_metrics import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from utils import misc
import random


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, n):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        x = self.head(self.norm(x[:, -n:]))
        return x


# Pretrain model
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
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

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C

        batch_size, seq_len, C = group_input_tokens.size()

        p = self.pos_embed(center)

        z = self.blocks(group_input_tokens, p)
        z = self.norm(z)
        return z[~bool_masked_pos].reshape(batch_size, -1, C), bool_masked_pos, z[bool_masked_pos].reshape(batch_size, -1, C)

class Point_MAE(nn.Module):
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
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.AE_decoder = PointTransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

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
        x_vis, x_msk, mask, center = self.encode(pts, False)

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


#Decoder
"""
Prediction on the masked patches only
w/ diffusion process
Use FC layer as the mask token convertor
"""

class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
            # create a 1D tensor of size num_steps, values are evenly spaced from beta1 and betaT.
            # beta1, ... , betaT are hyper-parameter that control the diffusion rate of the process.

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding a 0 at beginning.

        alphas = 1 - betas
        log_alphas = torch.log(alphas)  # (7)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
            # log alpha add all previous step
        alpha_bars = log_alphas.exp()  # ?

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)  # a 0 filled tensor with sigmas_flex dimension
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]  # (11)
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, num_of_group, ts):
        for _, block in enumerate(self.blocks):
            x = block(x + ts)
        x = self.head(self.norm(x[:, -num_of_group:]))
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb_dim = dim

    def forward(self, ts):
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=ts.device) * -emb)
        emb = ts[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Diff_Point_MAE(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.num_output = config.num_output
        self.num_channel = 3
        self.drop_path_rate = config.drop_path_rate
        self.mask_token = nn.Conv1d((self.num_channel * 2048) // self.num_group, self.trans_dim,  1)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)
        self.var = VarianceSchedule(
            num_steps=config.num_steps,
            beta_1=config.beta_1,
            beta_T=config.beta_T,
            mode=config.sched_mode
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, (self.num_channel * 2048) // self.num_group, 1)
        )

        self.increase_dim_hr = nn.Sequential(
            nn.Conv1d(self.trans_dim, (self.num_channel * 8192) // self.num_group, 1)
        )

        self.encoder = encoder

        self.timestep = config.num_steps
        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T

        self.betas = self.linear_schedule(timesteps=self.timestep)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, axis=0)
        self.alpha_bar_t_minus_one = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sigma = self.betas * (1.0 - self.alpha_bar_t_minus_one) / (1.0 - self.alpha_bar)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alpha_bar_minus_one = torch.sqrt(self.alpha_bar_t_minus_one)

        self.time_emb = nn.Sequential(
            TimeEmbedding(self.trans_dim),
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.ReLU()
        )

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = chamfer_distance_l1
        elif loss_type == 'cdl2':
            self.loss_func = chamfer_distance_l2
        elif loss_type == 'mse':
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def linear_schedule(self, timesteps):
        return torch.linspace(self.beta_1, self.beta_T, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        b = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0).to(x_0.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_bar, t, x_0.shape).to(x_0.device)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_bar, t, x_0.shape).to(x_0.device)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise


    def forward(self, x_0, t, x_vis, vis_pc, ori, debug=False):
        # print()
        # print(t.size())
        ts = self.time_emb(t.to(x_vis.device)).unsqueeze(1).expand(-1, self.num_group, -1)
        # print(ts.size())
        x_t, noise = self.forward_diffusion(x_0, t)

        B, M, C = x_vis.shape  # B VIS C

        mask_token = self.mask_token(x_t.reshape(B, self.num_group - M, -1).transpose(1, 2)).transpose(1, 2).to(x_vis.device)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        x_rec = self.MAE_decoder(x_full, self.num_group - M, ts)
        x_rec = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)
        pc_full = torch.cat([vis_pc, x_rec], dim=1)

        if debug:
            return x_t, noise, x_rec
        else:
            return self.loss_func(pc_full, ori)
            # return F.mse_loss(x_rec, x_0, reduction='mean')

    def sampling_t(self, x, t, x_vis):
        B, M, C = x_vis.shape  # B VIS C
        ts = self.time_emb(t.to(x_vis.device)).unsqueeze(1).expand(-1, self.num_group, -1)
        betas_t = self.get_index_from_list(self.betas, t, x.shape).to(x_vis.device)
        N = self.num_group - M
        mask_token = self.mask_token(x.reshape(B, N, -1).transpose(1, 2)).transpose(1, 2).to(x_vis.device)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        x_rec = self.MAE_decoder(x_full, N, ts)
        x_rec = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)


        alpha_bar_t = self.get_index_from_list(self.alpha_bar, t, x.shape).to(x_vis.device)
        alpha_bar_t_minus_one = self.get_index_from_list(self.alpha_bar_t_minus_one, t, x.shape).to(x_vis.device)
        sqrt_alpha_t = self.get_index_from_list(self.sqrt_alphas, t, x.shape).to(x_vis.device)
        sqrt_alphas_bar_t_minus_one = self.get_index_from_list(self.sqrt_alpha_bar_minus_one, t, x.shape).to(x_vis.device)

        model_mean = (sqrt_alpha_t * (1 - alpha_bar_t_minus_one)) / (1 - alpha_bar_t) * x + (sqrt_alphas_bar_t_minus_one * betas_t) / (1 - alpha_bar_t) * x_rec

        sigma_t = self.get_index_from_list(self.sigma, t, x.shape).to(x_vis.device)

        if t == 0:
            return model_mean
        else:
            return model_mean + torch.sqrt(sigma_t) * x_rec

    def sampling(self, x_vis, ret=False, noise_patch=None):
        B, M, C = x_vis.shape
        if noise_patch is None:
            noise_patch = torch.randn((B, (self.num_group - M) * self.group_size, 3)).to(x_vis.device)
        traj = []

        for i in range(0, self.timestep)[::-1]:
            t = torch.full((1,), i, device=x_vis.device)
            noise_patch = self.sampling_t(noise_patch, t, x_vis)
            if ret:
                traj.append(noise_patch.reshape(B, -1, 3))

        if ret:
            return traj
        else:
            return noise_patch.reshape(B, -1, 3)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('model'):
                    base_ckpt[k[len('model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            self.load_state_dict(base_ckpt, strict=False)
        else:
            self.apply(self._init_weights)