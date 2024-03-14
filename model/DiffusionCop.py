import torch.nn.functional as F
from metrics.evaluation_metrics import *
from model.Diffusion import VarianceSchedule, TimeEmbedding
from model.Decoder_Component import *

"""
Prediction on the masked patches only
w/ diffusion process
Use FC layer as the mask token convertor
"""


class Diff_Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.num_output = config.num_output
        self.num_channel = 3
        self.drop_path_rate = config.drop_path_rate
        self.mask_token = nn.Conv1d((self.num_channel * 2048) // self.num_group, self.trans_dim, 1)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = Transformer(
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
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_bar, t, x_0.shape).to(
            x_0.device)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, x_0, t, x_vis, mask, center, vis_pc, ori, debug=False):
        ts = self.time_emb(t.to(x_vis.device)).unsqueeze(1).expand(-1, self.num_group, -1)
        x_t, noise = self.forward_diffusion(x_0, t)

        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_msk = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        pos_full = torch.cat([pos_emd_vis, pos_emd_msk], dim=1)
        _, N, _ = pos_emd_msk.shape
        mask_token = self.mask_token(x_t.reshape(B, N, -1).transpose(1, 2)).transpose(1, 2).to(x_vis.device)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, N, ts)
        x_rec = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)
        pc_full = torch.cat([vis_pc, x_rec], dim=1)
        if debug:
            return x_t, noise, x_rec
        else:
            return self.loss_func(pc_full, ori)

    def sampling_t(self, x, t, mask, center, x_vis):
        center = center.float()
        B, _, C = x_vis.shape  # B VIS C
        ts = self.time_emb(t.to(x_vis.device)).unsqueeze(1).expand(-1, self.num_group, -1)
        betas_t = self.get_index_from_list(self.betas, t, x.shape).to(x_vis.device)

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_msk = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        pos_full = torch.cat([pos_emd_vis, pos_emd_msk], dim=1)
        _, N, _ = pos_emd_msk.shape
        mask_token = self.mask_token(x.reshape(B, N, -1).transpose(1, 2)).transpose(1, 2).to(x_vis.device)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        x_rec = self.MAE_decoder(x_full, pos_full, N, ts)
        x_rec = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)

        alpha_bar_t = self.get_index_from_list(self.alpha_bar, t, x.shape).to(x_vis.device)
        alpha_bar_t_minus_one = self.get_index_from_list(self.alpha_bar_t_minus_one, t, x.shape).to(x_vis.device)
        sqrt_alpha_t = self.get_index_from_list(self.sqrt_alphas, t, x.shape).to(x_vis.device)
        sqrt_alphas_bar_t_minus_one = self.get_index_from_list(self.sqrt_alpha_bar_minus_one, t, x.shape).to(
            x_vis.device)

        model_mean = (sqrt_alpha_t * (1 - alpha_bar_t_minus_one)) / (1 - alpha_bar_t) * x + (
                    sqrt_alphas_bar_t_minus_one * betas_t) / (1 - alpha_bar_t) * x_rec

        sigma_t = self.get_index_from_list(self.sigma, t, x.shape).to(x_vis.device)

        if t == 0:
            return model_mean
        else:
            return model_mean + torch.sqrt(sigma_t) * x_rec

    def sampling(self, x_vis, mask, center, trace=False, noise_patch=None):
        B, M, C = x_vis.shape
        if noise_patch is None:
            noise_patch = torch.randn((B, (self.num_group - M) * self.group_size, 3)).to(x_vis.device)
        diffusion_sequence = []

        for i in range(0, self.timestep)[::-1]:
            t = torch.full((1,), i, device=x_vis.device)
            noise_patch = self.sampling_t(noise_patch, t, mask, center, x_vis)
            if trace:
                diffusion_sequence.append(noise_patch.reshape(B, -1, 3))

        if trace:
            return diffusion_sequence
        else:
            return noise_patch.reshape(B, -1, 3)
