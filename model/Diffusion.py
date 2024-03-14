import torch
import torch.nn as nn
import numpy as np
import math

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


class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding for the diffusion process.

    Input:
    Timestep: the current timestep, in range [1, ..., T]
    """
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