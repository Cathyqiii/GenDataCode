# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=device) *
            (-torch.log(torch.tensor(self.max_period, device=device)) / half_dim)
        )
        emb = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, [0, 1])
        return emb


# 优化：增强条件特征，适配小样本
class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim: int, embedding_dim: int, dropout: float = 0.15):
        super().__init__()
        hidden_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            RMSNorm(embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, cond_input):
        return self.encoder(cond_input)