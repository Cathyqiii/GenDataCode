import torch
import torch.nn as nn
from mamba_ssm import Mamba
from .basic import RMSNorm

# ===================== 标准Mamba块 =====================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=min(d_model // 16, 64)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.mamba(self.norm(x))
        x = self.dropout(x)
        return x + residual

# ===================== 噪声预测器（修复输入维度） =====================
class MambaNoisePredictor(nn.Module):
    def __init__(self, input_dim, data_dim, mamba_dim, hidden_dim,
                 num_mamba_layers=1, kernel_size=3, dropout=0.1):
        super().__init__()
        # ✅ 修复：输入维度 = 2*embedding_dim（拼接后）
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2, mamba_dim),
            RMSNorm(mamba_dim),
            nn.GELU()
        )
        # 保留你原版的1层Mamba
        self.mamba_layers = nn.ModuleList([MambaBlock(mamba_dim) for _ in range(num_mamba_layers)])
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(mamba_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.final_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x):
        B, L, _ = x.shape
        x = self.input_proj(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.final_proj(x)