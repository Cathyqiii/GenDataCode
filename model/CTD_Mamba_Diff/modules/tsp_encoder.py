# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# ===================== SE通道注意力（保留有效模块） =====================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, C = x.shape
        y = self.avg_pool(x.transpose(1, 2)).view(B, C)
        y = self.fc(y).view(B, 1, C)
        return x * y.expand_as(x)


# ===================== TimeMAR 官方核心分解模块 =====================
class moving_avg(nn.Module):
    """TimeMAR原生移动平均层，用于趋势提取"""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 两端填充保证序列长度不变
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    """TimeMAR 多尺度MoE趋势-季节分解"""

    def __init__(self, kernel_size=[5, 7]):
        super().__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            ma = func(x)
            moving_mean.append(ma.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        # MoE加权融合多尺度趋势
        weight = torch.softmax(self.layer(x.unsqueeze(-1)), dim=-1)
        trend = torch.sum(moving_mean * weight, dim=-1)
        seasonal = x - trend
        return seasonal, trend


# ===================== 修复维度版TSP编码器：Trend+Seasonal+Peak 三分量 =====================
class TSPEncoder(nn.Module):
    def __init__(self, in_dim, embedding_dim, kernel_size=7, freq_modes=24, denoise_modes=48):
        super().__init__()

        # 1. TimeMAR 趋势-季节分解（核心，论文参考）
        self.decomp_ts = series_decomp_multi(kernel_size=[5, 7])

        # 3. 三分量独立编码层
        # 趋势分支
        self.trend_proj = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )
        # 季节分支
        self.season_proj = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )
        # 峰值分支
        self.peak_proj = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )

        # 4. 通道注意力（每个分量独立加权）
        self.trend_se = SEBlock(embedding_dim)
        self.season_se = SEBlock(embedding_dim)
        self.peak_se = SEBlock(embedding_dim)

        # 5. 特征融合 + 残差
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )
        self.residual = nn.Linear(in_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.nan_to_num(x)
        B, L, C = x.shape
        res = self.residual(x)

        # ==============================================
        # 核心：TimeMAR分解 + 三分量解耦（Trend/Seasonal/Peak）
        # ==============================================
        # 步骤1：TimeMAR分解 → 季节分量 + 趋势分量
        seasonal, trend = self.decomp_ts(x)

        # 步骤2：修复维度！从季节分量解耦峰值（强制保持长度=L，兼容奇数/偶数）
        # 自适应平均池化提取粗季节，保证长度不变
        coarse_season = F.avg_pool1d(
            seasonal.transpose(1, 2),
            kernel_size=3,
            stride=1,
            padding=1
        ).transpose(1, 2)

        # 峰值分量：季节残差（高频、突变、峰值）→ 长度完全一致
        peak = seasonal - coarse_season

        # ==============================================
        # 三分量独立编码 + 通道注意力加权
        # ==============================================
        trend_emb = self.trend_se(self.trend_proj(trend))
        season_emb = self.season_se(self.season_proj(coarse_season))  # 稳定季节
        peak_emb = self.peak_se(self.peak_proj(peak))  # 峰值特征

        # ==============================================
        # 三特征融合 + 残差连接
        # ==============================================
        fused = torch.cat([trend_emb, season_emb, peak_emb], dim=-1)
        fused = self.fusion(fused) + res
        fused = self.dropout(fused)

        return torch.nan_to_num(fused)