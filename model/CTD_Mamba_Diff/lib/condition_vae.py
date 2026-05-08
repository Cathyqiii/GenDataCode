# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model.CTD_Mamba_Diff.lib.config_loader import ConfigLoader
from TSlib.lib.dataloader import real_data_loading
from model.CTD_Mamba_Diff.lib.condition import build_conditions
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===================== 工具函数 =====================
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

# ==============================================
# 连续条件专用 CVAE
# ==============================================
class SeqConditionCVAE(nn.Module):
    def __init__(self, condition_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim)
        )

    def encode(self, c):
        feat = self.encoder(c)
        mu = self.mu_layer(feat)
        logvar = torch.clamp(self.logvar_layer(feat), -4, 4)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        c_recon = self.decoder(z)
        return c_recon

    def forward(self, c):
        mu, logvar = self.encode(c)
        z = self.reparameterize(mu, logvar)
        recon_c = self.decode(z)
        return recon_c, mu, logvar

    def inference(self, c):
        mu, logvar = self.encode(c)
        z = self.reparameterize(mu, logvar)
        return z

# ==============================================
# MMD损失
# ==============================================
def mmd_loss(x, y):
    x = x.reshape(x.size(0), -1)
    y = y.reshape(y.size(0), -1)
    sigmas = [0.5, 1, 2]
    loss = 0.0
    for sigma in sigmas:
        kxx = torch.exp(-torch.cdist(x, x) ** 2 / (2 * sigma ** 2))
        kyy = torch.exp(-torch.cdist(y, y) ** 2 / (2 * sigma ** 2))
        kxy = torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2))
        loss += (kxx.mean() + kyy.mean() - 2 * kxy.mean())
    return loss / len(sigmas)

# ==============================================
# 连续条件 CVAE 训练函数
# ==============================================
# ===================== 连续条件 CVAE 训练函数（稳定版） =====================
def train_seq_condition_cvae(condition_data, opt, device="cuda"):
    original_shape = condition_data.shape
    condition_dim = condition_data.shape[-1]

    batch_size = opt.batch_size
    epochs = 300
    lr = opt.lr
    save_path = opt.condition_vae_ckpt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cond_tensor = torch.from_numpy(condition_data).float().to(device)

    model = SeqConditionCVAE(
        condition_dim=condition_dim,
        latent_dim=opt.vae_latent_dim,
        hidden_dim=opt.vae_hidden_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    dataloader = DataLoader(
        TensorDataset(cond_tensor),
        batch_size=batch_size, shuffle=True, drop_last=True)

    best_loss = float('inf')
    early_stop_patience = 30
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_recon = 0.0
        total_kl = 0.0
        total_mmd = 0.0

        # ✅ KL warm-up（更强）
        kl_weight = min(0.1, epoch / 100 * 0.1)

        # ✅ MMD降权
        mmd_weight = 2.0

        for (batch_c,) in dataloader:
            batch_c = batch_c.to(device)
            recon_c, mu, logvar = model(batch_c)

            # ===================== Loss =====================
            recon_loss = F.mse_loss(recon_c, batch_c)

            # ✅ KL per-dim（free bits）
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            free_bits = 0.2
            kl_loss = torch.mean(torch.clamp(kl_per_dim, min=free_bits))

            # MMD
            mmd = mmd_loss(recon_c, batch_c)

            # ✅ latent 正则（防发散）
            z_norm = torch.mean(torch.norm(mu, dim=-1))

            loss = (
                recon_loss
                + kl_weight * kl_loss
                + mmd_weight * mmd
                + 0.001 * z_norm
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_mmd += mmd.item()

        scheduler.step()

        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        avg_mmd = total_mmd / len(dataloader)

        current_total = avg_recon + kl_weight * avg_kl + mmd_weight * avg_mmd

        if current_total < best_loss:
            best_loss = current_total
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:3d}] Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | MMD: {avg_mmd:.4f} | Best: {best_loss:.4f}")

        if early_stop_counter >= early_stop_patience:
            print("Early stop triggered.")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model
# ==============================================
# 主函数
# ==============================================
if __name__ == "__main__":
    data_name = "FD001"

    if data_name == 'etth1':
        opt = ConfigLoader("../config/etth1.conf")
    elif data_name == 'etth2':
        opt = ConfigLoader("../config/etth2-backup.conf")
    elif data_name == 'AirQuality(bj)':
        opt = ConfigLoader("../config/AirQuality(bj)_backup.conf")
    elif data_name == 'AirQuality(Italian)':
        opt = ConfigLoader("../config/AirQuality(Italian).conf")
    elif data_name == 'Traffic':
        opt = ConfigLoader("../config/Traffic.conf")
    elif data_name in ["FD001", "FD002", "FD003", "FD004"]:
        opt = ConfigLoader("../config/C-MAPSS.conf")
    else:
        opt = ConfigLoader("../config/etth1.conf")

    feature_mode = getattr(opt, 'feature_mode')
    train_data, val_data, test_data, train_data_g = real_data_loading(data_name, feature_mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cond = build_conditions(data_name, device=device)

    # ===================== 核心修改 =====================
    # 1. Traffic离散条件：跳过CVAE，直接保留原始数据，交给后续ConditionEncoder处理
    # 2. 完全删除离散条件嵌入层，无任何额外编码
    if data_name == "Traffic":
        print("🔹 Traffic离散天气条件：跳过CVAE，直接输入至ConditionEncoder")
        sample_cond = torch.from_numpy(cond[:2]).float().to(device)
        print(f"✅ 离散条件原始形状: {sample_cond.shape}（将直接传入ConditionEncoder）")
    # 连续时序条件：正常训练CVAE，输出隐变量
    else:
        print("🔹 连续时序条件：训练CVAE生成隐变量")
        vae_model = train_seq_condition_cvae(cond, opt, device)
        sample_cond = torch.from_numpy(cond[:2]).float().to(device)
        latent_z = vae_model.inference(sample_cond)
        print(f"✅ 连续条件隐变量形状: {latent_z.shape}（将传入ConditionEncoder）")