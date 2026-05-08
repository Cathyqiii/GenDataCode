# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.CTD_Mamba_Diff.lib.condition_vae import SeqConditionCVAE
from model.CTD_Mamba_Diff.modules.basic import SinusoidalEmbedding, ConditionEncoder, RMSNorm
from model.CTD_Mamba_Diff.modules.tsp_encoder import TSPEncoder
from model.CTD_Mamba_Diff.modules.mamba_blocks import MambaNoisePredictor


def debug_tensor(name, tensor):
    """调试函数，保留"""
    if tensor is None:
        print(f"[DEBUG] {name}: None")
        return
    t = tensor.detach().cpu()
    info = (
        f"[DEBUG] {name} | shape={tuple(t.shape)} | "
        f"min={t.min():.6f} | max={t.max():.6f} | "
        f"mean={t.mean():.6f} | std={t.std():.6f}"
    )
    print(info)
    if t.is_floating_point():
        if torch.isnan(t).any():
            print(f"[ERROR] {name} 包含 NaN！！！")
        if not torch.isfinite(t).all():
            print(f"[ERROR] {name} 包含 Inf 或 -Inf！！！")


class CTD_Mamba_Diff(nn.Module):
    def __init__(self, opt, train_data):
        super().__init__()
        self.opt = opt
        self.train_data = train_data
        # 训练数据转为张量，但不立即移到设备（在训练时再移）
        self.real_train_tensor = torch.tensor(train_data, dtype=torch.float32)

        # 读取配置参数
        self.seq_len = getattr(opt, 'seq_len')
        self.mamba_dim = getattr(opt, 'mamba_dim')
        self.hidden_dim = getattr(opt, 'hidden_dim')
        self.data_dim = getattr(opt, 'data_dim')
        self.embedding_dim = getattr(opt, 'embedding_dim')
        self.T = getattr(opt, 'diffusion_steps')
        self.condition_dim = getattr(opt, 'condition_dim')
        self.data_name = getattr(opt, 'data_name', '').lower()
        self.skip_vae_loading = (self.data_name == 'traffic')

        # 设备配置
        device_name = getattr(opt, 'device', 'cuda:0')
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

        # ===================== 子模块定义 =====================
        # 时间步嵌入（扩散步编码）
        self.time_embedding = SinusoidalEmbedding(self.embedding_dim)

        # 条件编码器（将VAE隐变量映射到嵌入维度）
        vae_latent_dim = getattr(opt, 'vae_latent_dim', 32)
        self.condition_encoder = ConditionEncoder(
            condition_dim=vae_latent_dim,
            embedding_dim=self.embedding_dim,
            dropout=getattr(opt, 'condition_dropout', 0.1)
        )

        # 条件VAE（用于连续条件编码，traffic数据集跳过）
        self.condition_vae = SeqConditionCVAE(
            condition_dim=self.condition_dim,
            latent_dim=vae_latent_dim,
            hidden_dim=getattr(opt, 'vae_hidden_dim', 128)
        )

        # 加载预训练VAE权重（若存在）
        if not self.skip_vae_loading:
            vae_ckpt = getattr(opt, 'condition_vae_ckpt', None)
            if vae_ckpt:
                vae_ckpt_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'lib', vae_ckpt
                )
                if os.path.exists(vae_ckpt_path):
                    try:
                        state_dict = torch.load(vae_ckpt_path, map_location='cpu')
                        self.condition_vae.load_state_dict(state_dict)
                        print(f"Loaded VAE checkpoint from {vae_ckpt_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load VAE checkpoint: {e}")

        # 冻结VAE参数
        self.condition_vae.eval()
        for p in self.condition_vae.parameters():
            p.requires_grad = False

        # 条件与时间步融合模块
        self.condition_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            RMSNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # 时序分解编码器
        self.feature_extractor = TSPEncoder(
            in_dim=self.data_dim,
            embedding_dim=self.embedding_dim,
            kernel_size=7,
            freq_modes=32
        )

        # 噪声预测器（输入是拼接后的特征：2*embedding_dim）
        self.noise_predictor = MambaNoisePredictor(
            input_dim=self.embedding_dim,
            data_dim=self.data_dim,
            mamba_dim=self.mamba_dim,
            hidden_dim=self.hidden_dim,
            num_mamba_layers=1,
            kernel_size=3,
            dropout=0.1
        )

        # ===================== 扩散参数初始化 =====================
        self._init_diffusion_params()

        # 数据统计量（用于分布匹配，可选）
        self.real_train_tensor = self.real_train_tensor.to(self.device)
        self.real_min = self.real_train_tensor.min()
        self.real_max = self.real_train_tensor.max()
        self.mu_true = self.real_train_tensor.mean([0, 1], keepdim=True)
        self.var_true = self.real_train_tensor.var([0, 1], keepdim=True, unbiased=False)
        self.std_true = torch.sqrt(self.var_true)
        self.kl_correction_gamma = 0.0001  # 可选的KL修正系数，暂时保留

        # 将所有模块移到目标设备
        self.to(self.device)

    def _init_diffusion_params(self):
        """
        初始化扩散过程所需的参数，参考标准DDPM实现。
        使用register_buffer确保参数随模型移动设备。
        """
        T = self.T
        s = 0.008  # 余弦调度偏移

        steps = T + 1
        t = torch.arange(steps, dtype=torch.float32)
        # 余弦调度 (Nichol & Dhariwal, 2021)
        alpha_bar = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.clamp(1 - (alpha_bar[1:] / alpha_bar[:-1]), min=0.0001, max=0.02)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 补上t=0时刻的值
        betas = torch.cat([torch.zeros(1), betas], dim=0)          # β_0 = 0
        alphas = torch.cat([torch.ones(1), alphas], dim=0)         # α_0 = 1
        alphas_cumprod = torch.cat([torch.ones(1), alphas_cumprod], dim=0)  # ᾱ_0 = 1

        # 计算后验方差：β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas[1:] * (1 - alphas_cumprod[:-1]) / (1 - alphas_cumprod[1:])

        # 注册为buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)

    def forward(self, x, conditions=None, t=None, use_condition=False, training=True):
        """
        前向传播：给定带噪数据x_t，时间步t，可选条件，预测噪声ε。
        """
        B, L, F = x.shape
        x = x.to(self.device)

        # 处理条件
        if use_condition and conditions is not None:
            conditions = conditions.to(self.device)
            with torch.no_grad():
                mu, logvar = self.condition_vae.encode(conditions)
                z = self.condition_vae.reparameterize(mu, logvar)
                h_c = self.condition_encoder(z)
                if h_c.dim() == 2:
                    h_c = h_c.unsqueeze(1).expand(-1, L, -1)
        else:
            h_c = torch.zeros(B, L, self.embedding_dim, device=self.device)

        # 时间步嵌入
        if t is not None:
            t = t.to(self.device)
            h_t = self.time_embedding(t)
            if h_t.dim() == 2:
                h_t = h_t.unsqueeze(1).expand(-1, L, -1)
        else:
            h_t = torch.zeros(B, L, self.embedding_dim, device=self.device)

        # 条件与时间步融合
        h_cond = self.condition_fusion(torch.cat([h_c, h_t], dim=-1))

        # 时序特征提取
        E_all = self.feature_extractor(x)

        # 拼接特征输入噪声预测器
        mamba_input = torch.cat([E_all, h_cond], dim=-1)
        eps_theta = self.noise_predictor(mamba_input)

        return eps_theta

    def diffusion_forward(self, x0, t, noise=None):
        """
        前向扩散过程：q(x_t | x_0)
        """
        x0 = x0.to(self.device)
        t = t.to(self.device)
        if noise is None:
            noise = torch.randn_like(x0, device=self.device)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    def train_diff(self, conditions=None):
        """训练扩散模型"""
        self.train()
        opt = self.opt
        optimizer = optim.AdamW(self.parameters(), lr=float(opt.lr), weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs)

        train_tensor = self.real_train_tensor.to(self.device)
        if conditions is not None:
            conditions = conditions.to(self.device)
            dataset = torch.utils.data.TensorDataset(train_tensor, conditions)
        else:
            dataset = torch.utils.data.TensorDataset(train_tensor)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True
        )

        best_loss = float('inf')
        early_stop_counter = 0
        patience = 30

        for epoch in range(opt.epochs):
            total_loss = 0.0
            for batch in dataloader:
                if len(batch) == 2:
                    x0, cond = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    x0, cond = batch[0].to(self.device), None

                t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
                xt, noise = self.diffusion_forward(x0, t)
                eps_theta = self(xt, cond, t, use_condition=(cond is not None))

                loss = F.mse_loss(eps_theta, noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch+1}/{opt.epochs}] loss = {avg_loss:.6f}")
            scheduler.step()

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                early_stop_counter = 0
                ckpt_dir = getattr(opt, 'checkpoint_dir', './checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(self.state_dict(), os.path.join(ckpt_dir, "CTD_Mamba_best.pth"))
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

    @torch.no_grad()
    def generation(self, num_samples=None, conditions=None):
        """
        逆扩散采样生成新样本。
        严格遵循DDPM采样公式，无额外KL修正。
        """
        self.eval()
        device = self.device

        if num_samples is None:
            num_samples = self.real_train_tensor.shape[0]
        B, L, F = num_samples, self.seq_len, self.data_dim

        # 从标准高斯噪声开始
        xt = torch.randn(B, L, F, device=device)

        # 处理条件
        if conditions is not None:
            conditions = conditions.to(device)
            if conditions.shape[0] < B:
                # 简单重复以满足batch_size
                repeat_times = (B // conditions.shape[0]) + 1
                conditions = conditions.repeat(repeat_times, 1, 1)[:B]
            elif conditions.shape[0] > B:
                conditions = conditions[:B]

        # 逆扩散循环
        for t in reversed(range(0, self.T)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # 预测噪声
            eps_theta = self(xt, conditions, t_tensor, use_condition=(conditions is not None))

            # 系数
            alpha_t = self.alphas[t_tensor].view(-1, 1, 1)
            alpha_t_cumprod = self.alphas_cumprod[t_tensor].view(-1, 1, 1)
            sqrt_alpha_t_cumprod = self.sqrt_alphas_cumprod[t_tensor].view(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t_tensor].view(-1, 1, 1)
            beta_t = self.betas[t_tensor].view(-1, 1, 1)

            # 预测原始样本x_0
            x0_pred = (xt - sqrt_one_minus_alpha_cumprod * eps_theta) / sqrt_alpha_t_cumprod

            # 裁剪到合理范围（防止溢出）
            x0_pred = torch.clamp(x0_pred, self.real_min, self.real_max)

            if t > 0:
                # 采样后验分布的均值
                # μ_θ = (√ᾱ_{t-1} * β_t / (1-ᾱ_t)) * x0_pred + (√α_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)) * x_t
                # 等价于常用公式：μ_θ = (x_t - β_t/√(1-ᾱ_t) * ε_θ) / √α_t
                # 这里使用基于x0_pred的公式，再加噪声
                alpha_t_cumprod_prev = self.alphas_cumprod[t_tensor - 1].view(-1, 1, 1)
                sqrt_alpha_t_cumprod_prev = torch.sqrt(alpha_t_cumprod_prev)

                # 后验均值
                posterior_mean = (
                    sqrt_alpha_t_cumprod_prev * beta_t / (1 - alpha_t_cumprod) * x0_pred +
                    torch.sqrt(alpha_t) * (1 - alpha_t_cumprod_prev) / (1 - alpha_t_cumprod) * xt
                )

                # 后验方差
                posterior_var = self.posterior_variance[t_tensor].view(-1, 1, 1)

                # 采样下一步的x_{t-1}
                noise = torch.randn_like(xt, device=device)
                xt = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                # t=0时直接输出x0
                xt = x0_pred

        # 最终裁剪
        xt = torch.clamp(xt, self.real_min, self.real_max)
        return xt.cpu()

    def sample_condition(self, num_samples):
        """从条件VAE先验采样生成条件"""
        self.condition_vae.eval()
        z = torch.randn(num_samples, self.condition_vae.latent_dim, device=self.device)
        return self.condition_vae.decode(z).cpu()

    def save_data(self, conditions=None):
        """保存生成的数据"""
        save_dir = getattr(self.opt, 'generated_data_dir', './output')
        os.makedirs(save_dir, exist_ok=True)
        data_name = getattr(self.opt, 'data_name', 'etth1')
        path = os.path.join(save_dir, f"{data_name}.npy")

        all_data = []
        for i in range(3):  # 生成三次取平均或合并
            batch_data = self.generation(conditions=conditions)
            all_data.append(batch_data.numpy())

        final_data = np.concatenate(all_data, axis=0)
        np.save(path, final_data)
        print(f"Generated data saved to {path}")
        return final_data