# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# =============================
# 强制禁用 Triton selective kernels
# =============================
selective_state_update = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from huggingface_hub import PyTorchModelHubMixin


class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # =============================
        # 强制关闭 Triton mem-efficient path
        # =============================
        self.use_mem_eff_path = False

        self.d_model = d_model
        self.expand = expand
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // headdim
        self.layer_idx = layer_idx
        self.activation = "silu"

        # ===== Input projection =====
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, **factory_kwargs)

        # ===== Conv1d =====
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # ===== dt / A / D =====
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        A = torch.empty(self.nheads, **factory_kwargs).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))

        # ===== RMSNorm =====
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        if rmsnorm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                norm_before_gate=norm_before_gate,
            )

        # ===== Output projection =====
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    # ======================================================
    # Forward（永远走 PyTorch reference scan）
    # ======================================================
    def forward(self, u):
        B, L, _ = u.shape
        zxbcdt = self.in_proj(u)

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # ---- Conv ----
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
        )

        x, Bv, Cv = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        # ---- SSM reference scan ----
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        Bv = rearrange(Bv, "b l (g n) -> b l g n", g=self.ngroups)
        Cv = rearrange(Cv, "b l (g n) -> b l g n", g=self.ngroups)

        state = torch.zeros(
            B, self.nheads, self.headdim, self.d_state, device=u.device, dtype=u.dtype
        )

        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dBx = torch.einsum("bh,bgn,bhp->bhpn", dt[:, t], Bv[:, t], x[:, t])
            state = state * dA.unsqueeze(-1) + dBx
            y = torch.einsum("bhpn,bgn->bhp", state, Cv[:, t])
            y = y + self.D.unsqueeze(-1) * x[:, t]
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = rearrange(y, "b l h p -> b l (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        return self.out_proj(y)
