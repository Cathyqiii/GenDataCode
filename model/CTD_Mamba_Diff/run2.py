# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import warnings

warnings.filterwarnings("ignore")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from lib.config_loader import ConfigLoader
from lib.condition import build_conditions
import torch
import numpy as np
import random
# 恢复你原来的两个数据加载函数
from TSlib.lib.dataloader import real_data_loading
from lib.dataloader import full_real_data_loading
from modules.model import CTD_Mamba_Diff
from scipy.stats import wasserstein_distance


# ===================== 全局随机种子 =====================
def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train():
    set_seed(36)
    # -------------------- 配置数据集 --------------------
    data_name = 'AirQuality(Italian)'
    if data_name == 'etth1':
        opt = ConfigLoader("./config/etth1.conf")
    elif data_name == 'etth2':
        opt = ConfigLoader("config/etth2.conf")
    elif data_name == 'AirQuality(bj)':
        opt = ConfigLoader("config/AirQuality(bj).conf")
    elif data_name == 'AirQuality(Italian)':
        opt = ConfigLoader("./config/AirQuality(Italian).conf")
    elif data_name == 'Traffic':
        opt = ConfigLoader("./config/Traffic.conf")
    elif data_name in ["FD001", "FD002", "FD003", "FD004"]:
        opt = ConfigLoader("config/C-MAPSS.conf")

    feature_mode = getattr(opt, 'feature_mode')

    # -------------------- 数据加载（恢复原版函数） --------------------
    print(f"[DATASET] 使用的数据集:{opt.data_name}")
    print("\n[DEBUG DATA] 开始加载数据")
    # ✅ 恢复你原来的 real_data_loading：加载 1 维训练数据（匹配 data_dim=1）
    train_data, val_data, test_data, train_data_g = real_data_loading(data_name, feature_mode)
    ori_data = train_data_g  # 这是你需要的 1 维数据，用于训练/生成

    print(f"[INFO] 训练数据形状: {ori_data.shape}")

    # -------------------- 构建条件（原版逻辑） --------------------
    _, _, _, train_data_g_full, _, full_labels = full_real_data_loading(data_name)
    conditions = build_conditions(data_name, ori_data=train_data_g_full, labels=full_labels)

    condition_tensor = torch.tensor(conditions, dtype=torch.float32)
    print(f"[INFO] 条件数据形状: {condition_tensor.shape}")

    # -------------------- 初始化模型 --------------------
    model = CTD_Mamba_Diff(opt, ori_data)

    outf = getattr(opt, 'outf', './output')
    checkpoint_dir = getattr(opt, 'checkpoint_dir', os.path.join(outf, 'checkpoint'))
    generated_data_dir = getattr(opt, 'generated_data_dir', os.path.join(outf, 'generated'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(generated_data_dir, exist_ok=True)

    # -------------------- 训练 --------------------
    print("\n[INFO] 开始训练 CTD_Mamba_Diff 模型...")
    model.train_diff(conditions=condition_tensor)

    # -------------------- 生成数据 --------------------
    print("\n[INFO] 开始生成合成数据...")
    generate_times = getattr(opt, 'generate_times')
    original_condition = condition_tensor.to(model.device)
    synthetic_list = []

    for i in range(generate_times):
        print(f"[INFO] 生成第 {i + 1}/{generate_times} 批数据...")
        batch_data = model.generation(num_samples=ori_data.shape[0], conditions=original_condition)
        synthetic_list.append(batch_data)

    synthetic_data = torch.cat(synthetic_list, axis=0)
    synthetic_data_np = synthetic_data.cpu().detach().numpy()

    print(f"[INFO] 生成数据范围: [{synthetic_data_np.min():.6f}, {synthetic_data_np.max():.6f}]")
    print(f"[INFO] 原始数据范围: [{ori_data.min():.6f}, {ori_data.max():.6f}]")

    save_path = os.path.join(generated_data_dir, f"{data_name}.npy")
    np.save(save_path, synthetic_data_np)
    print(f"\n[INFO] 合成数据保存至: {save_path}")
    print(f"[INFO] 合成数据形状: {synthetic_data_np.shape}")
    print("[INFO] 全流程执行完成！")

    # ====================== 质量检查 ======================
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    def dtw_distance(x, y):
        n, m = len(x), len(y)
        dtw = np.zeros((n + 1, m + 1))
        dtw[:, 0] = np.inf
        dtw[0, :] = np.inf
        dtw[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i - 1] - y[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return dtw[n, m]

    print("\n" + "=" * 60)
    print("📊 生成数据质量全面检测报告")
    print("=" * 60)

    print("\n【1. 基础信息检查】")
    print(f"真实数据 shape: {ori_data.shape}")
    print(f"生成数据 shape: {synthetic_data_np.shape}")

    real_flat = ori_data.reshape(-1)
    fake_flat = synthetic_data_np.reshape(-1)
    w_dist = wasserstein_distance(real_flat, fake_flat)
    print(f"Wasserstein距离: {w_dist:.4f}（越小越好）")

    real_seq = ori_data[0, :, 0]
    fake_seq = synthetic_data_np[0, :, 0]
    real_ac = pearsonr(real_seq[1:], real_seq[:-1])[0]
    fake_ac = pearsonr(fake_seq[1:], fake_seq[:-1])[0]
    ac_diff = abs(real_ac - fake_ac)
    print(f"时序自相关差异: {ac_diff:.4f}（<0.1 优秀）")

    dtw_dist = dtw_distance(real_seq, fake_seq)
    print(f"DTW距离: {dtw_dist:.4f}（越小越好）")

    plt.rcParams['figure.figsize'] = (14, 5)
    plt.subplot(1, 2, 1)
    plt.plot(ori_data[0, :, 0], linewidth=2, label='真实数据', color='#2E86AB')
    plt.title('真实OT时序曲线')
    plt.grid(alpha=0.3)
    plt.legend()


    print("\n【5. 质量评分（时序专用）】")
    score = 0
    # 数值分布
    if w_dist < 0.05:
        score += 1
    if abs(ori_data.mean() - synthetic_data_np.mean()) < 0.02:
        score += 1
    # 时序结构
    if ac_diff < 0.1:
        score += 1
    if dtw_dist < 5.0:
        score += 1

    if score == 4:
        res = "✅ 完美质量，可直接用于下游预测"
    elif score >= 3:
        res = "🟢 优秀质量，满足科研/工程要求"
    elif score >= 2:
        res = "🟡 合格质量，微调生成噪声即可提升"
    else:
        res = "🔴 质量较差"
    print(f"综合评分: {score}/4 | {res}")

    print("\n【6. 时序可视化对比】")
    plt.rcParams['figure.figsize'] = (14, 5)
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(1, 2, 1)
    plt.plot(ori_data[0, :, 0], linewidth=2, label='真实数据', color='#2E86AB')
    plt.title('真实时序曲线', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(synthetic_data_np[0, :, 0], linewidth=2, label='生成数据', color='#E74C3C')
    plt.title('生成时序曲线', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()