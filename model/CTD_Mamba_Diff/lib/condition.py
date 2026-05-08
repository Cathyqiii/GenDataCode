# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

from model.CTD_Mamba_Diff.lib.dataloader import full_real_data_loading


# ===================== 归一化工具（仅连续条件使用） =====================
def MinMaxScaler(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    norm_data = (data - min_val) / ((max_val - min_val) + 1e-7)
    return norm_data


# ===================== 核心：条件构建（真实标签+区分离散/连续） =====================
def build_conditions(data_name, ori_data=None, labels=None, device='cuda:0'):
    """
    🔥 最终修正版：
    1. Traffic离散天气：真实标签提取 → 独热编码 → 无归一化 → 无VAE
    2. 连续条件：归一化 → 用于CVAE
    3. 彻底删除随机标签生成！
    """
    # 1. 加载数据
    if ori_data is not None:
        full_data = ori_data
    else:
        train_data, val_data, test_data, train_data_g, _, _ = full_real_data_loading(data_name)
        full_data = train_data_g

    # 2. 连续条件处理（传感器数据，需要CVAE）
    if data_name in ['etth1', 'etth2']:
        cond = full_data[:, :, :6]
        cond_norm = MinMaxScaler(cond)

    elif data_name == 'AirQuality(bj)':
        cond = full_data[:, :, :6]
        cond_norm = MinMaxScaler(cond)

    elif data_name == 'AirQuality(Italian)':
        cond_index = [9, 10, 11, 14]
        cond = full_data[:, :, cond_index]
        cond_norm = MinMaxScaler(cond)

    # ===================== 🔥 Traffic离散天气：真实标签提取（核心修复） =====================
    elif data_name == 'Traffic':
        N, T, D = full_data.shape
        # 7类天气固定映射
        WEATHER_CLASSES = ['clear', 'clouds', 'drizzle', 'haze', 'mist', 'rain', 'snow']

        # ✅ 正确：从真实数据中提取天气标签（标准Traffic数据集：最后一列是天气数值标签 0-6）
        # 每个样本对应一个全局天气标签，取序列第一个时间步的标签
        numeric_labels = full_data[:, 0, -1].astype(int)  # [N,] 数值标签 0~6
        # 数值标签映射为文本标签
        labels = np.array([WEATHER_CLASSES[idx] for idx in numeric_labels])

        # 独热编码
        encoder = OneHotEncoder(
            sparse_output=False,
            categories=[WEATHER_CLASSES],
            dtype=np.float32,
            handle_unknown='ignore'
        )
        weather_onehot = encoder.fit_transform(labels.reshape(-1, 1))

        # 扩展为时序格式 [N, T, 7]
        cond_norm = np.repeat(weather_onehot.reshape(N, 1, -1), T, axis=1)

    elif data_name.startswith('FD'):
        cond = full_data[:, 0, -1:]
        cond_norm = MinMaxScaler(cond)

    else:
        raise ValueError(f"不支持的数据集: {data_name}")

    print(f"✅ 条件构建完成 | 数据集: {data_name} | 条件形状: {cond_norm.shape}")
    return cond_norm


# ===================== 生成阶段：指定天气标签（不变） =====================
def get_single_condition(weather_label: str, seq_len=108):
    """
    生成专用：输入自定义天气标签，返回独热条件
    """
    WEATHER_CLASSES = ['clear', 'clouds', 'drizzle', 'haze', 'mist', 'rain', 'snow']
    encoder = OneHotEncoder(sparse_output=False, categories=[WEATHER_CLASSES], dtype=np.float32)
    onehot = encoder.fit_transform(np.array([[weather_label]]))
    cond = np.repeat(onehot.reshape(1, 1, -1), seq_len, axis=1)
    return torch.tensor(cond, dtype=torch.float32)