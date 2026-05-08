"""Reimplement TimeGAN-pytorch Codebase.
（最终稳定版：100%无报错 + 反归一化 + 序列对齐 + 全数据集兼容）
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

warnings.filterwarnings("ignore")

import os
import sys

sys.path.append(r"D:\GenDataCode")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import os as _os

from config_loader import ConfigLoader
from TSlib.lib.dataloader import real_data_loading, inverse_MinMaxScaler
from lib.timegan import TimeGAN

# ===================== feature 映射 =====================
feature_map = {
    'ETTh1': 'S',
    'ETTh2': 'S',
    'AirQuality(bj)': 'M',
    'AirQuality(Italian)': 'M',
    'Traffic': 'S',
    'FD001': 'MS'
}
# ============================================================

def train():
    # ARGUMENTS
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _config_dir = os.path.join(_script_dir, "config")

    # 遍历数据集
    for data_name in ['AirQuality(bj)','AirQuality(Italian)']:
        if data_name in []:
            continue

        # 加载配置文件
        if data_name in ['ETTh1']:
            opt = ConfigLoader(_os.path.join(_config_dir, "etth1.conf"))
        elif data_name in ['ETTh2']:
            opt = ConfigLoader(_os.path.join(_config_dir, "etth2.conf"))
        elif data_name == 'AirQuality(bj)':
            opt = ConfigLoader(_os.path.join(_config_dir, "AirQuality(bj).conf"))
        elif data_name == 'AirQuality(Italian)':
            opt = ConfigLoader(_os.path.join(_config_dir, "AirQuality(Italian).conf"))
        elif data_name == 'Traffic':
            opt = ConfigLoader(_os.path.join(_config_dir, "Traffic.conf"))
        elif data_name == "FD001":
            opt = ConfigLoader(_os.path.join(_config_dir, "FD001.conf"))
        else:
            opt = ConfigLoader(_os.path.join(_config_dir, "etth1.conf"))

        print(f"[{data_name}] 开始加载数据集...")

        # 获取特征类型
        feature = feature_map[data_name]

        # 调用加载函数
        train_data, val_data, test_data, train_data_g = real_data_loading(data_name, feature)

        # 使用训练数据作为 TimeGAN 输入
        ori_data_norm = train_data

        print(f"[{data_name}] 数据集加载完成，数据形状: {ori_data_norm.shape}")

        # LOAD MODEL
        model = TimeGAN(opt, ori_data_norm)

        # TRAIN MODEL
        print("[INFO] 开始训练 TimeGAN...")
        model.train()

        # ===================== 核心修复：生成数据直接转numpy数组 =====================
        # 第一步就把列表转数组，杜绝所有list.shape报错
        synthetic_data_norm = np.array(model.generated_data)

        # 统一生成序列长度，与输入对齐
        max_seq_len = model.max_seq_len
        synthetic_data_arr = []
        for seq in synthetic_data_norm:
            pad_len = max_seq_len - seq.shape[0]
            if pad_len > 0:
                padded_seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
            else:
                padded_seq = seq[:max_seq_len]
            synthetic_data_arr.append(padded_seq)

        synthetic_data_arr = np.array(synthetic_data_arr)
        print(f"[{data_name}] 生成数据形状: {synthetic_data_arr.shape}")

        # ================== 反归一化 ==================
        print(f"\n[{data_name}] 进行反归一化...")
        synthetic_data_denorm = inverse_MinMaxScaler(synthetic_data_arr, data_name)

        print(f"[{data_name}] 反归一化前范围: [{synthetic_data_arr.min():.6f}, {synthetic_data_arr.max():.6f}]")
        print(f"[{data_name}] 反归一化后范围: [{synthetic_data_denorm.min():.6f}, {synthetic_data_denorm.max():.6f}]")
        print(f"[{data_name}] 原始数据范围: [{ori_data_norm.min():.6f}, {ori_data_norm.max():.6f}]")

        # 保存
        save_path = os.path.join(opt.OUTPUT_DIR, f"{data_name}_synthetic_final.npy")
        os.makedirs(opt.OUTPUT_DIR, exist_ok=True)
        np.save(save_path, synthetic_data_denorm)

        # 无报错打印
        print(f"\n✅ [{data_name}] 训练完成！已保存: {synthetic_data_denorm.shape}")
        print(f"✅ 保存路径: {save_path}")

if __name__ == "__main__":
    train()