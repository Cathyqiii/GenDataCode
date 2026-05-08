"""TimeGAN-pytorch Codebase.
（最终纯净版：无反归一化 + 维度100%匹配 + 适配下游预测）
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
from TSlib.lib.dataloader import real_data_loading
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
    _script_dir = _os.path.dirname(os.path.abspath(__file__))
    _config_dir = os.path.join(_script_dir, "config")

    # 遍历数据集
    for data_name in ['FD001']:
        if data_name in []:
            continue

        # 加载配置文件
        if data_name in ['ETTh1']:
            opt = ConfigLoader(os.path.join(_config_dir, "etth1.conf"))
        elif data_name in ['ETTh2']:
            opt = ConfigLoader(os.path.join(_config_dir, "etth2.conf"))
        elif data_name == 'AirQuality(bj)':
            opt = ConfigLoader(os.path.join(_config_dir, "AirQuality(bj).conf"))
        elif data_name == 'AirQuality(Italian)':
            opt = ConfigLoader(os.path.join(_config_dir, "AirQuality(Italian).conf"))
        elif data_name == 'Traffic':
            opt = ConfigLoader(os.path.join(_config_dir, "Traffic.conf"))
        elif data_name == "FD001":
            opt = ConfigLoader(os.path.join(_config_dir, "FD001.conf"))
        else:
            opt = ConfigLoader(os.path.join(_config_dir, "etth1.conf"))

        print(f"[{data_name}] 开始加载数据集...")

        # 获取特征类型
        feature = feature_map[data_name]

        # 调用加载函数
        train_data, val_data, test_data, train_data_g = real_data_loading(data_name, feature)

        # ===================== ✅ 关键修复：直接用全部维度，不丢特征 =====================
        ori_data_norm = train_data_g
        print(f"[{data_name}] 数据集加载完成，数据形状: {ori_data_norm.shape}")

        # LOAD MODEL
        model = TimeGAN(opt, ori_data_norm)

        # TRAIN MODEL
        print("[INFO] 开始训练 TimeGAN...")
        model.train()

        # ===================== 生成数据后处理（仅格式+长度对齐） =====================
        synthetic_data_norm = np.array(model.generated_data)

        # 统一序列长度，不修改特征维度
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

        # ===================== 输出路径 =====================
        _output_dir = os.path.join(_script_dir, "output", data_name)
        os.makedirs(_output_dir, exist_ok=True)

        save_path = os.path.join(_output_dir, f"{data_name}.npy")
        np.save(save_path, synthetic_data_arr)

        print(f"\n✅ [{data_name}] 训练完成！已保存: {synthetic_data_arr.shape}")
        print(f"✅ 保存路径: {save_path}")

if __name__ == "__main__":
    train()