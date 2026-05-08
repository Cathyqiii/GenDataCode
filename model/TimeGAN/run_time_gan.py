"""Reimplement TimeGAN-pytorch Codebase.
【完全稳定版 - 无报错、不卡住】
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
    'AirQuality(bj)': 'M',
    'AirQuality(Italian)': 'M',
}
# ============================================================

def train():
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _config_dir = os.path.join(_script_dir, "config")

    # 一次跑一个
    for data_name in ['AirQuality(bj)','AirQuality(Italian)']:

        # 加载配置
        if data_name == 'AirQuality(bj)':
            opt = ConfigLoader(os.path.join(_config_dir, "AirQuality(bj).conf"))
        elif data_name == 'AirQuality(Italian)':
            opt = ConfigLoader(os.path.join(_config_dir, "AirQuality(Italian).conf"))
        else:
            opt = ConfigLoader(os.path.join(_config_dir, "etth1.conf"))

        print(f"[{data_name}] 加载数据...")
        feature = feature_map[data_name]
        train_data, val_data, test_data, train_data_g = real_data_loading(data_name, feature)

        ori_data_norm = train_data
        print(f"[{data_name}] 形状: {ori_data_norm.shape}")

        # ===================== 【干净、无报错】模型初始化 =====================
        model = TimeGAN(opt, ori_data_norm)

        # 直接训练，不添加任何自定义代码
        print("[INFO] 开始训练...")
        model.train()

        # 保存生成数据
        synthetic_data = model.generated_data
        save_path = os.path.join(opt.outf, f"{data_name}_synthetic_final.npy")
        os.makedirs(opt.outf, exist_ok=True)
        np.save(save_path, synthetic_data)

        print(f"✅ [{data_name}] 训练完成！已保存：{synthetic_data.shape}")

if __name__ == "__main__":
    train()