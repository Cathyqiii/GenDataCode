#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加路径到Python环境
sys.path.append(r"D:\GenDataCode")

# ----------------------------------------------------------------------
# Project imports
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Utils.metric_utils import visualization
# 导入全局统一配置文件（核心：消除冗余）
from TSlib.lib.dataset_config import GENERATE_TIMES_MAP, standardize_name
from TSlib.lib.dataloader import inverse_MinMaxScaler


# ----------------------------------------------------------------------
# Utilities
def get_device(gpu_id: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def robust_build_dataloader(build_fn, configs, args):
    return build_fn(configs, args)


# ----------------------------------------------------------------------
# 1. Training
def run_train(configs, args, device):
    print("[INFO] Running Diffusion-TS training")
    # 使用原始DataLoader构建函数
    dl_info = build_dataloader(configs, args)
    dataset = dl_info["dataset"]

    model = instantiate_from_config(configs["model"]).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

    # 执行训练
    trainer.train()

    print("[INFO] Training finished")


# ----------------------------------------------------------------------
# 2. Unconditional sampling
def run_sampling(configs, args, device):
    print("[INFO] Running unconditional sampling")
    # 使用原始DataLoader构建函数
    dl_info = build_dataloader(configs, args)
    dataset = dl_info["dataset"]
    data_name = args.dataset_name

    # 标准化数据集名称（兼容大小写/特殊字符）
    std_data_name = standardize_name(data_name)

    model = instantiate_from_config(configs["model"]).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)
    trainer.load(args.milestone)

    seq_len = dataset.window
    feat_dim = dataset.var_num

    # 取小样本集数量
    real_num = len(dataset.train_data_g)

    # ===================== 全局配置，无冗余 =====================
    generate_times = GENERATE_TIMES_MAP[std_data_name]
    generate_num = real_num * generate_times
    # ================================================================

    # 采样（原始尺度数据）
    print(f"[INFO] {data_name} 原始样本数: {real_num} | 生成倍数: {generate_times} | 总生成: {generate_num}")
    samples = trainer.sample(num=generate_num, size_every=2001, shape=[seq_len, feat_dim])

    # 使用原始数据可视化
    ori_data = dataset.raw_data

    # 可视化（原始尺度）
    samples_dir = os.path.join(args.save_dir, "samples")
    visualization(ori_data, samples, analysis="pca", compare=ori_data.shape[0], save_path=samples_dir)
    visualization(ori_data, samples, analysis="tsne", compare=ori_data.shape[0], save_path=samples_dir)

    # ================== 🔧 关键修复：反归一化 ==================
    print("\n[INFO] 进行反归一化...")
    samples_denorm = inverse_MinMaxScaler(samples, data_name)

    print(f"[INFO] 反归一化前 - 数据范围: [{samples.min():.6f}, {samples.max():.6f}]")
    print(f"[INFO] 反归一化后 - 数据范围: [{samples_denorm.min():.6f}, {samples_denorm.max():.6f}]")
    print(f"[INFO] 原始数据范围: [{ori_data.min():.6f}, {ori_data.max():.6f}]")

    # 保存反归一化的采样结果
    save_path = os.path.join(args.save_dir, f"{data_name}.npy")
    np.save(save_path, samples_denorm)  # ✅ 保存反归一化后的数据
    print(f"[INFO] Sampling results saved to {save_path}")
    print(f"[INFO] Visualization images saved to {samples_dir}")


# ----------------------------------------------------------------------
# 3. Conditional generation (infill)
def run_infill(configs, args, device):
    print("[INFO] Running conditional generation (infill)")
    dl_info = robust_build_dataloader(build_dataloader_cond, configs, args)
    dataloader = dl_info["dataloader"]
    dataset = dl_info["dataset"]
    data_name = configs["dataloader"]["test_dataset"]["params"]["name"]

    model = instantiate_from_config(configs["model"]).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)
    trainer.load(args.milestone)

    coef = configs["dataloader"]["test_dataset"]["coefficient"]
    stepsize = configs["dataloader"]["test_dataset"]["step_size"]
    sampling_steps = configs["dataloader"]["test_dataset"]["sampling_steps"]
    seq_len = dataset.window
    feat_dim = dataset.var_num

    # 填充生成（原始尺度数据）
    samples, ori_data, masks = trainer.restore(dataloader, [seq_len, feat_dim], coef, stepsize,
                                               sampling_steps)

    # 可视化（原始尺度）
    plot_infill_results(samples, ori_data, masks,
                        save_path=os.path.join(args.save_dir, "diffusion_ts_infill.png"))

    # 保存原始尺度结果
    save_results(args.save_dir, samples, ori_data, masks, seq_len, feat_dim)
    print("[INFO] Infill results saved (原始尺度)")


# ----------------------------------------------------------------------
# Visualization helpers
def plot_infill_results(samples, ori_data, masks, save_path):
    plt.rcParams["font.size"] = 12
    feat_dim = min(ori_data.shape[2], 5)

    fig, axes = plt.subplots(feat_dim, 1, figsize=(12, 3 * feat_dim))
    if feat_dim == 1:
        axes = [axes]

    for i in range(feat_dim):
        axes[i].plot(samples[0, :, i], label="Diffusion-TS", color="g")
        axes[i].plot(
            ori_data[0, :, i] * masks[0, :, i],
            label="Observed",
            linestyle="None",
            marker="o",
            color="b",
        )
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ----------------------------------------------------------------------
# Saving helpers
def save_results(save_dir, samples, ori_data, masks, seq_len, feat_dim):
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 保存原始尺度数据 ✅ 修复：results → results_dir
    np.save(os.path.join(results_dir, "samples.npy"), samples)
    np.save(os.path.join(results_dir, "ori_data.npy"), ori_data)
    np.save(os.path.join(results_dir, "masks.npy"), masks)

    metadata = {
        "seq_length": seq_len,
        "feature_dim": feat_dim,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_scale": "original scale"  # 标注数据尺度为原始尺度
    }

    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ----------------------------------------------------------------------
# Argument parser
def parse_args_etth1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ETTh1")
    parser.add_argument("--config_path", type=str, default="./Config/etth1.yaml", help="Config YAML file")
    parser.add_argument("--save_dir", type=str, default="./output/ETTh1", help="Save directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--mode", type=str, default="sample", choices=["train", "sample", "infill"], help="Run mode")
    parser.add_argument("--missing_ratio", type=float, default=0.5)
    parser.add_argument("--milestone", type=int, default=10)
    return parser.parse_args()


def parse_args_etth2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ETTh2")
    parser.add_argument("--config_path", type=str, default="./Config/etth2.yaml", help="Config YAML file")
    parser.add_argument("--save_dir", type=str, default="./output/ETTh2", help="Save directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "infill"], help="Run mode")
    parser.add_argument("--missing_ratio", type=float, default=0.5)
    parser.add_argument("--milestone", type=int, default=10)
    return parser.parse_args()


def parse_args_AirQuality_bj():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="AirQuality(bj)")
    parser.add_argument("--config_path", type=str, default="./Config/AirQuality(bj).yaml", help="Config YAML file")
    parser.add_argument("--save_dir", type=str, default="./output/AirQuality(bj)", help="Save directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "infill"], help="Run mode")
    parser.add_argument("--missing_ratio", type=float, default=0.5)
    parser.add_argument("--milestone", type=int, default=10)
    return parser.parse_args()


def parse_args_AirQuality_Italian():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="AirQuality(Italian)")
    parser.add_argument("--config_path", type=str, default="./Config/AirQuality(Italian).yaml", help="Config YAML file")
    parser.add_argument("--save_dir", type=str, default="./output/AirQuality(Italian)", help="Save directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "infill"], help="Run mode")
    parser.add_argument("--missing_ratio", type=float, default=0.5)
    parser.add_argument("--milestone", type=int, default=10)
    return parser.parse_args()


def parse_args_Traffic():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Traffic")
    parser.add_argument("--config_path", type=str, default="./Config/Traffic.yaml", help="Config YAML file")
    parser.add_argument("--save_dir", type=str, default="./output/Traffic", help="Save directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "infill"], help="Run mode")
    parser.add_argument("--missing_ratio", type=float, default=0.5)
    parser.add_argument("--milestone", type=int, default=10)
    return parser.parse_args()


def parse_args_CMAPSS():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="FD001")
    parser.add_argument("--config_path", type=str, default="./Config/FD001.yaml", help="Config YAML file")
    parser.add_argument("--save_dir", type=str, default="./output/FD001", help="Save directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "infill"], help="Run mode")
    parser.add_argument("--missing_ratio", type=float, default=0.5)
    parser.add_argument("--milestone", type=int, default=10)
    return parser.parse_args()


# ----------------------------------------------------------------------
# Main entry
if __name__ == "__main__":
    dataset_name = "AirQuality(bj)"

    if dataset_name == "ETTh1":
        args = parse_args_etth1()
    elif dataset_name == "ETTh2":
        args = parse_args_etth2()
    elif dataset_name == "AirQuality(bj)":
        args = parse_args_AirQuality_bj()
    elif dataset_name == "AirQuality(Italian)":
        args = parse_args_AirQuality_Italian()
    elif dataset_name == "Traffic":
        args = parse_args_Traffic()
    elif dataset_name in ["FD001", "FD002", "FD003", "FD004"]:
        args = parse_args_CMAPSS()

    os.makedirs(args.save_dir, exist_ok=True)
    configs = load_yaml_config(args.config_path)
    device = get_device(args.gpu)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Config: {args.config_path}")

    if args.mode == "train":
        run_train(configs, args, device)
    elif args.mode == "sample":
        run_sampling(configs, args, device)
    elif args.mode == "infill":
        run_infill(configs, args, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("[INFO] Diffusion-TS pipeline finished")