"""
Reimplement TimeVAE Pipeline in a unified experiment-style entry.
【已修复】数据划分错误 + 重复归一化/反归一化问题
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import warnings
import argparse

warnings.filterwarnings("ignore")
sys.path.append(r"D:\GenDataCode")
import numpy as np

from config_loader import ConfigLoader

from lib.data import (
    load_yaml_file,
    split_data,
    save_data,
)

from lib.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    load_vae_model,
    get_posterior_samples,
    get_prior_samples,
)

from visualize import (
    plot_samples,
    plot_latent_space_samples,
    visualize_and_save_tsne,
)
from TSlib.lib.dataloader import real_data_loading

# ------------------------------------------------------------------------------
# 数据集-特征模式映射
def get_feature_mode_by_dataset(dataset_name):
    mode_mapping = {
        "etth1": "S",
        "etth2": "S",
        "AirQuality(bj)": "M",
        "AirQuality(Italian)": "M",
        "Traffic": "S",
        "FD001": "MS",
        "FD002": "MS",
        "FD003": "MS",
        "FD004": "MS"
    }
    return mode_mapping.get(dataset_name)

# ------------------------------------------------------------------------------
# 主实验函数
def run_timevae(dataset_name: str, vae_type: str, feature_mode: str):
    # 0. 配置文件选择
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(current_dir, "src", "config")

    if dataset_name == "etth1":
        config_path = os.path.join(config_dir, "etth1.conf")
    elif dataset_name == "etth2":
        config_path = os.path.join(config_dir, "etth2.conf")
    elif dataset_name in ["AirQuality(bj)", "AirQuality(Italian)", "Traffic"]:
        config_path = os.path.join(config_dir, f"{dataset_name}.conf")
    elif dataset_name in ["FD001", "FD002", "FD003", "FD004"]:
        config_path = os.path.join(config_dir, "C-MAPSS.conf")
    else:
        raise ValueError(f"Unknown dataset for config loading: {dataset_name}")

    opt = ConfigLoader(config_path)

    # 1. 加载数据（已归一化）
    print(f"[INFO] Loading dataset: {dataset_name} (feature_mode: {feature_mode})")
    train_data, val_data, test_data, train_data_g = real_data_loading(dataset_name, feature_mode)

    scaled_train_data = train_data.astype(np.float32)
    scaled_valid_data = val_data.astype(np.float32)

    print(f"[INFO] Train data shape: {scaled_train_data.shape}")

    # 4. 加载超参数
    hyperparams_all = load_yaml_file(opt.HYPERPARAMETERS_FILE_PATH)
    hyperparameters = hyperparams_all[vae_type]
    print(f"[INFO] Hyperparameters: {hyperparameters}")

    # 5. 实例化模型
    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters
    )
    print(f"[INFO] Model instantiated: {vae_type}")

    # 6. 训练模型
    print("[INFO] Training VAE...")
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=hyperparameters.get("max_epochs", 200),
        verbose=1,
    )

    # 7. 保存模型
    model_save_dir = os.path.join(opt.MODELS_DIR, dataset_name)
    os.makedirs(model_save_dir, exist_ok=True)
    save_vae_model(vae_model, model_save_dir)
    print(f"[INFO] Model saved to: {model_save_dir}")

    # 8. 重建可视化
    x_recon = get_posterior_samples(vae_model, scaled_train_data)
    plot_samples(
        samples1=scaled_train_data,
        samples1_name="Original (Train)",
        samples2=x_recon,
        samples2_name="Reconstructed",
        num_samples=5,
    )

    # 9. 先验采样
    prior_samples = get_prior_samples(
        vae_model,
        dataset_name=dataset_name,
        small_sample_num=scaled_train_data.shape[0]
    )

    # 11. 保存生成数据（不做反归一化！）
    output_base_dir = "output"
    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # 直接保存归一化域数据，不做反归一化！
    final_gen_data = prior_samples

    npy_file_path = os.path.join(dataset_output_dir, f"{dataset_name}.npy")
    np.save(npy_file_path, final_gen_data)

    npz_file_path = os.path.join(dataset_output_dir, f"{vae_type}_{dataset_name}_prior_samples.npz")
    save_data(data=final_gen_data, output_file=npz_file_path)

    print(f"[INFO] 生成数据已保存（归一化域）")
    print(f"[INFO] .npy file: {npy_file_path}")

    # 12. 潜在空间可视化
    if hyperparameters.get("latent_dim", None) == 2:
        plot_latent_space_samples(vae=vae_model, n=8, figsize=(15, 15))

    print("[INFO] TimeVAE experiment finished successfully.")
    return scaled_train_data, prior_samples

# ------------------------------------------------------------------------------
# 命令行选择数据集&模型
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run TimeVAE Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="etth2",
        choices=["etth1", "etth2", "AirQuality(bj)", "AirQuality(Italian)", "Traffic", "FD001", "FD002", "FD003", "FD004"],
        help="Dataset name"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="timeVAE",
        choices=["vae_dense", "vae_conv", "timeVAE"],
        help="VAE models type"
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        default=None,
        choices=["M", "S", "MS"],
        help="Feature mode (auto-selected if not specified)"
    )

    args = parser.parse_args()
    args.dataset = 'etth2'
    args.models = 'timeVAE'

    if args.feature_mode is not None:
        feature_mode = args.feature_mode
    else:
        feature_mode = get_feature_mode_by_dataset(args.dataset)

    print(f"[INFO] Using feature_mode: {feature_mode} for dataset {args.dataset}")

    run_timevae(dataset_name=args.dataset, vae_type=args.models, feature_mode=feature_mode)