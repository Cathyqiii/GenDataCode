import random
from typing import Union, List, Optional
import os
import sys
import torch
import numpy as np

# 获取当前文件所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将其添加到模块搜索路径的开头
sys.path.insert(0, current_dir)

from vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from timevae import TimeVAE


def set_seeds(seed: int = 111) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for Python built-in random module
    random.seed(seed)


def instantiate_vae_model(
    vae_type: str, sequence_length: int, feature_dim: int, batch_size: int, **kwargs
) -> Union[VAE_Dense, VAE_Conv, TimeVAE]:
    set_seeds(seed=123)

    if vae_type == "vae_dense":
        vae = VAE_Dense(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    elif vae_type == "vae_conv":
        vae = VAE_Conv(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    elif vae_type == "timeVAE":
        vae = TimeVAE(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unrecognized models type [{vae_type}]. "
            "Please choose from vae_dense, vae_conv, timeVAE."
        )

    return vae


def train_vae(vae, train_data, max_epochs, verbose=0):
    vae.fit_on_data(train_data, max_epochs, verbose)


def save_vae_model(vae, dir_path: str) -> None:
    vae.save(dir_path)


def load_vae_model(vae_type: str, dir_path: str) -> Union[VAE_Dense, VAE_Conv, TimeVAE]:
    if vae_type == "vae_dense":
        vae = VAE_Dense.load(dir_path)
    elif vae_type == "vae_conv":
        vae = VAE_Conv.load(dir_path)
    elif vae_type == "timeVAE":
        vae = TimeVAE.load(dir_path)
    else:
        raise ValueError(
            f"Unrecognized models type [{vae_type}]. "
            "Please choose from vae_dense, vae_conv, timeVAE."
        )

    return vae


def get_posterior_samples(vae, data):
    return vae.predict(data)


def get_prior_samples(vae, dataset_name: str, small_sample_num: int):
    """
    Get prior samples from the VAE models.
    内置数据集专属生成倍数，自动计算生成数量
    """
    generate_times_mapping = {
        "etth1": 5,
        "etth2": 6,
        "AirQuality(bj)": 2,
        "AirQuality(Italian)": 2,
        "FD001": 2,
        "FD002": 2,
        "FD003": 2,
        "FD004": 2,
        "Traffic": 2
    }

    generate_times = generate_times_mapping[dataset_name]
    num_samples = small_sample_num * generate_times

    return vae.get_prior_samples(num_samples=num_samples)