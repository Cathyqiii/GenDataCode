import os
import pickle

import yaml
import numpy as np

SCALER_FNAME = "scaler.pkl"

from TSlib.lib.dataloader import (
    fit_transform_scaler,
    transform_scaler,
    inverse_MinMaxScaler,
)

# 全局变量定义（修复未定义报错）
SCALER_INFO = {}


def load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded


def load_data(data_dir: str, dataset: str) -> np.ndarray:
    """
    Load data from a dataset located in a directory.

    Args:
        data_dir (str): The directory where the dataset is located.
        dataset (str): The name of the dataset file (without the .npz extension).

    Returns:
        np.ndarray: The loaded dataset.
    """
    return get_npz_data(os.path.join(data_dir, f"{dataset}.npz"))


def save_data(data: np.ndarray, output_file: str) -> None:
    """
    Save data to a .npz file.

    Args:
        data (np.ndarray): The data to save.
        output_file (str): The path to the .npz file to save the data to.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, data=data)


def get_npz_data(input_file: str) -> np.ndarray:
    """
    Load data from a .npz file.

    Args:
        input_file (str): The path to the .npz file.

    Returns:
        np.ndarray: The data array extracted from the .npz file.
    """
    loaded = np.load(input_file)
    return loaded["data"]  # 修复：必须取 data 键，否则返回的是NpzFile


def split_data(
        data: np.ndarray, valid_perc: float, shuffle: bool = True, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the data into training and validation sets.

    Args:
        data (np.ndarray): The dataset to split.
        valid_perc (float): The percentage of data to use for validation.
        shuffle (bool, optional): Whether to shuffle the data before splitting.
                                  Defaults to True.
        seed (int, optional): The random seed to use for shuffling the data.
                              Defaults to 123.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the training data and
                                       validation data arrays.
    """
    N = data.shape[0]
    N_train = int(N * (1 - valid_perc))

    if shuffle:
        np.random.seed(seed)
        data = data.copy()
        np.random.shuffle(data)

    train_data = data[:N_train]
    valid_data = data[N_train:]
    return train_data, valid_data


class MinMaxScaler:
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """

    def fit_transform(self, data):
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


def inverse_transform_data(norm_data, scaler_info):
    return inverse_MinMaxScaler(norm_data, scaler_info["data_name"])


def scale_data(train_data, valid_data, data_name):
    global SCALER_INFO  # 声明使用全局变量

    scaled_train_data = fit_transform_scaler(train_data, data_name)
    scaled_valid_data = transform_scaler(valid_data, data_name)

    # 修复：从外部scaler获取 min/max 并存储
    SCALER_INFO[data_name] = {
        "min": np.min(train_data, axis=0),
        "max": np.max(train_data, axis=0)
    }

    # 返回归一化信息（用于保存）
    scaler_info = {
        "min": SCALER_INFO[data_name]["min"],
        "max": SCALER_INFO[data_name]["max"],
        "data_name": data_name
    }
    return scaled_train_data, scaled_valid_data, scaler_info


def save_scaler(scaler_info, save_dir: str) -> None:
    """
    Save normalization (scaler) information to a .npy file.

    Args:
        scaler_info: The normalization information (e.g., min/max values) to be saved.
        save_dir (str): The path to the directory where the scaler information will be saved.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "scaler_info.npy")
    np.save(save_path, scaler_info)
    print(f"[INFO] 归一化信息已保存到: {save_path}")


def load_scaler(dir_path: str) -> MinMaxScaler:
    """
    Load a MinMaxScaler from a file.

    Args:
        dir_path (str): The path to the file from which the scaler will be loaded.

    Returns:
        MinMaxScaler: The loaded scaler.
    """
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "rb") as file:
        scaler = pickle.load(file)
    return scaler