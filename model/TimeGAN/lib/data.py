#!/usr/bin/env python
# encoding: utf-8
import os
import numpy as np
import sys

# 全局根目录计算
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, '../../../'))
sys.path.append(_repo_root)

def MinMaxScaler(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    denominator = max_val - min_val + 1e-7
    norm_data = (data - min_val) / denominator
    return norm_data, min_val, max_val


def real_data_loading(data_name):
    """
    加载时序数据集
    :param data_name: 数据集名称
    :return: train_data, val_data, test_data, train_data_g
    """
    # 输入校验
    supported_datasets = ["etth1", "etth2", 'AirQuality(bj)', 'AirQuality(Italian)', 'Traffic', "FD001"]
    assert data_name in supported_datasets, f"不支持的数据集: {data_name}"

    ROOT_DIR = _repo_root
    clusters = []

    # 加载不同数据集
    if data_name in ["etth1", "etth2"]:
        fpath = os.path.join(ROOT_DIR, f'dataProcess/ETT/output/ETT{data_name[-2:]}')
        for i in [0, 1, 2]:
            file_path = os.path.join(fpath, f'cluster_{i}.npy')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{data_name}文件缺失: {file_path}")
            data = np.load(file_path, allow_pickle=True)
            clusters.append(data)

    elif data_name in ['AirQuality(bj)', 'AirQuality(Italian)']:
        fpath = os.path.join(ROOT_DIR, f'dataProcess/{data_name}/output')
        for i in [0, 1, 2]:
            file_path = os.path.join(fpath, f'cluster_{i}.npy')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{data_name}文件缺失: {file_path}")
            data = np.load(file_path, allow_pickle=True)
            clusters.append(data)

    elif data_name == "Traffic":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/Traffic/output')
        weather_files = ['clear.npy', 'clouds.npy', 'drizzle.npy', 'haze.npy', 'mist.npy', 'rain.npy', 'snow.npy']
        for fname in weather_files:
            file_path = os.path.join(fpath, fname)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Traffic文件缺失: {file_path}")
            data = np.load(file_path, allow_pickle=True)
            data = data[:, :, -1:]
            clusters.append(data)

    elif data_name == "FD001":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/C-MAPSS/output')
        for fname in ['degraded.npy', 'normal.npy']:
            file_path = os.path.join(fpath, fname)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"FD001文件缺失: {file_path}")
            data = np.load(file_path, allow_pickle=True)
            clusters.append(data)

    # 数据划分
    orig_data = np.concatenate(clusters, axis=0)
    orig_data,min,max = MinMaxScaler(orig_data)
    split_idx = np.cumsum([c.shape[0] for c in clusters[:-1]])
    clusters = np.split(orig_data, split_idx, axis=0)

    train_list, val_list, test_list = [], [], []
    for c in clusters:
        n = c.shape[0]
        t = int(n * 0.1)
        train_list.append(c[:int(n * 0.7)])
        val_list.append(c[-3 * t:-2 * t])
        test_list.append(c[-2 * t:])

    train_data = np.concatenate(train_list, axis=0)
    val_data = np.concatenate(val_list, axis=0)
    test_data = np.concatenate(test_list, axis=0)

    # 生成任务专用数据
    train_data_g = None
    if data_name in ["etth1", "etth2", 'AirQuality(bj)', 'AirQuality(Italian)']:
        cluster_sizes = [len(c) for c in clusters]
        minority_idx = np.argmin(cluster_sizes)
        train_data_g = train_list[minority_idx]

    elif data_name == "Traffic":
        cluster_sizes = [len(c) for c in clusters]
        sorted_idx = np.argsort(cluster_sizes)
        small_idx = sorted_idx[:4]
        train_data_g = np.concatenate([train_list[i] for i in small_idx], axis=0)

    elif data_name == "FD001":
        cluster_sizes = [len(c) for c in clusters]
        minority_idx = np.argmin(cluster_sizes)
        train_data_g = train_list[minority_idx]

    return train_data, val_data, test_data, train_data_g


def load_data(opt):
    """生成任务专用数据加载入口"""
    _, _, _, ori_data = real_data_loading(opt.data_name)
    return ori_data


def batch_generator(data, time, batch_size):
    """
    生成训练用mini-batch数据
    :param data: 输入数据 (N, seq_len, D)
    :param time: 时间步标识 (N, seq_len)
    :param batch_size: 批次大小
    :return: 批次数据, 批次时间步
    """
    no = len(data)
    if no < batch_size:
        raise ValueError(f"数据量 {no} 小于批次大小 {batch_size}")

    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def gen_data_loading(data_name, gen_mode):
    """加载生成的合成数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(current_dir, '../')))
    fpath = os.path.join(ROOT_DIR, f'model/{gen_mode}/output/{data_name}')
    g_data = np.load(f'{fpath}/{data_name}.npy', allow_pickle=True)
    return g_data