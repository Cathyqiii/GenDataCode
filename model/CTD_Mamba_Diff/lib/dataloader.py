import torch
import os
import numpy as np

# 全局配置：支持 数据集 / 簇 / 生成器 三种归一化记录
SCALER_INFO = {
    "etth1": {"min": None, "max": None},
    "etth2": {"min": None, "max": None},
    "AirQuality(bj)": {"min": None, "max": None},
    "AirQuality(Italian)": {"min": None, "max": None},
    "Traffic": {"min": None, "max": None},
    "FD001": {"min": None, "max": None}, "FD002": {"min": None, "max": None},
    "FD003": {"min": None, "max": None}, "FD004": {"min": None, "max": None},
    "etth1_g": {"min": None, "max": None}, "etth2_g": {"min": None, "max": None},
    "AirQuality(bj)_g": {"min": None, "max": None}, "AirQuality(Italian)_g": {"min": None, "max": None},
    "Traffic_g": {"min": None, "max": None},
    "FD001_g": {"min": None, "max": None}, "FD002_g": {"min": None, "max": None},
    "FD003_g": {"min": None, "max": None}, "FD004_g": {"min": None, "max": None},
}


def clean_numpy_data(data):
    cleaned_data = np.empty_like(data, dtype=np.float32)
    for idx in np.ndindex(data.shape):
        try:
            cleaned_data[idx] = float(data[idx])
        except (ValueError, TypeError):
            cleaned_data[idx] = np.nan
    mean_val = np.nanmean(cleaned_data)
    cleaned_data = np.nan_to_num(cleaned_data, nan=mean_val)
    return cleaned_data


def MinMaxScaler(data):
    if len(data.shape) == 3:
        min_val = np.min(data, axis=(0, 1), keepdims=True)
        max_val = np.max(data, axis=(0, 1), keepdims=True)
    else:
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)

    denominator = max_val - min_val
    denominator[denominator < 1e-7] = 1.0
    norm_data = 2 * ((data - min_val) / denominator) - 1

    min_val = min_val.squeeze()
    max_val = max_val.squeeze()
    return norm_data, min_val, max_val


def set_scaler_info(data_name, min_val, max_val):
    if data_name not in SCALER_INFO:
        raise ValueError(f"不支持的数据集: {data_name}")
    SCALER_INFO[data_name]["min"] = min_val
    SCALER_INFO[data_name]["max"] = max_val


def inverse_MinMaxScaler(norm_data, data_name):
    if data_name not in SCALER_INFO:
        raise ValueError(f"不支持的数据集: {data_name}")
    min_val = SCALER_INFO[data_name]["min"]
    max_val = SCALER_INFO[data_name]["max"]

    if min_val is None or max_val is None:
        raise ValueError(f"未找到归一化信息: {data_name}")

    if len(norm_data.shape) == 3:
        min_val = min_val.reshape(1, 1, -1)
        max_val = max_val.reshape(1, 1, -1)

    denominator = max_val - min_val
    denominator[denominator < 1e-7] = 1.0
    orig_data = (norm_data + 1) / 2 * denominator + min_val
    return orig_data


def fit_transform_scaler(data, data_name):
    norm_data, min_val, max_val = MinMaxScaler(data)
    set_scaler_info(data_name, min_val, max_val)
    return norm_data


def transform_scaler(data, data_name):
    if data_name not in SCALER_INFO:
        raise ValueError(f"不支持的数据集: {data_name}")
    min_val = SCALER_INFO[data_name]["min"]
    max_val = SCALER_INFO[data_name]["max"]

    if len(data.shape) == 3:
        min_val = min_val.reshape(1, 1, -1)
        max_val = max_val.reshape(1, 1, -1)

    denominator = max_val - min_val
    denominator[denominator < 1e-7] = 1.0
    norm_data = 2 * ((data - min_val) / denominator) - 1
    return norm_data


def full_real_data_loading(data_name):
    assert data_name in ["etth1", "etth2", 'AirQuality(bj)', 'AirQuality(Italian)', 'Traffic',
                         "FD001", "FD002", "FD003", "FD004"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(current_dir, '../../')))

    full_labels = np.array([])

    # ==================== 1. 加载原始文件（所有数据集） ====================
    if data_name == "etth1":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/ETT/output/ETTh1')
        cluster_0 = np.load(f'{fpath}/cluster_0.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/cluster_1.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/cluster_2.npy', allow_pickle=True)

    elif data_name == "etth2":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/ETT/output/ETTh2')
        cluster_0 = np.load(f'{fpath}/cluster_0.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/cluster_1.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/cluster_2.npy', allow_pickle=True)

    elif data_name == "AirQuality(bj)":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/AirQuality(bj)/output')
        cluster_0 = np.load(f'{fpath}/cluster_0.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/cluster_1.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/cluster_2.npy', allow_pickle=True)

    elif data_name == "AirQuality(Italian)":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/AirQuality(Italian)/output')
        cluster_0 = np.load(f'{fpath}/cluster_0.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/cluster_1.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/cluster_2.npy', allow_pickle=True)

    elif data_name == "Traffic":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/Traffic/output')
        cluster_0 = clean_numpy_data(np.load(f'{fpath}/clear.npy', allow_pickle=True))
        cluster_1 = clean_numpy_data(np.load(f'{fpath}/clouds.npy', allow_pickle=True))
        cluster_2 = clean_numpy_data(np.load(f'{fpath}/drizzle.npy', allow_pickle=True))
        cluster_3 = clean_numpy_data(np.load(f'{fpath}/haze.npy', allow_pickle=True))
        cluster_4 = clean_numpy_data(np.load(f'{fpath}/mist.npy', allow_pickle=True))
        cluster_5 = clean_numpy_data(np.load(f'{fpath}/rain.npy', allow_pickle=True))
        cluster_6 = clean_numpy_data(np.load(f'{fpath}/snow.npy', allow_pickle=True))

        label_0 = np.array(['clear'] * cluster_0.shape[0])
        label_1 = np.array(['clouds'] * cluster_1.shape[0])
        label_2 = np.array(['drizzle'] * cluster_2.shape[0])
        label_3 = np.array(['haze'] * cluster_3.shape[0])
        label_4 = np.array(['mist'] * cluster_4.shape[0])
        label_5 = np.array(['rain'] * cluster_5.shape[0])
        label_6 = np.array(['snow'] * cluster_6.shape[0])
        full_labels = np.concatenate([label_0, label_1, label_2, label_3, label_4, label_5, label_6])

    # ==================== 🔧 关键修复：FD 数据集按名称加载对应文件 ====================
    elif data_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        fpath = os.path.join(ROOT_DIR, 'dataProcess/C-MAPSS/output')
        # ✅ 修复前：只加载无编号前缀的 degraded.npy / normal.npy（实际全是FD001）
        # ✅ 修复后：根据 data_name 加载 FD001_degraded.npy / FD002_degraded.npy 等
        cluster_0 = np.load(f'{fpath}/{data_name}_degraded.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/{data_name}_normal.npy', allow_pickle=True)
        label_0 = np.array(['degraded'] * cluster_0.shape[0])
        label_1 = np.array(['normal'] * cluster_1.shape[0])
        full_labels = np.concatenate([label_0, label_1])

    # ==================== 2. 数据划分（完全对齐 real_data_loading 逻辑） ====================
    if data_name in ["etth1", "etth2", 'AirQuality(bj)', 'AirQuality(Italian)']:
        orig_data = np.concatenate([cluster_0, cluster_1, cluster_2], axis=0)
        orig_data, min_val, max_val = MinMaxScaler(orig_data)
        set_scaler_info(data_name, min_val, max_val)

        cluster_sizes = [cluster_0.shape[0], cluster_1.shape[0], cluster_2.shape[0]]
        cluster_0 = orig_data[:cluster_0.shape[0]]
        cluster_1 = orig_data[cluster_0.shape[0]:cluster_0.shape[0] + cluster_1.shape[0]]
        cluster_2 = orig_data[cluster_0.shape[0] + cluster_1.shape[0]:]

        c0_test = int(cluster_0.shape[0] * 0.1)
        c1_test = int(cluster_1.shape[0] * 0.1)
        c2_test = int(cluster_2.shape[0] * 0.1)

        test_data = np.concatenate([cluster_0[-c0_test * 2:], cluster_1[-c1_test * 2:], cluster_2[-c2_test * 2:]],
                                   axis=0)
        val_data = np.concatenate([cluster_0[-c0_test * 3:-c0_test * 2],
                                   cluster_1[-c1_test * 3:-c1_test * 2],
                                   cluster_2[-c2_test * 3:-c2_test * 2]], axis=0)

        minority = np.argmin(cluster_sizes)
        ratios = [1.0, 1.0, 1.0]
        ratios[minority] = 0.85

        c0_train_num = int(cluster_0.shape[0] * 0.7 * ratios[0])
        c1_train_num = int(cluster_1.shape[0] * 0.7 * ratios[1])
        c2_train_num = int(cluster_2.shape[0] * 0.7 * ratios[2])

        c0_train_idx = slice(0, c0_train_num)
        c1_train_idx = slice(0, c1_train_num)
        c2_train_idx = slice(0, c2_train_num)

        train_data = np.concatenate([cluster_0[c0_train_idx], cluster_1[c1_train_idx], cluster_2[c2_train_idx]], axis=0)

        if minority == 0:
            train_data_g = cluster_0[c0_train_idx]
        elif minority == 1:
            train_data_g = cluster_1[c1_train_idx]
        else:
            train_data_g = cluster_2[c2_train_idx]

    elif data_name == "Traffic":
        orig_data = np.concatenate([cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6],
                                   axis=0)
        orig_data, min_val, max_val = MinMaxScaler(orig_data)
        set_scaler_info(data_name, min_val, max_val)

        clusters = [cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6]
        sizes = [c.shape[0] for c in clusters]
        split_idx = np.cumsum(sizes[:-1])
        cluster_0 = orig_data[:split_idx[0]]
        cluster_1 = orig_data[split_idx[0]:split_idx[1]]
        cluster_2 = orig_data[split_idx[1]:split_idx[2]]
        cluster_3 = orig_data[split_idx[2]:split_idx[3]]
        cluster_4 = orig_data[split_idx[3]:split_idx[4]]
        cluster_5 = orig_data[split_idx[4]:split_idx[5]]
        cluster_6 = orig_data[split_idx[5]:]
        clusters = [cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6]

        train_list, val_list, test_list = [], [], []
        for c in clusters:
            n = c.shape[0]
            t = int(n * 0.1)
            train_list.append(c[:int(n * 0.7)])
            val_list.append(c[-3 * t:-2 * t])
            test_list.append(c[-2 * t:])
        val_data = np.concatenate(val_list, axis=0)
        test_data = np.concatenate(test_list, axis=0)

        train_sampled = []
        for c in train_list:
            n = c.shape[0]
            sample_n = max(1, int(n * 0.7))
            idx = slice(0, sample_n)
            train_sampled.append(c[idx])
        train_data = np.concatenate(train_sampled, axis=0)

        sizes = [c.shape[0] for c in clusters]
        sorted_idx = np.argsort(sizes)
        filtered_idx = sorted_idx[2:]
        selected_small_idx = filtered_idx[:4]
        selected_clusters = [clusters[i] for i in selected_small_idx]
        combined_clusters = np.concatenate(selected_clusters, axis=0)
        total_samples = combined_clusters.shape[0]
        keep_num = max(1, int(total_samples * 0.85))
        sample_idx = slice(0, keep_num)
        train_data_g = combined_clusters[sample_idx]

    # ==================== FD 划分逻辑（完全对齐 real_data_loading） ====================
    elif data_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        orig_data = np.concatenate([cluster_0, cluster_1], axis=0)
        orig_data, min_val, max_val = MinMaxScaler(orig_data)
        set_scaler_info(data_name, min_val, max_val)

        len0 = cluster_0.shape[0]
        len1 = cluster_1.shape[0]
        cluster_0 = orig_data[:len0]
        cluster_1 = orig_data[len0:]

        c0_test = int(len0 * 0.1)
        c1_test = int(len1 * 0.1)
        test_data = np.concatenate([cluster_0[-c0_test * 2:], cluster_1[-c1_test * 2:]], axis=0)
        val_data = np.concatenate([
            cluster_0[-c0_test * 3:-c0_test * 2],
            cluster_1[-c1_test * 3:-c1_test * 2]
        ], axis=0)

        # 多数类抽样比例 0.7，少数类全取
        majority = np.argmax([len0, len1])
        ratios = [1, 1]
        ratios[majority] = 0.7

        c0_train_g = int(len0 * 0.7 * ratios[0])
        c1_train_g = int(len1 * 0.7 * ratios[1])

        c0_train_idx = slice(0, c0_train_g)
        c1_train_idx = slice(0, c1_train_g)

        train_data = np.concatenate([cluster_0[c0_train_idx], cluster_1[c1_train_idx]], axis=0)
        # 生成集取多数类的前 c_train_g 条
        train_data_g = cluster_0[c0_train_idx] if majority == 0 else cluster_1[c1_train_idx]

        print(f"数据集 {data_name} 划分完成：")
        print(
            f"训练集: {train_data.shape}, 验证集: {val_data.shape}, 测试集: {test_data.shape}, 生成用训练集: {train_data_g.shape}")

    return train_data, val_data, test_data, train_data_g, orig_data, full_labels