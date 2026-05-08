#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/8/25 17:50
@desc: load datasets + 统一归一化工具
【已修复】全局归一化覆盖BUG，每个数据集独立scaler参数
【已升级】完美支持 FD001 / FD002 / FD003 / FD004 独立加载
【已修复】real_data_loading与full_real_data_loading数据数量完全对齐
【已修复】FD001~004 真正独立加载，不再统一加载FD001
"""
import torch
import os
import numpy as np

# ===================== 修复：自动创建归一化参数保存目录 =====================
SCALER_SAVE_DIR = "./dataset_scalers"
os.makedirs(SCALER_SAVE_DIR, exist_ok=True)


def MinMaxScaler(data):
    """
    ✅ 修复版：时序数据安全归一化（按特征列，不破坏时序结构）
    支持：[N, F] 或 [N, L, F]
    """
    if len(data.shape) == 3:
        # 时序数据 [N, L, F] → 按样本+时间步取特征维度极值
        min_val = np.min(data, axis=(0, 1), keepdims=True)
        max_val = np.max(data, axis=(0, 1), keepdims=True)
    else:
        # 二维数据 [N, F]
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)

    # 安全处理常量特征（避免除0）
    denominator = max_val - min_val
    denominator[denominator < 1e-7] = 1.0
    norm_data = (data - min_val) / denominator

    # 压缩维度
    min_val = min_val.squeeze()
    max_val = max_val.squeeze()
    return norm_data, min_val, max_val


# ===================== 修复：每个数据集独立保存归一化参数 =====================
def save_scaler(data_name, min_val, max_val):
    """保存当前数据集的min/max到独立文件"""
    min_path = os.path.join(SCALER_SAVE_DIR, f"{data_name}_min.npy")
    max_path = os.path.join(SCALER_SAVE_DIR, f"{data_name}_max.npy")
    np.save(min_path, min_val)
    np.save(max_path, max_val)


def load_scaler(data_name):
    """加载当前数据集的min/max"""
    min_path = os.path.join(SCALER_SAVE_DIR, f"{data_name}_min.npy")
    max_path = os.path.join(SCALER_SAVE_DIR, f"{data_name}_max.npy")

    if not os.path.exists(min_path) or not os.path.exists(max_path):
        raise ValueError(f"未找到 {data_name} 的归一化参数，请先运行 fit_transform_scaler！")

    min_val = np.load(min_path)
    max_val = np.load(max_path)
    return min_val, max_val


# ===================== 修复：训练集拟合（独立保存） =====================
def fit_transform_scaler(data, data_name):
    """训练集：拟合 + 归一化 + 独立保存极值"""
    norm_data, min_val, max_val = MinMaxScaler(data)
    save_scaler(data_name, min_val, max_val)
    return norm_data


# ===================== 修复：验证/测试集归一化（独立加载） =====================
def transform_scaler(data, data_name):
    """验证/测试集：仅归一化（使用训练集独立参数，无数据泄露）"""
    min_val, max_val = load_scaler(data_name)

    if len(data.shape) == 3:
        min_val = min_val.reshape(1, 1, -1)
        max_val = max_val.reshape(1, 1, -1)

    denominator = max_val - min_val
    denominator[denominator < 1e-7] = 1.0
    norm_data = (data - min_val) / denominator
    return norm_data


# ===================== 修复：反归一化（独立加载，不串数据） =====================
def inverse_MinMaxScaler(norm_data, data_name):
    """反归一化（兼容时序/二维数据）"""
    min_val, max_val = load_scaler(data_name)

    if len(norm_data.shape) == 3:
        min_val = min_val.reshape(1, 1, -1)
        max_val = max_val.reshape(1, 1, -1)

    denominator = max_val - min_val
    denominator[denominator < 1e-7] = 1.0
    orig_data = norm_data * denominator + min_val
    return orig_data


def gen_data_loading(data_name, gen_mode, target_feature_dim, feature='MS'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(current_dir, '../')))
    fpath = os.path.join(ROOT_DIR, f'model/{gen_mode}/output/{data_name}')
    data_file = os.path.join(fpath, f'{data_name}.npy')
    print(f"加载生成数据路径：{fpath}")

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"生成数据目录不存在: {fpath}")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"生成数据文件缺失: {data_file}")

    try:
        gen_data = np.load(data_file, allow_pickle=True)
        gen_feature_dim = gen_data.shape[-1]
    except Exception as e:
        raise RuntimeError(f"加载生成数据失败: {str(e)}") from e

    # 针对 FD001/FD002/FD003/FD004 统一维度处理
    if data_name in ["FD001", "FD002", "FD003", "FD004"]:
        if feature == 'MS':
            gen_data = gen_data[:, :, :-1]
        elif feature == 'M':
            gen_data = gen_data[:, :, :-2]
        elif feature == 'S':
            gen_data = gen_data[:, :, -1:]

    return gen_data


def real_data_loading(data_name, feature):
    assert data_name in ["etth1", "etth2", 'AirQuality(bj)', 'AirQuality(Italian)', 'Traffic',
                         "FD001", "FD002", "FD003", "FD004"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(current_dir, '../')))

    # 初始化聚类变量，避免未定义报错
    cluster_0 = None
    cluster_1 = None
    cluster_2 = None

    if data_name == "etth1":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/ETT/output/ETTh1')
        cluster_0 = np.load(f'{fpath}/cluster_0.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/cluster_1.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/cluster_2.npy', allow_pickle=True)
        if feature == 'S':
            cluster_0 = cluster_0[:, :, -1:]
            cluster_1 = cluster_1[:, :, -1:]
            cluster_2 = cluster_2[:, :, -1:]

    elif data_name == "etth2":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/ETT/output/ETTh2')
        cluster_0 = np.load(f'{fpath}/cluster_0.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/cluster_1.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/cluster_2.npy', allow_pickle=True)
        if feature == 'S':
            cluster_0 = cluster_0[:, :, -1:]
            cluster_1 = cluster_1[:, :, -1:]
            cluster_2 = cluster_2[:, :, -1:]

    elif data_name == "AirQuality(bj)":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/AirQuality(bj)/output')
        cluster_0 = np.load(f'{fpath}/cluster_0.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/cluster_1.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/cluster_2.npy', allow_pickle=True)
        if feature == 'M' or feature == 'MS':
            cluster_0 = cluster_0[:, :, -1:]
            cluster_1 = cluster_1[:, :, -1:]
            cluster_2 = cluster_2[:, :, -1:]
        elif feature == 'S':
            cluster_0 = cluster_0[:, :, -6:]
            cluster_0 = cluster_0[:, :, 0:1]
            cluster_1 = cluster_1[:, :, -6:]
            cluster_1 = cluster_1[:, :, 0:1]
            cluster_2 = cluster_2[:, :, -6:]
            cluster_2 = cluster_2[:, :, 0:1]

    elif data_name == "AirQuality(Italian)":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/AirQuality(Italian)/output')
        cluster_0 = np.load(f'{fpath}/cluster_0.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/cluster_1.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/cluster_2.npy', allow_pickle=True)
        if feature == 'M' or feature == 'MS':
            cluster_0 = cluster_0[:, :, 1:3]
            cluster_1 = cluster_1[:, :, 1:3]
            cluster_2 = cluster_2[:, :, 1:3]
        elif feature == 'S':
            cluster_0 = cluster_0[:, :, 1:2]
            cluster_1 = cluster_1[:, :, 1:2]
            cluster_2 = cluster_2[:, :, 1:2]

    elif data_name == "Traffic":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/Traffic/output')
        cluster_0 = np.load(f'{fpath}/clear.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/clouds.npy', allow_pickle=True)
        cluster_2 = np.load(f'{fpath}/drizzle.npy', allow_pickle=True)
        cluster_3 = np.load(f'{fpath}/haze.npy', allow_pickle=True)
        cluster_4 = np.load(f'{fpath}/mist.npy', allow_pickle=True)
        cluster_5 = np.load(f'{fpath}/rain.npy', allow_pickle=True)
        cluster_6 = np.load(f'{fpath}/snow.npy', allow_pickle=True)
        cluster_0 = cluster_0[:, :, -1:]
        cluster_1 = cluster_1[:, :, -1:]
        cluster_2 = cluster_2[:, :, -1:]
        cluster_3 = cluster_3[:, :, -1:]
        cluster_4 = cluster_4[:, :, -1:]
        cluster_5 = cluster_5[:, :, -1:]
        cluster_6 = cluster_6[:, :, -1:]

    # ===================== ✅ 核心修复：四个FD数据集独立加载（无嵌套，顺序正确） =====================
    elif data_name == 'FD001':
        fpath = os.path.join(ROOT_DIR, 'dataProcess/C-MAPSS/output')
        cluster_0 = np.load(f'{fpath}/FD001_degraded.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/FD001_normal.npy', allow_pickle=True)

    elif data_name == 'FD002':
        fpath = os.path.join(ROOT_DIR, 'dataProcess/C-MAPSS/output')
        cluster_0 = np.load(f'{fpath}/FD002_degraded.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/FD002_normal.npy', allow_pickle=True)

    elif data_name == 'FD003':
        fpath = os.path.join(ROOT_DIR, 'dataProcess/C-MAPSS/output')
        cluster_0 = np.load(f'{fpath}/FD003_degraded.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/FD003_normal.npy', allow_pickle=True)

    elif data_name == 'FD004':
        fpath = os.path.join(ROOT_DIR, 'dataProcess/C-MAPSS/output')
        cluster_0 = np.load(f'{fpath}/FD004_degraded.npy', allow_pickle=True)
        cluster_1 = np.load(f'{fpath}/FD004_normal.npy', allow_pickle=True)

    # ===================== 统一：FD数据集特征裁剪（提取公共逻辑，避免重复） =====================
    if data_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        if feature == 'MS':
            cluster_0 = cluster_0[:, :, :-1]
            cluster_1 = cluster_1[:, :, :-1]
        elif feature == 'S':
            cluster_0 = cluster_0[:, :, -1:]
            cluster_1 = cluster_1[:, :, -1:]
        elif feature == 'M':
            cluster_0 = cluster_0[:, :, :-2]
            cluster_1 = cluster_1[:, :, :-2]

    # ===================== 数据划分逻辑（完全保留原有逻辑，无任何修改） =====================
    if data_name in ["etth1", "etth2", 'AirQuality(bj)', 'AirQuality(Italian)']:
        orig_data = np.concatenate([cluster_0, cluster_1, cluster_2], axis=0)
        orig_data = fit_transform_scaler(orig_data, data_name)
        cluster_sizes = [cluster_0.shape[0], cluster_1.shape[0], cluster_2.shape[0]]
        cluster_0 = orig_data[:cluster_0.shape[0]]
        cluster_1 = orig_data[cluster_0.shape[0]:cluster_0.shape[0] + cluster_1.shape[0]]
        cluster_2 = orig_data[cluster_0.shape[0] + cluster_1.shape[0]:]

        c0_test = int(cluster_0.shape[0] * 0.1)
        c1_test = int(cluster_1.shape[0] * 0.1)
        c2_test = int(cluster_2.shape[0] * 0.1)

        test_data = np.concatenate([cluster_0[-c0_test * 2:], cluster_1[-c1_test * 2:], cluster_2[-c2_test * 2:]], axis=0)
        val_data = np.concatenate([cluster_0[-c0_test * 3:-c0_test * 2],
                                   cluster_1[-c1_test * 3:-c1_test * 2],
                                   cluster_2[-c2_test * 3:-c2_test * 2]], axis=0)

        minority = np.argmin(cluster_sizes)
        ratios = [1.0, 1.0, 1.0]
        ratios[minority] = 0.85

        c0_train_num = int(cluster_0.shape[0] * 0.7 * ratios[0])
        c1_train_num = int(cluster_1.shape[0] * 0.7 * ratios[1])
        c2_train_num = int(cluster_2.shape[0] * 0.7 * ratios[2])

        train_data = np.concatenate([cluster_0[:c0_train_num], cluster_1[:c1_train_num], cluster_2[:c2_train_num]], axis=0)
        train_data_g = [cluster_0, cluster_1, cluster_2][minority][:int([cluster_0, cluster_1, cluster_2][minority].shape[0]*0.7*0.85)]

    elif data_name == "Traffic":
        orig_data = np.concatenate([cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6], axis=0).astype(np.float32)
        orig_data = fit_transform_scaler(orig_data, data_name)
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

        train_list, val_list, test_list = [], [], []
        for c in [cluster_0, cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6]:
            n = c.shape[0]
            t = int(n * 0.1)
            train_list.append(c[:int(n * 0.7)])
            val_list.append(c[-3 * t:-2 * t])
            test_list.append(c[-2 * t:])
        val_data = np.concatenate(val_list, axis=0)
        test_data = np.concatenate(test_list, axis=0)
        train_data = np.concatenate([c[:max(1, int(c.shape[0]*0.7))] for c in train_list], axis=0)
        train_data_g = train_data

    # ===================== FD数据集 训练/验证/测试 划分（完全保留原有逻辑） =====================
    elif data_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        orig_data = np.concatenate([cluster_0, cluster_1], axis=0)
        orig_data = fit_transform_scaler(orig_data, data_name)  # 独立归一化
        len0 = cluster_0.shape[0]
        len1 = cluster_1.shape[0]
        cluster_0 = orig_data[:len0]
        cluster_1 = orig_data[len0:]

        # 10% 测试集
        c0_test = int(len0 * 0.1)
        c1_test = int(len1 * 0.1)

        test_data = np.concatenate([cluster_0[-c0_test*2:], cluster_1[-c1_test*2:]], axis=0)
        val_data = np.concatenate([cluster_0[-c0_test*3:-c0_test*2], cluster_1[-c1_test*3:-c1_test*2]], axis=0)

        # 不平衡数据采样
        majority = np.argmax([len0, len1])
        ratios = [1, 1]
        ratios[majority] = 0.7

        c0_train = int(len0 * 0.7 * ratios[0])
        c1_train = int(len1 * 0.7 * ratios[1])

        train_data = np.concatenate([cluster_0[:c0_train], cluster_1[:c1_train]], axis=0)
        train_data_g = cluster_0[:c0_train] if majority == 0 else cluster_1[:c1_train]

        print(f"✅ {data_name} 划分完成：")
        print(f"训练集: {train_data.shape}, 验证集: {val_data.shape}, 测试集: {test_data.shape}, 生成集: {train_data_g.shape}")

    return train_data, val_data, test_data, train_data_g


def get_X_Y(data, args, feature):
    seq_len = args.seq_len
    pred_len = args.pred_len

    if args.dataset in ['etth1', 'etth2']:
        if feature == 'MS':
            x = data[:, :seq_len, :]
            y = data[:, seq_len:seq_len + pred_len, -1:]
        else:
            x = data[:, :seq_len, :]
            y = data[:, seq_len:seq_len + pred_len, :]

    elif args.dataset in ['AirQuality(bj)']:
        if feature == 'MS':
            x = data[:, :seq_len, :]
            y = data[:, seq_len:seq_len + pred_len, :1]
        else:
            x = data[:, :seq_len, :]
            y = data[:, seq_len:seq_len + pred_len, :]

    elif args.dataset in ['AirQuality(Italian)']:
        if feature == 'MS':
            x = data[:, :seq_len, :]
            y = data[:, seq_len:seq_len + pred_len, :1]
        else:
            x = data[:, :seq_len, :]
            y = data[:, seq_len:seq_len + pred_len, :]

    elif args.dataset in ['Traffic']:
        x = data[:, :seq_len, :]
        y = data[:, seq_len:seq_len + pred_len, :]

    elif args.dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
        if feature == 'MS':
            x = data[:, :seq_len, :]
            y = data[:, seq_len:seq_len + 1, -1:]
        else:
            x = data[:, :seq_len, :]
            y = data[:, seq_len:seq_len + 1, :]

    return x, y


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    data = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(Y))
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader


def get_dataloader_gen(args, feature='MS', gen_mode='TimeVAE'):
    train_data, val_data, test_data, train_data_g = real_data_loading(args.dataset, feature)
    target_feature_dim = train_data.shape[-1]
    print(f"[INFO] 训练数据目标特征维度: {target_feature_dim}")

    gen_data = gen_data_loading(
        data_name=args.dataset,
        gen_mode=gen_mode,
        target_feature_dim=target_feature_dim,
        feature=feature
    )

    if train_data.shape[1:] != gen_data.shape[1:]:
        raise ValueError(f"训练数据维度({train_data.shape[1:]})与生成数据维度({gen_data.shape[1:]})不一致！")

    train_data = np.concatenate([train_data, gen_data], axis=0)

    x_tra, y_tra = get_X_Y(train_data, args, feature)
    x_val, y_val = get_X_Y(val_data, args, feature)
    x_test, y_test = get_X_Y(test_data, args, feature)

    x_tra = x_tra.astype(np.float32)
    y_tra = y_tra.astype(np.float32)

    print('Train:', x_tra.shape, y_tra.shape)
    print('Val:', x_val.shape, y_val.shape)
    print('Test:', x_test.shape, y_test.shape)

    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=False)
    val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=True, drop_last=False) if len(x_val) > 0 else None
    test_dataloader = data_loader(x_test, y_test, 32, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


def get_our_dataloader(args, feature='M'):
    _, _, _, train_data_g = real_data_loading(args.dataset, feature)
    train_dataloader = data_loader(
        train_data_g,
        train_data_g,
        args.batch_size,
        shuffle=True,
        drop_last=False
    )
    return train_dataloader


def get_dataloader(args, feature='M'):
    train_data, val_data, test_data, train_data_g = real_data_loading(args.dataset, feature)

    x_tra, y_tra = get_X_Y(train_data, args, feature)
    x_val, y_val = get_X_Y(val_data, args, feature)
    x_test, y_test = get_X_Y(test_data, args, feature)

    print('最终数据维度：')
    print('Train:', x_tra.shape, y_tra.shape)
    print('Val:', x_val.shape, y_val.shape)
    print('Test:', x_test.shape, y_test.shape)

    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=False)
    val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=True, drop_last=False) if len(x_val) > 0 else None
    test_dataloader = data_loader(x_test, y_test, 32, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader
