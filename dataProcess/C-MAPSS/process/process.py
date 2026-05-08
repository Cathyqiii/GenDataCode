# -*- coding: utf-8 -*-
"""
滑动窗口处理 C-MAPSS 全部4个数据集 FD001/FD002/FD003/FD004
路径已修复为你本地真实路径
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_remaining_useful_life(df):
    """计算每个样本的RUL（剩余使用寿命）"""
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    result_frame = df.merge(
        max_cycle.to_frame(name="max_cycle"), left_on="unit_nr", right_index=True
    )

    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def add_operating_condition(df):
    """提取运行条件"""
    df_op_cond = df.copy()

    df_op_cond["setting_1"] = abs(df_op_cond["setting_1"].round())
    df_op_cond["setting_2"] = abs(df_op_cond["setting_2"].round(decimals=2))

    df_op_cond["op_cond"] = (
            df_op_cond["setting_1"].astype(str)
            + "_"
            + df_op_cond["setting_2"].astype(str)
            + "_"
            + df_op_cond["setting_3"].astype(str)
    )

    return df_op_cond


def condition_scaler(df, sensor_names):
    """条件感知缩放"""
    df = df.copy()
    df[sensor_names] = df[sensor_names].astype(np.float32)

    scaler = StandardScaler()
    for condition in df["op_cond"].unique():
        mask = df["op_cond"] == condition
        scaler.fit(df.loc[mask, sensor_names])
        df.loc[mask, sensor_names] = scaler.transform(
            df.loc[mask, sensor_names]
        )

    return df, scaler


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    """指数平滑"""
    df = df.copy()

    # 对每个engine分别进行指数加权平均
    df[sensors] = (
        df.groupby("unit_nr")[sensors]
        .apply(lambda x: x.ewm(alpha=alpha).mean())
        .reset_index(level=0, drop=True)
    )

    # 丢弃前n_samples以减少滤波延迟
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = (
        df.groupby("unit_nr")["unit_nr"]
        .transform(create_mask, samples=n_samples)
        .astype(bool)
    )
    df = df[mask]

    return df


def generate_sequences_for_unit(df_unit, window_length, step_size, sensors):
    """
    为单个engine生成滑动窗口序列

    参数:
    - df_unit: 单个engine的数据
    - window_length: 窗口长度 (33)
    - step_size: 步长 (10)
    - sensors: 传感器列表

    返回:
    - sequences: (n_sequences, window_length, n_sensors+2)
      最后两列为：原始RUL值、125截断后的RUL值
    - labels: (n_sequences,) RUL标签为窗口最后一个时间点的RUL
    """
    data = df_unit[sensors].values
    rul_values = df_unit["RUL"].values

    num_samples = data.shape[0]
    sequences = []
    labels = []

    # 按照步长生成滑动窗口
    for start in range(0, num_samples - window_length + 1, step_size):
        end = start + window_length

        # 提取序列数据
        sequence = data[start:end, :]

        # 提取窗口对应的RUL值
        window_rul = rul_values[start:end]  # 原始RUL值
        window_rul_clipped = np.clip(window_rul, 0, 125)  # 125截断后的RUL值

        # 将原始RUL和截断RUL作为额外列添加到序列中
        rul_features = np.column_stack([window_rul, window_rul_clipped])
        sequence_with_rul = np.concatenate([sequence, rul_features], axis=1)

        # 标签为窗口最后一个时间点的RUL
        label = rul_values[end - 1]

        sequences.append(sequence_with_rul)
        labels.append(label)

    if sequences:
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return sequences, labels
    else:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)


def generate_all_sequences(df, window_length, step_size, sensors):
    """
    为所有engine生成滑动窗口序列
    确保每个engine的数据不混到另一个里

    返回:
    - all_sequences: (total_sequences, window_length, n_sensors)
    - all_labels: (total_sequences,)
    """
    all_sequences = []
    all_labels = []

    unit_nrs = df["unit_nr"].unique()
    print(f"\n开始为每个engine生成序列...")

    for unit_nr in sorted(unit_nrs):
        df_unit = df[df["unit_nr"] == unit_nr].reset_index(drop=True)

        sequences, labels = generate_sequences_for_unit(
            df_unit, window_length, step_size, sensors
        )

        if len(sequences) > 0:
            all_sequences.append(sequences)
            all_labels.append(labels)
            print(f"  Engine {unit_nr}: {len(sequences)} 个序列")

    if all_sequences:
        all_sequences = np.concatenate(all_sequences, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_sequences = np.array([], dtype=np.float32)
        all_labels = np.array([], dtype=np.float32)

    print(f"\n总计生成序列: {all_sequences.shape[0]}")
    return all_sequences, all_labels


def save_processed_data_to_csv(df, output_dir, dataset="FD001"):
    """
    保存处理后的数据到CSV文件
    包含所有处理步骤（RUL、运行条件、缩放、平滑）的结果
    """
    print("\n" + "=" * 80)
    print("保存处理后的数据到CSV")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, f"train_{dataset}.csv")
    df.to_csv(csv_file, index=False)

    print(f"\n处理后的数据已保存:")
    print(f"  文件: {csv_file}")
    print(f"  行数: {len(df)}")
    print(f"  列数: {len(df.columns)}")


def save_sequences_by_label(sequences, labels, output_dir, dataset):
    """
    根据标签保存序列数据到npy文件
    按照RUL是否大于125进行分类 + 数据集名称区分
    """
    print("\n" + "=" * 80)
    print(f"保存 {dataset} 序列数据（按RUL > 125分类）")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # 二元分类标签
    binary_labels = (labels > 125).astype(np.int32)
    label_names = {0: "degraded", 1: "normal"}

    print(f"\n标签分布:")
    unique_labels, counts = np.unique(binary_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label_names[label]}: {count} 个序列")

    # 带数据集名保存
    for label in [0, 1]:
        label_indices = np.where(binary_labels == label)[0]
        if len(label_indices) > 0:
            label_sequences = sequences[label_indices]
            filename = f'{output_dir}/{dataset}_{label_names[label]}.npy'
            np.save(filename, label_sequences)
            print(f"  {dataset}_{label_names[label]}: {label_sequences.shape}")

    # 保存标签
    np.save(f'{output_dir}/{dataset}_all_labels.npy', binary_labels)
    print(f"\n{dataset} 数据保存完成！")


def process_fd_dataset(
        dataset="FD001",
        window_length=33,
        step_size=10,
        alpha=0.3,
        n_samples_smooth=0,
):
    """
    处理单个 C-MAPSS 数据集
    """
    print("\n" + "=" * 80)
    print(f"处理 {dataset} 数据集")
    print("=" * 80)

    # ✅ 已修复为你真实的路径
    data_dir = r"D:\GenDataCode_V1\dataProcess\C-MAPSS\data"
    output_dir = r"D:\GenDataCode_V1\dataProcess\C-MAPSS\output"

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(data_dir, f"train_{dataset}.txt")

    # 列名定义
    index_names = ["unit_nr", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = ["s_{}".format(i + 1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names

    # 读取数据
    print(f"\n读取 {dataset} 数据...")
    train = pd.read_csv(train_file, sep=r"\s+", header=None, names=col_names)
    print(f"  原始数据形状: {train.shape}")

    # 1. RUL
    train = add_remaining_useful_life(train)
    print(f"  RUL 范围: [{train['RUL'].min():.0f}, {train['RUL'].max():.0f}]")

    # 2. 运行条件
    train = add_operating_condition(train)
    print(f"  运行条件数: {train['op_cond'].nunique()}")

    # 传感器选择
    sensors = ["s_{}".format(i) for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]

    # 3. 条件缩放
    train, scaler = condition_scaler(train, sensors)

    # 4. 指数平滑
    train = exponential_smoothing(train, sensors, n_samples_smooth, alpha=alpha)
    print(f"  处理后数据形状: {train.shape}")

    # 5. 保存CSV
    save_processed_data_to_csv(train, output_dir, dataset)

    # 6. 滑动窗口
    sequences, labels = generate_all_sequences(train, window_length, step_size, sensors)

    print(f"\n{dataset} 序列形状: {sequences.shape}")
    print(f"标签范围: [{labels.min():.1f}, {labels.max():.1f}]")

    # 7. 保存序列
    save_sequences_by_label(sequences, labels, output_dir, dataset)

    return sequences, labels


def process_all_datasets():
    """
    批量处理 4 个数据集
    """
    datasets = ["FD001", "FD002", "FD003", "FD004"]

    for ds in datasets:
        try:
            process_fd_dataset(
                dataset=ds,
                window_length=33,
                step_size=10,
                alpha=0.3,
                n_samples_smooth=0
            )
        except Exception as e:
            print(f"\n❌ {ds} 处理失败: {e}")

    print("\n🎉 全部 4 个数据集处理完成！")


if __name__ == "__main__":
    process_all_datasets()