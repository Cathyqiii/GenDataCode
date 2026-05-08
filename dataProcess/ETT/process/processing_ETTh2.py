# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import pickle
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings
import sys
import time

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(2026)
torch.manual_seed(2026)

# 全局设置
seq_len = 96  # 输入序列长度
pred_len = 12  # 输出序列长度
window_size = seq_len + pred_len  # 滑动窗口大小
stride = 24  # 滑动步长 24h
cond_features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
target_features = ['OT']  # 目标特征
all_features = cond_features + target_features
data_path = 'ETT数据集/data/ETTh2.csv'
candidate_k = range(2, 8)

output_dir = 'ETT数据集/output/ETTh2'
os.makedirs(output_dir, exist_ok=True)


# 数据加载与预处理
def load_and_preprocess(data_path, seq_len, pred_len, window_size, stride):
    """
    加载数据并使用滑动窗口生成时间序列数据
    """
    # 加载数据
    df = pd.read_csv(data_path)

    # 分离条件特征和目标特征
    cond_data = df[cond_features].values
    target_data = df[target_features].values

    # 合并数据
    original_data = np.hstack([cond_data, target_data])

    # 使用滑动窗口生成时间序列数据
    sequences = []
    indices = []

    # 计算可以生成的序列数量
    n_samples = len(original_data)
    i = 0
    while i + window_size <= n_samples:
        sequence = original_data[i:i + window_size]
        sequences.append(sequence)
        indices.append(i)
        i += stride

    sequences = np.array(sequences)
    indices = np.array(indices)

    print(f"原始数据形状: {original_data.shape}")
    print(f"生成时间序列数量: {len(sequences)}")
    print(f"每个序列形状: {sequences[0].shape}")
    print(f"总时间序列数据形状: {sequences.shape}")
    print("注意：数据未进行标准化，返回原始值")

    return sequences, indices


# 标准化函数
def standardize_sequences(sequences, cond_features, target_features):
    n_samples = sequences.shape[0]
    window_size = sequences.shape[1]
    n_features = sequences.shape[2]

    # 分离条件特征和目标特征
    cond_sequences = sequences[:, :, :len(cond_features)]
    target_sequences = sequences[:, :, len(cond_features):]

    # 对条件特征进行标准化
    cond_scaler = StandardScaler()
    cond_sequences_reshaped = cond_sequences.reshape(-1, len(cond_features))
    cond_sequences_scaled = cond_scaler.fit_transform(cond_sequences_reshaped)
    cond_sequences_scaled = cond_sequences_scaled.reshape(n_samples, window_size, len(cond_features))

    # 对目标特征进行标准化
    target_scaler = StandardScaler()
    target_sequences_reshaped = target_sequences.reshape(-1, len(target_features))
    target_sequences_scaled = target_scaler.fit_transform(target_sequences_reshaped)
    target_sequences_scaled = target_sequences_scaled.reshape(n_samples, window_size, len(target_features))

    # 合并标准化后的特征
    scaled_sequences = np.concatenate([cond_sequences_scaled, target_sequences_scaled], axis=2)

    print(f"标准化完成:")

    return scaled_sequences, cond_scaler, target_scaler


# 聚类函数（只保留KMeans）
def perform_clustering(scaled_cond_sequences, candidate_k):
    """
    对标准化后的时间序列数据进行KMeans聚类
    """
    n_samples = scaled_cond_sequences.shape[0]
    window_size = scaled_cond_sequences.shape[1]
    n_features = scaled_cond_sequences.shape[2]

    # 将标准化后的条件特征展平为向量用于聚类
    flattened_cond_sequences = scaled_cond_sequences.reshape(n_samples, -1)

    # 使用肘部法则和轮廓系数确定最优聚类数量
    inertia = []
    silhouette_scores = []
    calinski_scores = []

    for k in candidate_k:
        print(f"正在进行KMeans聚类，k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=2026, n_init=10)
        kmeans.fit(flattened_cond_sequences)

        inertia.append(kmeans.inertia_)

        # 计算轮廓系数
        if n_samples > 1000:
            sample_indices = np.random.choice(n_samples, size=min(1000, n_samples), replace=False)
            sample_data = flattened_cond_sequences[sample_indices]
            sample_labels = kmeans.labels_[sample_indices]
            silhouette_scores.append(silhouette_score(sample_data, sample_labels))
        else:
            silhouette_scores.append(silhouette_score(flattened_cond_sequences, kmeans.labels_))

        # 计算Calinski-Harabasz指数
        calinski_scores.append(calinski_harabasz_score(flattened_cond_sequences, kmeans.labels_))

        print(f"  k={k}: 惯性={inertia[-1]:.2f}, 轮廓系数={silhouette_scores[-1]:.4f}, "
              f"Calinski-Harabasz={calinski_scores[-1]:.2f}")

    # 可视化聚类评估指标
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 肘部法则图
    axes[0].plot(candidate_k, inertia, 'bo-')
    axes[0].set_xlabel('聚类数量 k')
    axes[0].set_ylabel('惯性 (Inertia)')
    axes[0].set_title('KMeans - 肘部法则')
    axes[0].grid(True)

    # 轮廓系数图
    axes[1].plot(candidate_k, silhouette_scores, 'ro-')
    axes[1].set_xlabel('聚类数量 k')
    axes[1].set_ylabel('轮廓系数 (Silhouette Score)')
    axes[1].set_title('KMeans - 轮廓系数')
    axes[1].grid(True)

    # Calinski-Harabasz指数图
    axes[2].plot(candidate_k, calinski_scores, 'go-')
    axes[2].set_xlabel('聚类数量 k')
    axes[2].set_ylabel('Calinski-Harabasz指数')
    axes[2].set_title('KMeans - Calinski-Harabasz指数')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/clustering_metrics_kmeans.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 选择最优k值（轮廓系数最大）
    best_k_idx = np.argmax(silhouette_scores)
    best_k = candidate_k[best_k_idx]
    print(f"\n最优聚类数量: k={best_k} (轮廓系数最高)")

    # 使用最优k值重新聚类
    print(f"\n使用最优k={best_k}进行最终聚类...")
    final_clusterer = KMeans(n_clusters=best_k, random_state=2026, n_init=10)
    labels = final_clusterer.fit_predict(flattened_cond_sequences)

    # 统计每个类别的样本数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n聚类结果统计:")
    for label, count in zip(unique_labels, counts):
        print(f"  类别 {label}: {count} 个样本 ({count / len(labels) * 100:.1f}%)")

    # 保存聚类指标
    cluster_metrics = {
        'method': 'kmeans',
        'inertia': inertia,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'best_k': best_k,
        'labels': labels,
        'cluster_counts': dict(zip(unique_labels, counts)),
        'clusterer': final_clusterer
    }

    return best_k, labels, cluster_metrics


# 保存每个簇的序列到npy文件（支持后缀）
def save_cluster_sequences(sequences, labels, output_dir, suffix='', target_cluster_id=None):
    """
    将每个簇的序列保存为单独的npy文件
    suffix: 文件后缀标识（如'_few_shot'）
    target_cluster_id: 仅保存指定的簇（用于小样本数据）；如果为None则保存所有簇

    Parameters:
    -----------
    sequences : numpy.ndarray
        原始序列数据，形状为 (num_samples, window_size, num_features)
    labels : numpy.ndarray
        聚类标签
    output_dir : str
        输出目录
    suffix : str
        文件后缀标识（如'_few_shot'）
    target_cluster_id : int
        仅保存指定的簇（用于小样本数据）；如果为None则保存所有簇
    """
    if suffix:
        print(f"\n保存每个簇的序列为npy文件 ({suffix})...")
    else:
        print("\n保存每个簇的序列为npy文件...")

    # 获取唯一的标签
    unique_labels = np.unique(labels)
    
    # 决定要保存的簇
    if target_cluster_id is not None:
        clusters_to_save = [target_cluster_id]
    else:
        clusters_to_save = unique_labels

    for cluster_id in clusters_to_save:
        # 获取属于当前簇的序列索引
        cluster_indices = np.where(labels == cluster_id)[0]

        # 提取当前簇的序列
        cluster_sequences = sequences[cluster_indices]

        # 保存为npy文件
        filename = f'{output_dir}/cluster_{cluster_id}{suffix}.npy'
        np.save(filename, cluster_sequences)

        print(f"  簇 {cluster_id}: {cluster_sequences.shape} -> 保存到 {filename}")

    # 保存所有标签（仅在保存全部簇时保存）
    if not suffix and target_cluster_id is None:
        np.save(f'{output_dir}/all_labels.npy', labels)
        print(f"  所有标签 -> 保存到 {output_dir}/all_labels.npy")

    # 返回簇序列的字典
    cluster_dict = {}
    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_dict[cluster_id] = sequences[cluster_indices]

    return cluster_dict


# 模拟小样本情况
def simulate_few_shot(sequences, labels, target_class=None, reduction_ratio=0.9,
                      target_class_type='样本最多'):
    """
    模拟小样本情况：只减少某一类别的样本数量
    """
    # 统计每个类别的样本数
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 根据策略选择目标类别
    if target_class is None:
        if target_class_type == '样本最多':
            # 选择样本数量最多的类别
            target_class = unique_labels[np.argmax(counts)]
        elif target_class_type == '样本最少':
            # 选择样本数量最少的类别
            target_class = unique_labels[np.argmin(counts)]
        elif target_class_type == '中间样本':
            # 选择样本数量中间的类别
            sorted_indices = np.argsort(counts)
            middle_idx = len(sorted_indices) // 2
            target_class = unique_labels[sorted_indices[middle_idx]]
        else:
            raise ValueError(f"不支持的target_class_type: {target_class_type}")

    print(f"\n选择目标类别: {target_class}")
    print(f"选择方式: {target_class_type}")
    print(f"减少比例: {reduction_ratio * 100:.0f}%")

    # 获取所有类别的索引
    all_indices = np.arange(len(labels))

    # 获取目标类别的索引
    target_indices = all_indices[labels == target_class]

    # 获取其他类别的索引
    other_indices = all_indices[labels != target_class]

    # 计算目标类别保留的样本数量
    n_target_retain = max(1, int(len(target_indices) * (1 - reduction_ratio)))

    # 随机选择目标类别中保留的样本
    np.random.shuffle(target_indices)
    retained_target_indices = target_indices[:n_target_retain]

    # 合并所有保留的索引
    retained_indices = np.concatenate([other_indices, retained_target_indices])
    retained_indices.sort()

    # 提取保留的序列和标签
    few_shot_sequences = sequences[retained_indices]
    few_shot_labels = labels[retained_indices]

    # 统计结果
    print(f"\n小样本模拟结果:")
    print(f"  原始总样本数: {len(sequences)}")
    print(f"  处理后总样本数: {len(few_shot_sequences)}")
    print(f"  目标类别 {target_class}:")
    print(f"    原始样本数: {len(target_indices)}")
    print(f"    保留样本数: {len(retained_target_indices)}")
    print(f"    保留比例: {len(retained_target_indices) / len(target_indices) * 100:.1f}%")
    print(f"    减少数量: {len(target_indices) - len(retained_target_indices)}")

    return few_shot_sequences, few_shot_labels, retained_indices, target_class


# 将序列数据转换为DataFrame（每个时间步为一行，简化版）
def sequences_to_simple_dataframe(sequences, labels, indices, seq_len, pred_len,
                                  cond_features, target_features, data_type='原始值'):
    n_samples = sequences.shape[0]
    window_size = sequences.shape[1]
    n_features = sequences.shape[2]

    # 确保特征数量匹配
    assert n_features == len(cond_features) + len(target_features), "特征数量不匹配"

    # 准备数据列表
    data = []

    for sample_idx in range(n_samples):
        # 获取当前样本的序列和元数据
        sample_sequence = sequences[sample_idx]
        label = labels[sample_idx]

        # 遍历序列中的每个时间步
        for t in range(window_size):
            # 确定时间步类型和编号
            if t < seq_len:
                step_type = "输入"
                step_num = t
            else:
                step_type = "输出"
                step_num = t - seq_len

            # 获取当前时间步的特征值
            features = sample_sequence[t]

            # 创建简化版行数据
            row = {
                '样本ID': sample_idx,
                '聚类': label,
                '时间步类型': step_type,
                '时间步编号': step_num
            }

            # 添加所有特征值
            for i, feature_name in enumerate(all_features):
                row[feature_name] = features[i]

            data.append(row)

    # 创建DataFrame
    df_simple = pd.DataFrame(data)

    return df_simple


# 可视化聚类结果
def visualize_clustering_results(sequences, scaled_sequences, labels,
                                 few_shot_sequences, scaled_few_shot_sequences,
                                 few_shot_labels, target_class):
    """
    可视化聚类结果
    """
    print(f"\n可视化KMeans聚类结果...")

    # 使用PCA降维可视化（使用标准化后的条件特征）
    pca = PCA(n_components=2)

    # 只使用标准化后的条件特征进行降维

    scaled_cond_sequences = scaled_sequences[:, :, :len(cond_features)]
    flattened_scaled_cond_sequences = scaled_cond_sequences.reshape(len(sequences), -1)
    pca_result = pca.fit_transform(flattened_scaled_cond_sequences)

    plt.figure(figsize=(14, 6))

    # 原始聚类结果
    plt.subplot(1, 2, 1)
    # 获取所有唯一的标签
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # 为每个类别分配颜色（使用tab20色图）
    cmap = plt.cm.get_cmap('tab20', n_clusters)

    # 绘制每个类别的点
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                    color=cmap(i), alpha=0.7, s=30, label=f'类别 {label}')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('KMeans - 聚类结果 (基于标准化条件特征, PCA降维)')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)

    # 小样本处理后的聚类结果
    plt.subplot(1, 2, 2)
    # 只使用标准化后的条件特征进行降维
    scaled_few_shot_cond_sequences = scaled_few_shot_sequences[:, :, :len(cond_features)]
    flattened_scaled_few_shot_cond_sequences = scaled_few_shot_cond_sequences.reshape(len(scaled_few_shot_sequences),
                                                                                      -1)
    pca_few_shot = pca.transform(flattened_scaled_few_shot_cond_sequences)

    # 分离目标类别和其他类别
    target_mask = (few_shot_labels == target_class)
    non_target_mask = ~target_mask

    # 先绘制非目标类别（使用原始颜色）
    non_target_labels = few_shot_labels[non_target_mask]
    unique_non_target_labels = np.unique(non_target_labels)

    for i, label in enumerate(unique_non_target_labels):
        mask = (few_shot_labels == label)
        # 找到这个标签在原始标签中的索引
        original_label_idx = np.where(unique_labels == label)[0][0]
        plt.scatter(pca_few_shot[mask, 0], pca_few_shot[mask, 1],
                    color=cmap(original_label_idx), alpha=0.7, s=30,
                    label=f'类别 {label}')

    # 最后绘制目标类别
    if np.any(target_mask):
        # 找到目标类别在原始标签中的索引
        target_label_idx = np.where(unique_labels == target_class)[0][0]
        plt.scatter(pca_few_shot[target_mask, 0], pca_few_shot[target_mask, 1],
                    color=cmap(target_label_idx), marker='x', s=100,
                    alpha=0.9, linewidths=2,
                    label=f'类别 {target_class}')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('KMeans - 小样本处理后的聚类分布')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/clustering_results_kmeans.png', dpi=150, bbox_inches='tight')
    plt.close()


# 运行KMeans聚类方法
def run_clustering(original_sequences=None, scaled_sequences=None,
                   indices=None, cond_scaler=None, target_scaler=None):
    """
    运行KMeans聚类方法
    """
    if original_sequences is None or scaled_sequences is None or indices is None:
        print(f"数据未提供，重新加载数据...")
        original_sequences, indices = load_and_preprocess(
            data_path, seq_len, pred_len, window_size, stride)
        scaled_sequences, cond_scaler, target_scaler = standardize_sequences(
            original_sequences, cond_features, target_features)
    else:
        print(f"使用预加载的数据...")

    print(f"\n{'=' * 60}")
    print(f"使用KMeans方法进行条件可控的小样本时间序列数据生成")
    print(f"{'=' * 60}")

    # 1. 执行聚类（使用标准化后的条件特征）
    print(f"\n1. 使用KMeans进行聚类分析...")

    # 提取标准化后的条件特征
    scaled_cond_sequences = scaled_sequences[:, :, :len(cond_features)]

    best_k, labels, metrics = perform_clustering(
        scaled_cond_sequences, candidate_k)

    if best_k is None:
        print(f"KMeans聚类失败")
        return None

    # 1.5 保存全量数据的每个簇的序列到npy文件
    print(f"\n1.5 保存全量数据的每个簇的序列到npy文件...")
    cluster_dict = save_cluster_sequences(original_sequences, labels, output_dir, suffix='')

    # 2. 模拟小样本情况
    print("\n2. 模拟小样本情况...")
    # 选择样本最多的类别作为目标类别，减少90%的样本
    reduction_ratio = 0.9
    few_shot_sequences, few_shot_labels, retained_indices, target_class = simulate_few_shot(
        original_sequences, labels, target_class=None, reduction_ratio=reduction_ratio,
        target_class_type='样本最多')

    # 2.5 保存小样本的聚类分组（仅保存被减少的目标簇）
    print("\n2.5 保存小样本的聚类分组...")
    cluster_dict_few_shot = save_cluster_sequences(few_shot_sequences, few_shot_labels, 
                                                   output_dir, suffix='_few_shot', 
                                                   target_cluster_id=target_class)

    # 3. 对小样本数据进行标准化
    print("\n3. 对小样本数据进行标准化...")
    # 分离条件特征和目标特征
    n_samples_few = few_shot_sequences.shape[0]
    window_size_few = few_shot_sequences.shape[1]

    cond_sequences_few = few_shot_sequences[:, :, :len(cond_features)]
    target_sequences_few = few_shot_sequences[:, :, len(cond_features):]

    # 使用相同的标准化器进行转换
    cond_sequences_few_reshaped = cond_sequences_few.reshape(-1, len(cond_features))
    cond_sequences_few_scaled = cond_scaler.transform(cond_sequences_few_reshaped)
    cond_sequences_few_scaled = cond_sequences_few_scaled.reshape(n_samples_few, window_size_few, len(cond_features))

    target_sequences_few_reshaped = target_sequences_few.reshape(-1, len(target_features))
    target_sequences_few_scaled = target_scaler.transform(target_sequences_few_reshaped)
    target_sequences_few_scaled = target_sequences_few_scaled.reshape(n_samples_few, window_size_few,
                                                                      len(target_features))

    # 合并标准化后的小样本序列
    scaled_few_shot_sequences = np.concatenate([cond_sequences_few_scaled, target_sequences_few_scaled], axis=2)

    # 4. 准备训练数据
    print("\n4. 准备训练数据...")
    # 输入：前96个时间步的标准化条件特征（6个特征）
    X = scaled_few_shot_sequences[:, :seq_len, :len(cond_features)]  # 输入序列（标准化条件特征）
    # 输出：后12个时间步的标准化OT特征（1个特征）
    y = scaled_few_shot_sequences[:, seq_len:, len(cond_features):]  # 输出序列（标准化OT特征）

    print(f"  输入数据形状 (X): {X.shape}")
    print(f"  输出数据形状 (y): {y.shape}")

    # 5. 保存所有数据
    print("\n5. 保存数据...")
    data_dict = {
        'original_sequences': original_sequences,
        'scaled_sequences': scaled_sequences,
        'indices': indices,
        'labels': labels,
        'cluster_metrics': metrics,
        'few_shot_sequences': few_shot_sequences,
        'scaled_few_shot_sequences': scaled_few_shot_sequences,
        'few_shot_labels': few_shot_labels,
        'retained_indices': retained_indices,
        'target_class': target_class,
        'reduction_ratio': reduction_ratio,
        'X_train': X,
        'y_train': y,
        'cond_scaler': cond_scaler,
        'target_scaler': target_scaler,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'window_size': window_size,
        'stride': stride,
        'cond_features': cond_features,
        'target_features': target_features,
        'cluster_dict': cluster_dict,
        'cluster_dict_few_shot': cluster_dict_few_shot
    }

    # 6. 可视化部分结果
    visualize_clustering_results(
        original_sequences, scaled_sequences, labels,
        few_shot_sequences, scaled_few_shot_sequences,
        few_shot_labels, target_class
    )

    print(f"\nKMeans方法的数据处理完成！")

    return data_dict


# 主函数
def main():
    print("=" * 80)
    print("使用KMeans的条件可控的小样本时间序列数据生成")
    print("=" * 80)

    # 运行KMeans聚类
    data_dict = run_clustering()

    # 显示生成的npy文件信息
    if data_dict is not None:
        print(f"\n生成的文件总结:")
        print(f"1. 聚类结果图: {output_dir}/clustering_metrics_kmeans.png")
        print(f"2. 聚类分布图: {output_dir}/clustering_results_kmeans.png")

        # 列出生成的npy文件
        npy_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
        print(f"4. npy文件 ({len(npy_files)}个):")
        for npy_file in sorted(npy_files):
            if npy_file != 'all_labels.npy':
                # 加载文件并显示形状
                data = np.load(f'{output_dir}/{npy_file}')
                print(f"   - {npy_file}: {data.shape} (序列数量×108×7)")

    print(f"\n所有输出文件已保存到 '{output_dir}' 文件夹中")


if __name__ == "__main__":
    main()