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
from sklearn.manifold import TSNE
import umap
from sklearn.utils import shuffle
import pickle
import os
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
stride = 24  # 滑动步长

# AirQuality(北京) 数据集配置
input_file = r'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(北京)\data\processed_PRSA.csv'
output_dir = r'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(北京)\output'
os.makedirs(output_dir, exist_ok=True)

# 原始数据特征（来自 processed_PRSA.csv）
# 包括: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM, time, wind_angle, wind_sin, wind_cos, AQI

# 需要删除的列
# columns_to_drop = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'wd', 'wind_angle']
columns_to_drop = ['wd', 'wind_angle']


# 最终特征顺序
final_feature_order = [
    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wind_sin', 'wind_cos', 'AQI',
    'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'
]


# 保留用于聚类的特征
retained_features = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wind_sin', 'wind_cos', 'AQI']

# 可选参数
candidate_k = range(2, 8)  # 候选K值
selected_k = None  # 如果为None，则使用最优k；否则使用指定的k值
clustering_features = None  # 如果为None，则使用所有保留特征；否则指定使用的特征列表


# 数据加载与预处理
def load_and_preprocess(input_file, seq_len, pred_len, window_size, stride, 
                        columns_to_drop=None, selected_features=None):
    """
    加载数据并使用滑动窗口生成时间序列数据
    返回所有特征的序列数据，但selected_features仅用于指定聚类特征
    """
    print("加载数据...")
    # 加载CSV数据
    df = pd.read_csv(input_file)
    
    print(f"原始列: {df.columns.tolist()}")
    print(f"原始数据形状: {df.shape}")
    
    # 删除指定的列（如果有的话）
    if columns_to_drop is not None:
        cols_to_drop_exist = [col for col in columns_to_drop if col in df.columns]
        if cols_to_drop_exist:
            df = df.drop(cols_to_drop_exist, axis=1)
            print(f"已删除列: {cols_to_drop_exist}")
    
    # 删除 time 列（不用于特征，仅用于排序）
    if 'time' in df.columns:
        df = df.drop('time', axis=1)
    
    # 获取所有保留的特征
    # all_features = df.columns.tolist()
    # 确保最终特征顺序
    all_features = [f for f in final_feature_order if f in df.columns]
    df = df[all_features]

    
    # 确定聚类特征（如果指定了则使用指定的，否则使用所有特征）
    if selected_features is None:
        clustering_features = all_features
    else:
        clustering_features = [f for f in selected_features if f in df.columns]


    
    print(f"所有特征: {all_features}")
    print(f"用于聚类的特征: {clustering_features}")
    
    # 使用所有特征生成序列数据
    data = df[all_features].values
    
    # 使用滑动窗口生成时间序列数据
    sequences = []
    indices = []
    
    # 计算可以生成的序列数量
    n_samples = len(data)
    i = 0
    while i + window_size <= n_samples:
        sequence = data[i:i + window_size]
        sequences.append(sequence)
        indices.append(i)
        i += stride
    
    sequences = np.array(sequences)
    indices = np.array(indices)
    
    print(f"删除列后数据形状: {data.shape}")
    print(f"生成时间序列数量: {len(sequences)}")
    print(f"每个序列形状: {sequences[0].shape}")
    print(f"总时间序列数据形状: {sequences.shape}")
    
    return sequences, indices, all_features, clustering_features


# 标准化函数
def standardize_sequences(sequences, features):
    n_samples = sequences.shape[0]
    window_size = sequences.shape[1]
    n_features = sequences.shape[2]
    
    # 标准化所有特征
    scaler = StandardScaler()
    sequences_reshaped = sequences.reshape(-1, n_features)
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences_scaled = sequences_scaled.reshape(n_samples, window_size, n_features)
    
    print(f"标准化完成")
    
    return sequences_scaled, scaler


# 聚类函数
def perform_clustering(scaled_sequences, candidate_k, selected_k=None):
    """
    对标准化后的时间序列数据进行KMeans聚类
    """
    n_samples = scaled_sequences.shape[0]
    window_size = scaled_sequences.shape[1]
    n_features = scaled_sequences.shape[2]
    
    # 将标准化后的序列展平为向量用于聚类
    flattened_sequences = scaled_sequences.reshape(n_samples, -1)
    
    # 使用肘部法则和轮廓系数确定最优聚类数量
    inertia = []
    silhouette_scores = []
    calinski_scores = []
    
    for k in candidate_k:
        print(f"正在进行KMeans聚类，k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=2026, n_init=10)
        kmeans.fit(flattened_sequences)
        
        inertia.append(kmeans.inertia_)
        
        # 计算轮廓系数
        if n_samples > 1000:
            sample_indices = np.random.choice(n_samples, size=min(1000, n_samples), replace=False)
            sample_data = flattened_sequences[sample_indices]
            sample_labels = kmeans.labels_[sample_indices]
            silhouette_scores.append(silhouette_score(sample_data, sample_labels))
        else:
            silhouette_scores.append(silhouette_score(flattened_sequences, kmeans.labels_))
        
        # 计算Calinski-Harabasz指数
        calinski_scores.append(calinski_harabasz_score(flattened_sequences, kmeans.labels_))
        
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
    plt.close()
    
    best_k_idx = np.argmax(silhouette_scores)
    best_k = list(candidate_k)[best_k_idx]
    print(f"\n最优聚类数量: k={best_k} (轮廓系数最高)")
    if selected_k is not None:
        print(f"\n指定的聚类数量: k={selected_k}")

    # 选择最优k值
    if selected_k is not None:
        best_k = selected_k
        print(f"\n使用指定的k={best_k}进行最终聚类...")
    else:
        best_k_idx = np.argmax(silhouette_scores)
        best_k = list(candidate_k)[best_k_idx]
        print(f"\n使用最优聚类数量: k={best_k} (轮廓系数最高)进行最终聚类...")
    
    # 使用最优k值重新聚类
    final_clusterer = KMeans(n_clusters=best_k, random_state=2026, n_init=10)
    labels = final_clusterer.fit_predict(flattened_sequences)
    
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
            target_class = unique_labels[np.argmax(counts)]
        elif target_class_type == '样本最少':
            target_class = unique_labels[np.argmin(counts)]
        elif target_class_type == '中间样本':
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


# 可视化聚类结果（PCA + t-SNE + UMAP）
def visualize_clustering_results(sequences, scaled_sequences, labels,
                                 few_shot_sequences, scaled_few_shot_sequences,
                                 few_shot_labels, target_class):
    """
    使用PCA、t-SNE、UMAP进行可视化
    """
    print(f"\n可视化聚类结果...")
    
    # 展平序列数据用于降维
    flattened_sequences = scaled_sequences.reshape(len(sequences), -1)
    flattened_few_shot_sequences = scaled_few_shot_sequences.reshape(len(few_shot_sequences), -1)
    
    # 获取唯一的标签和颜色映射
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    
    # ============ PCA 降维 ============
    print("执行PCA降维...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flattened_sequences)
    pca_few_shot = pca.transform(flattened_few_shot_sequences)
    
    # ============ t-SNE 降维 ============
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=2026, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(flattened_sequences)
    
    # 对小样本数据应用相同的t-SNE变换（使用拟合的模型）
    tsne_few_shot_data = flattened_few_shot_sequences
    tsne_few_shot = TSNE(n_components=2, random_state=2026, perplexity=min(30, len(tsne_few_shot_data)-1), 
                         n_iter=1000).fit_transform(tsne_few_shot_data)
    
    # ============ UMAP 降维 ============
    print("执行UMAP降维...")
    umap_model = umap.UMAP(n_components=2, random_state=2026, n_neighbors=15, min_dist=0.1)
    umap_result = umap_model.fit_transform(flattened_sequences)
    
    # 对小样本数据应用UMAP
    umap_few_shot = umap_model.transform(flattened_few_shot_sequences)
    
    # 绘制完整数据的三种可视化
    print("绘制完整数据的可视化图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (ax, data, method_name) in enumerate([
        (axes[0], pca_result, 'PCA'),
        (axes[1], tsne_result, 't-SNE'),
        (axes[2], umap_result, 'UMAP')
    ]):
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            ax.scatter(data[mask, 0], data[mask, 1],
                      color=cmap(i), alpha=0.7, s=30, label=f'类别 {label}')
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(f'{method_name} - 聚类结果')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/clustering_results_full_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("完整数据可视化已保存")
    
    # 绘制小样本数据的三种可视化
    print("绘制小样本数据的可视化图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (ax, data, method_name) in enumerate([
        (axes[0], pca_few_shot, 'PCA'),
        (axes[1], tsne_few_shot, 't-SNE'),
        (axes[2], umap_few_shot, 'UMAP')
    ]):
        # 绘制非目标类别
        target_mask = (few_shot_labels == target_class)
        non_target_mask = ~target_mask
        non_target_labels = few_shot_labels[non_target_mask]
        unique_non_target_labels = np.unique(non_target_labels)
        
        for label in unique_non_target_labels:
            mask = (few_shot_labels == label)
            original_label_idx = np.where(unique_labels == label)[0][0]
            ax.scatter(data[mask, 0], data[mask, 1],
                      color=cmap(original_label_idx), alpha=0.7, s=30,
                      label=f'类别 {label}')
        
        # 绘制目标类别
        if np.any(target_mask):
            target_label_idx = np.where(unique_labels == target_class)[0][0]
            ax.scatter(data[target_mask, 0], data[target_mask, 1],
                      color=cmap(target_label_idx), marker='x', s=100,
                      alpha=0.9, linewidths=2,
                      label=f'类别 {target_class}')
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(f'{method_name} - 小样本数据')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/clustering_results_few_shot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("小样本数据可视化已保存")


# 主要处理函数
def run_clustering(selected_features=None, selected_k_value=None):
    """
    运行KMeans聚类方法
    """
    print("=" * 80)
    print("使用KMeans进行AirQuality(北京)数据聚类分析")
    print("=" * 80)
    
    # 1. 加载和预处理数据
    print("\n1. 加载和预处理数据...")
    original_sequences, indices, all_features, clustering_features = load_and_preprocess(
        input_file, seq_len, pred_len, window_size, stride,
        columns_to_drop=columns_to_drop,
        selected_features=selected_features)
    
    # 2. 标准化
    print("\n2. 标准化数据...")
    scaled_sequences, scaler = standardize_sequences(original_sequences, all_features)
    
    # 3. 提取用于聚类的特征
    print("\n3. 提取用于聚类的特征...")
    feature_indices = [all_features.index(f) for f in clustering_features]
    scaled_sequences_for_clustering = scaled_sequences[:, :, feature_indices]
    print(f"用于聚类的数据形状: {scaled_sequences_for_clustering.shape}")
    
    # 4. 执行聚类
    print("\n4. 执行KMeans聚类...")
    best_k, labels, metrics = perform_clustering(
        scaled_sequences_for_clustering, candidate_k, selected_k=selected_k_value)
    
    if best_k is None:
        print("KMeans聚类失败")
        return None
    
    # 5. 保存每个簇的序列
    print("\n5. 保存每个簇的序列...")
    cluster_dict = save_cluster_sequences(original_sequences, labels, output_dir, suffix='')
    
    # 6. 模拟小样本情况
    print("\n6. 模拟小样本情况...")
    reduction_ratio = 0.9
    few_shot_sequences, few_shot_labels, retained_indices, target_class = simulate_few_shot(
        original_sequences, labels, target_class=None, reduction_ratio=reduction_ratio,
        target_class_type='样本最多')
    
    #  保存小样本的聚类分组
    print("\n6.5 保存小样本的数据...")
    cluster_dict_few_shot = save_cluster_sequences(
        original_sequences[retained_indices],   
        few_shot_labels,
        output_dir,
        suffix='_few_shot',
        target_cluster_id=target_class
    )
    
    # 6. 对小样本数据进行标准化
    print("\n7. 对小样本数据进行标准化...")
    n_samples_few = few_shot_sequences.shape[0]
    window_size_few = few_shot_sequences.shape[1]
    n_features_few = few_shot_sequences.shape[2]
    
    sequences_few_reshaped = few_shot_sequences.reshape(-1, n_features_few)
    sequences_few_scaled = scaler.transform(sequences_few_reshaped)
    scaled_few_shot_sequences = sequences_few_scaled.reshape(n_samples_few, window_size_few, n_features_few)
    
    # 7. 准备训练数据
    print("\n8. 准备训练数据...")
    X = scaled_few_shot_sequences[:, :seq_len, :]  # 输入序列
    y = scaled_few_shot_sequences[:, seq_len:, :]  # 输出序列
    
    print(f"  输入数据形状 (X): {X.shape}")
    print(f"  输出数据形状 (y): {y.shape}")
    
    # 8. 可视化聚类结果
    print("\n9. 可视化聚类结果...")
    visualize_clustering_results(
        original_sequences, scaled_sequences, labels,
        few_shot_sequences, scaled_few_shot_sequences,
        few_shot_labels, target_class
    )
    
    # 9. 保存所有数据
    print("\n10. 保存数据...")
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
        'scaler': scaler,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'window_size': window_size,
        'stride': stride,
        'all_features': all_features,
        'clustering_features': clustering_features,
        'cluster_dict': cluster_dict,
        'cluster_dict_few_shot': cluster_dict_few_shot
    }
    
    print(f"\nAirQuality(北京)数据聚类处理完成！")
    
    return data_dict


# 主函数
def main():
    print("\n" + "=" * 80)
    print("AirQuality(北京) 数据集聚类分析")
    print("=" * 80)
    
    # 配置参数
    # 可选：指定用于聚类的特征（默认使用所有保留特征）
    # selected_features_for_clustering = None  # 使用所有特征
    selected_features_for_clustering = None  # 使用所有保留特征 (TEMP, PRES, DEWP, RAIN, WSPM, wind_sin, wind_cos, AQI)

    # 可选：指定K值（默认使用最优K）
    # selected_k_value = None  # 自动计算最优K
    selected_k_value = 3  # 指定K值
    
    # 运行聚类
    data_dict = run_clustering(
        selected_features=selected_features_for_clustering,
        selected_k_value=selected_k_value
    )
    
    # 输出总结
    if data_dict is not None:
        print(f"\n" + "=" * 80)
        print("生成的文件总结")
        print("=" * 80)
        print(f"1. 聚类评估指标图: {output_dir}/clustering_metrics_kmeans.png")
        print(f"2. 完整数据可视化: {output_dir}/clustering_results_full_data.png")
        print(f"3. 小样本数据可视化: {output_dir}/clustering_results_few_shot.png")
        
        # 列出生成的npy文件
        npy_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        print(f"4. npy文件 ({len(npy_files)}个):")
        for npy_file in npy_files:
            data = np.load(f'{output_dir}/{npy_file}')
            print(f"   - {npy_file}: {data.shape}")
        
        print(f"\n所有输出文件已保存到 '{output_dir}' 文件夹中")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
