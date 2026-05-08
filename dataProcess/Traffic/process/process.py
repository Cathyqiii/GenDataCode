import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局设置
seq_len = 96  # 输入序列长度
pred_len = 12  # 输出序列长度
window_size = seq_len + pred_len  # 滑动窗口大小
stride = 24  # 滑动步长

# 输出目录
output_dir = 'Traffic\output'
os.makedirs(output_dir, exist_ok=True)


class TrafficDataProcessor:
    """交通流量数据处理器"""
    
    # 数值编码映射关系
    WEATHER_MAIN_ENCODING = {}
    WEATHER_DESCRIPTION_ENCODING = {}
    
    def __init__(self, data_path):
        """初始化数据处理器"""
        self.data = pd.read_csv(data_path)
        self.original_length = len(self.data)
    
    def print_statistics_before_cleaning(self):
        """
        清理之前打印weather_main和weather_description的统计信息
        """
        print("\n" + "=" * 60)
        print("数据清理前的统计信息")
        print("=" * 60)
        
        # 统计weather_main
        print("\nweather_main 类别统计：")
        print("-" * 60)
        weather_main_counts = self.data['weather_main'].value_counts()
        weather_main_total = len(self.data)
        weather_main_stat = pd.DataFrame({
            '次数': weather_main_counts,
            '百分比(%)': (weather_main_counts / weather_main_total * 100).round(2)
        })
        print(weather_main_stat)
        print(f"总数: {weather_main_total}")
        
        # 统计weather_description
        print("\nweather_description 类别统计：")
        print("-" * 60)
        weather_desc_counts = self.data['weather_description'].value_counts()
        weather_desc_total = len(self.data)
        weather_desc_stat = pd.DataFrame({
            '次数': weather_desc_counts,
            '百分比(%)': (weather_desc_counts / weather_desc_total * 100).round(2)
        })
        print(weather_desc_stat)
        print(f"总数: {weather_desc_total}")
        print("=" * 60)
        
    def standardize_categorical_features(self):
        """
        统一分类特征为小写
        """
        if 'weather_main' in self.data.columns:
            self.data['weather_main'] = self.data['weather_main'].str.lower()
        if 'weather_description' in self.data.columns:
            self.data['weather_description'] = self.data['weather_description'].str.lower()
        print("\n分类特征已统一为小写")
    
    def remove_abnormal_data(self, rain_threshold=2000, temp_threshold=200):
        """
        删除异常数据
        
        Args:
            rain_threshold: rain_1h阈值，删除大于该值的数据
            temp_threshold: temp阈值，删除小于该值的数据
        """
        initial_length = len(self.data)
        
        # 删除rain_1h异常值
        rain_mask = self.data['rain_1h'] > rain_threshold
        rain_count = rain_mask.sum()
        self.data = self.data[~rain_mask]
        
        # 删除temp异常值
        temp_mask = self.data['temp'] < temp_threshold
        temp_count = temp_mask.sum()
        self.data = self.data[~temp_mask]
        
        self.data.reset_index(drop=True, inplace=True)
        
        # 显示删除信息
        total_deleted = initial_length - len(self.data)
        print("=" * 60)
        print("数据删除统计信息")
        print("=" * 60)
        print(f"原数据长度: {initial_length}")
        print(f"  - 删除rain_1h异常值 (>2000): {rain_count} 行")
        print(f"  - 删除temp异常值 (<200): {temp_count} 行")
        print(f"总共删除: {total_deleted} 行")
        print(f"剩余数据长度: {len(self.data)}")
        print("=" * 60)
       
    def save_processed_data(self, output_path):
        """保存处理后的数据"""
        self.data.to_csv(output_path, index=False)
        print(f"\n数据已保存到: {output_path}")
        print(f"最终数据形状: {self.data.shape}")
    
    def get_processed_data(self):
        """返回处理后的数据"""
        return self.data


def generate_time_series(df, seq_len, pred_len, window_size, stride):
    """
    使用滑动窗口生成时间序列数据
    """
    print("\n" + "=" * 80)
    print("生成时间序列数据")
    print("=" * 80)
    
    # 获取所有特征
    all_features = df.columns.tolist()
    data = df[all_features].values
    
    # 使用滑动窗口生成时间序列数据
    sequences = []
    indices = []
    
    n_samples = len(data)
    i = 0
    while i + window_size <= n_samples:
        sequence = data[i:i + window_size]
        sequences.append(sequence)
        indices.append(i)
        i += stride
    
    sequences = np.array(sequences)
    indices = np.array(indices)
    
    print(f"原始数据形状: {data.shape}")
    print(f"生成时间序列数量: {len(sequences)}")
    print(f"每个序列形状: {sequences[0].shape}")
    print(f"总时间序列数据形状: {sequences.shape}")
    print(f"参数配置: seq_len={seq_len}, pred_len={pred_len}, window_size={window_size}, stride={stride}")
    
    return sequences, indices, all_features


def extract_labels_from_sequences(sequences, indices, df, all_features, window_size):
    """
    为每条时间序列提取weather_main的众数作为标签
    """
    print("\n" + "=" * 80)
    print("提取序列标签（weather_main的众数）")
    print("=" * 80)
    
    # 获取weather_main列的值
    if 'weather_main' in all_features:
        weather_main_values = df['weather_main'].values
    else:
        raise ValueError("未找到weather_main列")
    
    labels = []
    
    for i, idx in enumerate(indices):
        # 提取窗口内的weather_main值
        window_start = idx
        window_end = idx + window_size
        
        window_weather = weather_main_values[window_start:window_end]
        
        # 计算众数
        unique_vals, counts = np.unique(window_weather, return_counts=True)
        mode_value = unique_vals[np.argmax(counts)]
        
        labels.append(mode_value)
    
    labels = np.array(labels)
    
    print(f"提取完成，共{len(labels)}条序列")
    print(f"标签数据类型: {labels.dtype}")
    print(f"标签示例: {labels[:10]}")
    
    return labels


def analyze_labels(labels):
    """
    对标签进行统计分析
    """
    print("\n" + "=" * 80)
    print("标签统计分析")
    print("=" * 80)
    
    # 获取唯一的标签和计数
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 按计数降序排列
    sorted_indices = np.argsort(-counts)
    unique_labels = unique_labels[sorted_indices]
    counts = counts[sorted_indices]
    
    # 创建统计DataFrame
    stats_df = pd.DataFrame({
        '类别': unique_labels,
        '数量': counts,
        '百分比(%)': (counts / len(labels) * 100).round(2)
    })
    
    print(f"\n总序列数: {len(labels)}")
    print(f"总类别数: {len(unique_labels)}\n")
    print(stats_df.to_string(index=False))
    
    return stats_df


def save_sequences_by_label(sequences, labels, output_dir):
    """
    根据标签保存序列数据到npy文件
    参考process_traffic_clustering的保存方法
    """
    print("\n" + "=" * 80)
    print("保存序列数据")
    print("=" * 80)
    
    # 获取唯一的标签
    unique_labels = np.unique(labels)
    
    print(f"\n按标签保存序列数据...")
    
    for label in unique_labels:
        # 获取属于当前标签的序列索引
        label_indices = np.where(labels == label)[0]
        
        # 提取当前标签的序列
        label_sequences = sequences[label_indices]
        
        # 保存为npy文件
        filename = f'{output_dir}/{label}.npy'
        np.save(filename, label_sequences)
        
        print(f"  标签 '{label}': {label_sequences.shape} -> 保存到 {filename}")
    
    # 保存所有标签
    np.save(f'{output_dir}/all_labels.npy', labels)
    print(f"  所有标签 -> 保存到 {output_dir}/all_labels.npy")
    
    print(f"\n数据保存完成！所有文件已保存到 '{output_dir}' 文件夹中")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Traffic 数据集处理与时间序列生成")
    print("=" * 80)
    
    # 初始化处理器
    processor = TrafficDataProcessor("Traffic/data/Metro_Interstate_Traffic_Volume.csv")
    
    # 统一分类特征为小写
    processor.standardize_categorical_features()

    # 清理前打印统计信息
    processor.print_statistics_before_cleaning()

    # 删除异常数据
    processor.remove_abnormal_data(rain_threshold=2000, temp_threshold=200)
    
    # 获取处理后的数据
    df = processor.get_processed_data()
    
    # 生成时间序列
    sequences, indices, all_features = generate_time_series(
        df, seq_len, pred_len, window_size, stride)
    
    # 提取标签
    labels = extract_labels_from_sequences(
        sequences, indices, df, all_features, window_size)
    
    # 分析标签
    stats_df = analyze_labels(labels)
    
    # 保存序列数据
    save_sequences_by_label(sequences, labels, output_dir)
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()