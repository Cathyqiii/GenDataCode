"""
AirQuality 数据处理脚本
支持多种缺失值填补方式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 全局配置参数 ====================
class Config:
    # 一般缺失值填补方式: 'forward_fill', 'backward_fill', 'mean'
    GENERAL_FILL_METHOD = 'forward_fill'
    
    # 输入输出路径
    INPUT_FILE = r'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\data\AirQualityUCI.xlsx'
    OUTPUT_DELETE = r'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\data\AirQualityUCI_delete.xlsx'
    OUTPUT_FINAL = r'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\data\AirQualityUCI_final.xlsx'
    
    # 缺失值标记
    MISSING_VALUE = -200.0
    
    # 特殊字段处理配置: 格式为 (col1, col2, model_type)
    # model_type 可选: 'linear', 'log', 'exp', 'general'
    SPECIAL_FIELDS_CONFIG = {
        ('CO(GT)', 'PT08.S1(CO)'): 'linear',        # 线性回归
        ('NOx(GT)', 'PT08.S3(NOx)'): 'exp',         # 指数回归
        ('NO2(GT)', 'PT08.S4(NO2)'): 'general',     # 一般方法填补
    }


class AirQualityProcessor:
    """空气质量数据处理类"""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.df = None
        self.df_deleted = None
        self.df_final = None
        self.regression_models = {}
        
    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("步骤 1: 加载数据")
        print("=" * 60)
        self.df = pd.read_excel(self.config.INPUT_FILE)
        print(f"✓ 加载成功, 形状: {self.df.shape}")
        print(f"✓ 列: {self.df.columns.tolist()}")
        return self.df
    
    def basic_cleanup(self):
        """基本清理: 删除 NMHC(GT) 字段，删除缺失值过多的行"""
        print("\n" + "=" * 60)
        print("步骤 2: 基本清理")
        print("=" * 60)
        
        # 将 -200 替换为 NaN（便于处理）
        self.df = self.df.replace(self.config.MISSING_VALUE, np.nan)
        
        # 删除 NMHC(GT) 字段
        if 'NMHC(GT)' in self.df.columns:
            self.df = self.df.drop('NMHC(GT)', axis=1)
            print("✓ 删除 NMHC(GT) 字段")
        
        # 计算每行缺失值数量（不包括 Date 和 Time）
        data_cols = self.df.columns[2:]  # 跳过 Date 和 Time
        missing_per_row = self.df[data_cols].isnull().sum(axis=1)
        
        # 删除缺失值数量 > 2 的行
        rows_to_drop = missing_per_row[missing_per_row > 2].index
        print(f"✓ 原始数据行数: {len(self.df)}")
        print(f"✓ 删除缺失值 > 2 的行数: {len(rows_to_drop)}")
        
        self.df_deleted = self.df.drop(rows_to_drop).reset_index(drop=True)
        print(f"✓ 清理后数据行数: {len(self.df_deleted)}")
        
        # 保存中间结果
        self.df_deleted.to_excel(self.config.OUTPUT_DELETE, index=False)
        print(f"✓ 保存到: {self.config.OUTPUT_DELETE}")
        
        return self.df_deleted
    
    def visualize_missing(self):
        """可视化缺失值"""
        print("\n" + "=" * 60)
        print("步骤 3: 可视化缺失值")
        print("=" * 60)
        
        # 创建缺失值矩阵
        data_cols = self.df_deleted.columns[2:]  # 跳过 Date 和 Time
        missing_matrix = self.df_deleted[data_cols].isnull().astype(int)
        
        # 绘制热力图
        plt.figure(figsize=(14, 6))
        sns.heatmap(missing_matrix.T, cbar=True, cmap='RdYlGn_r', 
                    yticklabels=True, xticklabels=False)
        plt.title('缺失值分布（红色为缺失值，绿色为有效值）')
        plt.xlabel('样本索引')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig(
            r'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\output\missing_heatmap.png',
            dpi=100, bbox_inches='tight'
        )
        print("✓ 缺失值热力图已保存")
        
        # 打印缺失值统计
        print("\n缺失值统计:")
        missing_counts = self.df_deleted[data_cols].isnull().sum()
        for col, count in missing_counts.items():
            print(f"  {col}: {count} ({count/len(self.df_deleted)*100:.2f}%)")
    
    def build_regression_model(self, col1, col2, model_type='linear'):
        """
        建立两个列之间的回归模型
        col1: 目标变量 (GT)
        col2: 特征变量 (传感器读数)
        model_type: 'linear', 'log', 或 'exp'
        """
        print(f"\n  建立{model_type}回归模型: {col1} ~ {col2}")
        
        # 获取不含缺失值的数据
        valid_mask = (~self.df_deleted[col1].isnull()) & (~self.df_deleted[col2].isnull())
        X = self.df_deleted.loc[valid_mask, [col2]].values.flatten()
        y = self.df_deleted.loc[valid_mask, col1].values
        
        if len(X) < 2:
            print(f"  ⚠ 缺乏有效数据，无法建立回归模型")
            return None
        
        if model_type == 'linear':
            return self._build_linear_model(col1, col2, X, y)
        elif model_type == 'log':
            return self._build_log_model(col1, col2, X, y)
        elif model_type == 'exp':
            return self._build_exp_model(col1, col2, X, y)
        else:
            return None
    
    def _build_linear_model(self, col1, col2, X, y):
        """线性回归: y = a*x + b"""
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        
        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X.reshape(-1, 1), y)
        
        print(f"  回归方程: {col1} = {coef:.6f} * {col2} + {intercept:.6f}")
        print(f"  R² = {r2:.6f}")
        print(f"  样本数: {len(X)}")
        
        self._plot_regression_linear(col1, col2, X, y, coef, intercept, r2)
        
        return {'type': 'linear', 'coef': coef, 'intercept': intercept, 'r2': r2}
    
    def _build_log_model(self, col1, col2, X, y):
        """对数回归: y = a*ln(x) + b"""
        # 过滤掉 X <= 0 的数据
        valid_x = X > 0
        X_valid = X[valid_x]
        y_valid = y[valid_x]
        
        if len(X_valid) < 2:
            print(f"  ⚠ 有效数据不足，无法建立对数回归模型")
            return None
        
        X_log = np.log(X_valid)
        model = LinearRegression()
        model.fit(X_log.reshape(-1, 1), y_valid)
        
        coef = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X_log.reshape(-1, 1), y_valid)
        
        print(f"  回归方程: {col1} = {coef:.6f} * ln({col2}) + {intercept:.6f}")
        print(f"  R² = {r2:.6f}")
        print(f"  样本数: {len(X_valid)}")
        
        self._plot_regression_log(col1, col2, X_valid, y_valid, coef, intercept, r2)
        
        return {'type': 'log', 'coef': coef, 'intercept': intercept, 'r2': r2}
    
    def _build_exp_model(self, col1, col2, X, y):
        """指数回归: y = a * exp(b*x)"""
        # 过滤掉 y <= 0 的数据
        valid_y = y > 0
        X_valid = X[valid_y]
        y_valid = y[valid_y]
        
        if len(X_valid) < 2:
            print(f"  ⚠ 有效数据不足，无法建立指数回归模型")
            return None
        
        y_log = np.log(y_valid)
        model = LinearRegression()
        model.fit(X_valid.reshape(-1, 1), y_log)
        
        coef = model.coef_[0]
        intercept = model.intercept_
        a = np.exp(intercept)
        
        # 计算 R²
        y_pred_log = model.predict(X_valid.reshape(-1, 1))
        y_pred = a * np.exp(coef * X_valid)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"  回归方程: {col1} = {a:.6f} * exp({coef:.6f} * {col2})")
        print(f"  R² = {r2:.6f}")
        print(f"  样本数: {len(X_valid)}")
        
        self._plot_regression_exp(col1, col2, X_valid, y_valid, a, coef, r2)
        
        return {'type': 'exp', 'a': a, 'coef': coef, 'r2': r2}
    
    def _plot_regression_linear(self, col1, col2, X, y, coef, intercept, r2):
        """绘制线性回归图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5, label='实际数据')
        
        X_line = np.linspace(X.min(), X.max(), 100)
        y_pred = coef * X_line + intercept
        plt.plot(X_line, y_pred, 'r-', linewidth=2, label=f'拟合线 (R²={r2:.4f})')
        
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.title(f'线性回归: {col1} vs {col2}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{col1}_vs_{col2}_linear_regression.png"
        filepath = rf'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\output\{filename}'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 线性回归图已保存: {filename}")
    
    def _plot_regression_log(self, col1, col2, X, y, coef, intercept, r2):
        """绘制对数回归图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5, label='实际数据')
        
        X_line = np.linspace(X.min(), X.max(), 100)
        y_pred = coef * np.log(X_line) + intercept
        plt.plot(X_line, y_pred, 'r-', linewidth=2, label=f'拟合线 (R²={r2:.4f})')
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.title(f'对数回归: {col1} vs {col2}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{col1}_vs_{col2}_log_regression.png"
        filepath = rf'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\output\{filename}'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 对数回归图已保存: {filename}")
    
    def _plot_regression_exp(self, col1, col2, X, y, a, coef, r2):
        """绘制指数回归图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5, label='实际数据')
        
        X_line = np.linspace(X.min(), X.max(), 100)
        y_pred = a * np.exp(coef * X_line)
        plt.plot(X_line, y_pred, 'r-', linewidth=2, label=f'拟合线 (R²={r2:.4f})')
        
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.title(f'指数回归: {col1} vs {col2}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{col1}_vs_{col2}_exp_regression.png"
        filepath = rf'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\output\{filename}'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 指数回归图已保存: {filename}")
    
    def _plot_regression(self, col1, col2, X, y, model, coef, intercept, r2):
        """绘制回归图 (已弃用，保留兼容性)"""
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5, label='实际数据')
        
        # 绘制拟合线
        X_line = np.array([[X.min()], [X.max()]])
        y_pred = model.predict(X_line)
        plt.plot(X_line, y_pred, 'r-', linewidth=2, label=f'拟合线 (R²={r2:.4f})')
        
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.title(f'回归分析: {col1} vs {col2}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        filename = f"{col1}_vs_{col2}_regression.png"
        filepath = rf'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\output\{filename}'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 回归图已保存: {filename}")
    
    def handle_missing_values(self):
        """处理缺失值"""
        print("\n" + "=" * 60)
        print("步骤 4: 处理缺失值")
        print("=" * 60)
        print(f"处理方式配置:")
        for (col1, col2), model_type in self.config.SPECIAL_FIELDS_CONFIG.items():
            print(f"  {col1} & {col2}: {model_type}")
        print(f"  其他字段: {self.config.GENERAL_FILL_METHOD}")
        
        self.df_final = self.df_deleted.copy()
        
        # 获取所有数据列 (除 Date 和 Time)
        data_cols = set(self.df_final.columns[2:])
        special_cols = set()
        
        for (col1, col2), _ in self.config.SPECIAL_FIELDS_CONFIG.items():
            special_cols.add(col1)
            special_cols.add(col2)
        
        general_cols = data_cols - special_cols
        
        # 处理特殊字段
        print("\n处理特殊字段对...")
        for (col1, col2), model_type in self.config.SPECIAL_FIELDS_CONFIG.items():
            if model_type == 'general':
                print(f"\n{col1} & {col2}: 使用一般方法填补")
                self._fill_general_field([col1, col2])
            else:
                print(f"\n{col1} & {col2}: 使用{model_type}回归填补")
                self._handle_special_field_regression(col1, col2, model_type)
        
        # 处理一般字段
        print(f"\n使用 {self.config.GENERAL_FILL_METHOD} 方法处理一般字段...")
        self._handle_general_fields(general_cols)
        
        print(f"\n✓ 缺失值处理完成")
        print(f"  处理后缺失值数: {self.df_final.isnull().sum().sum()}")
    
    def _handle_special_field_regression(self, col1, col2, model_type):
        """使用回归方法处理特殊字段对"""
        print(f"  处理对: {col1} 和 {col2}")
        
        # 建立回归模型
        result = self.build_regression_model(col1, col2, model_type)
        if result is None:
            print(f"  ⚠ 无法建立{model_type}回归模型，切换为一般处理")
            self._fill_general_field([col1, col2])
            return
        
        # 应用回归填补
        for idx in self.df_final.index:
            val1 = self.df_final.at[idx, col1]
            val2 = self.df_final.at[idx, col2]
            
            # 两个都缺失 -> 使用一般方法
            if pd.isnull(val1) and pd.isnull(val2):
                continue
            
            # col1 缺失，col2 有效 -> 用 col2 预测 col1
            elif pd.isnull(val1) and not pd.isnull(val2):
                if model_type == 'linear':
                    pred = result['coef'] * val2 + result['intercept']
                    self.df_final.at[idx, col1] = pred
                elif model_type == 'log':
                    if val2 > 0:
                        pred = result['coef'] * np.log(val2) + result['intercept']
                        self.df_final.at[idx, col1] = pred
                elif model_type == 'exp':
                    pred = result['a'] * np.exp(result['coef'] * val2)
                    self.df_final.at[idx, col1] = pred
            
            # col2 缺失，col1 有效 -> 用反演方程预测 col2
            elif not pd.isnull(val1) and pd.isnull(val2):
                if model_type == 'linear':
                    if abs(result['coef']) > 1e-10:
                        pred = (val1 - result['intercept']) / result['coef']
                        self.df_final.at[idx, col2] = pred
                elif model_type == 'log':
                    if abs(result['coef']) > 1e-10:
                        pred = np.exp((val1 - result['intercept']) / result['coef'])
                        self.df_final.at[idx, col2] = pred
                elif model_type == 'exp':
                    if abs(result['coef']) > 1e-10 and result['a'] > 0:
                        pred = np.log(val1 / result['a']) / result['coef']
                        self.df_final.at[idx, col2] = pred
        
        print(f"  ✓ 已应用{model_type}回归填补")
    
    def _fill_general_field(self, cols):
        """对一般字段进行填补"""
        for col in cols:
            if col not in self.df_final.columns:
                continue
            
            if self.config.GENERAL_FILL_METHOD == 'forward_fill':
                self.df_final[col].fillna(method='ffill', inplace=True)
                self.df_final[col].fillna(method='bfill', inplace=True)
            
            elif self.config.GENERAL_FILL_METHOD == 'backward_fill':
                self.df_final[col].fillna(method='bfill', inplace=True)
                self.df_final[col].fillna(method='ffill', inplace=True)
            
            elif self.config.GENERAL_FILL_METHOD == 'mean':
                mean_val = self.df_final[col].mean()
                self.df_final[col].fillna(mean_val, inplace=True)
    
    def _handle_general_fields(self, general_cols):
        """处理一般字段的缺失值"""
        for col in general_cols:
            if col not in self.df_final.columns:
                continue
            
            if self.config.GENERAL_FILL_METHOD == 'forward_fill':
                self.df_final[col].fillna(method='ffill', inplace=True)
                self.df_final[col].fillna(method='bfill', inplace=True)
            
            elif self.config.GENERAL_FILL_METHOD == 'backward_fill':
                self.df_final[col].fillna(method='bfill', inplace=True)
                self.df_final[col].fillna(method='ffill', inplace=True)
            
            elif self.config.GENERAL_FILL_METHOD == 'mean':
                mean_val = self.df_final[col].mean()
                self.df_final[col].fillna(mean_val, inplace=True)
    
    def calc_iaqi_1h(self, pollutant, Cp):
        """
        计算单一污染物的一小时 IAQI（按 HJ 633-2012）
        pollutant: 污染物名称（字符串）
        Cp: 浓度值（float）
        """
        # 一小时 AQI 分级表（严格按标准）
        AQI_BREAKPOINTS = {
            "SO2_1h": [
                (0, 150, 0, 50),
                (150, 500, 50, 100),
                (500, 650, 100, 150),
                (650, 800, 150, 200),
                (800, 1600, 200, 300),
                (1600, 2100, 300, 400),
                (2100, 2620, 400, 500),
            ],
            "NO2_1h": [
                (0, 100, 0, 50),
                (100, 200, 50, 100),
                (200, 700, 100, 150),
                (700, 1200, 150, 200),
                (1200, 2340, 200, 300),
                (2340, 3090, 300, 400),
                (3090, 3840, 400, 500),
            ],
            "CO_1h": [  # 单位：mg/m3
                (0, 5, 0, 50),
                (5, 10, 50, 100),
                (10, 35, 100, 150),
                (35, 60, 150, 200),
                (60, 90, 200, 300),
                (90, 120, 300, 400),
                (120, 150, 400, 500),
            ],
            "O3_1h": [
                (0, 160, 0, 50),
                (160, 200, 50, 100),
                (200, 300, 100, 150),
                (300, 400, 150, 200),
                (400, 800, 200, 300),
                (800, 1000, 300, 400),
                (1000, 1200, 400, 500),
            ],
            "PM10_1h": [
                (0, 50, 0, 50),
                (50, 150, 50, 100),
                (150, 250, 100, 150),
                (250, 350, 150, 200),
                (350, 420, 200, 300),
                (420, 500, 300, 400),
                (500, 600, 400, 500),
            ],
            "PM25_1h": [
                (0, 35, 0, 50),
                (35, 75, 50, 100),
                (75, 115, 100, 150),
                (115, 150, 150, 200),
                (150, 250, 200, 300),
                (250, 350, 300, 400),
                (350, 500, 400, 500),
            ],
        }

        if pollutant not in AQI_BREAKPOINTS:
            raise ValueError(f"不支持的污染物名称: {pollutant}")

        for BP_low, BP_high, IAQI_low, IAQI_high in AQI_BREAKPOINTS[pollutant]:
            if BP_low <= Cp <= BP_high:
                iaqi = (IAQI_high - IAQI_low) / (BP_high - BP_low) * (Cp - BP_low) + IAQI_low
                return round(iaqi)

        # 超过最大量程，直接封顶
        return 500
    
    def save_final_data(self):
        """保存最终处理数据"""
        print("\n" + "=" * 60)
        print("步骤 5: 保存最终数据")
        print("=" * 60)
        
        # 计算AQI指数
        print("\n计算AQI指数...")
        self.df_final['CO_IAQI'] = self.df_final['CO(GT)'].apply(lambda x: self.calc_iaqi_1h('CO_1h', x))
        self.df_final['NO2_IAQI'] = self.df_final['NO2(GT)'].apply(lambda x: self.calc_iaqi_1h('NO2_1h', x))
        self.df_final['AQI'] = self.df_final[['CO_IAQI', 'NO2_IAQI']].max(axis=1)
        print("✓ AQI指数计算完成")
        
        # # 打印AQI统计信息
        # print(f"\nAQI统计信息:")
        # print(f"  CO_IAQI: min={self.df_final['CO_IAQI'].min()}, max={self.df_final['CO_IAQI'].max()}, mean={self.df_final['CO_IAQI'].mean():.2f}")
        # print(f"  NO2_IAQI: min={self.df_final['NO2_IAQI'].min()}, max={self.df_final['NO2_IAQI'].max()}, mean={self.df_final['NO2_IAQI'].mean():.2f}")
        # print(f"  AQI: min={self.df_final['AQI'].min()}, max={self.df_final['AQI'].max()}, mean={self.df_final['AQI'].mean():.2f}")
        
        self.df_final.to_excel(self.config.OUTPUT_FINAL, index=False)
        print(f"\n✓ 保存到: {self.config.OUTPUT_FINAL}")
        print(f"✓ 最终数据形状: {self.df_final.shape}")
        print(f"✓ 缺失值总数: {self.df_final.isnull().sum().sum()}")
    
    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "=" * 60)
        print("处理摘要")
        print("=" * 60)
        print(f"原始数据形状: {self.df.shape}")
        print(f"删除过多缺失值后: {self.df_deleted.shape}")
        print(f"最终处理后: {self.df_final.shape}")
        print(f"\n处理参数:")
        print(f"  缺失值标记: {self.config.MISSING_VALUE}")
        print(f"  特殊字段配置:")
        for (col1, col2), model_type in self.config.SPECIAL_FIELDS_CONFIG.items():
            print(f"    {col1} & {col2}: {model_type}")
        print(f"  一般字段填补: {self.config.GENERAL_FILL_METHOD}")
        print("=" * 60)
    
    def process(self):
        """执行完整处理流程"""
        self.load_data()
        self.basic_cleanup()
        self.visualize_missing()
        self.handle_missing_values()
        self.save_final_data()
        self.print_summary()


if __name__ == '__main__':
    # 创建输出目录
    import os
    os.makedirs(r'd:\useless\研究生\时间序列生成与增强\数据处理\AirQuality(Italian)\output', 
                exist_ok=True)
    
    # 设置全局参数
    config = Config()
    

    config.SPECIAL_FIELDS_CONFIG = {
        ('CO(GT)', 'PT08.S1(CO)'): 'linear',        
        ('NOx(GT)', 'PT08.S3(NOx)'): 'exp',       
        ('NO2(GT)', 'PT08.S4(NO2)'): 'general',  
    }
    config.GENERAL_FILL_METHOD = 'forward_fill'    # 其他字段用前值填补
    
    # 可选配置模式：
    # - 'linear': 线性回归 y = a*x + b
    # - 'log': 对数回归 y = a*ln(x) + b
    # - 'exp': 指数回归 y = a*exp(b*x)
    # - 'general': 使用 GENERAL_FILL_METHOD 进行填补
    
    # 执行处理
    processor = AirQualityProcessor(config)
    processor.process()
    
    print("\n✓ 所有处理完成！")
