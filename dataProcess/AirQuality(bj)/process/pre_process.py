import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取文件（改为你的文件路径）
fn = "AirQuality(北京)/data/PRSA_Data_Aotizhongxin_20130301-20170228.csv"
df = pd.read_csv(fn)
# 1. 删除第一列（No）和最后一列（station），使用列名匹配以提高稳健性
col_lower_map = {c.lower(): c for c in df.columns}
if 'no' in col_lower_map:
    df.drop(col_lower_map['no'], axis=1, inplace=True)
elif df.columns.size > 0 and df.columns[0].lower().startswith('no'):
    df.drop(df.columns[0], axis=1, inplace=True)
if 'station' in col_lower_map:
    df.drop(col_lower_map['station'], axis=1, inplace=True)

# 2. 合并 year, month, day, hour 为 time（datetime）
# 假定列名为 'year','month','day','hour'（如果不同请改列名）
df['time'] = pd.to_datetime(
    df['year'].astype(str).str.zfill(4) + '-' +
    df['month'].astype(str).str.zfill(2) + '-' +
    df['day'].astype(str).str.zfill(2) + ' ' +
    df['hour'].astype(str).str.zfill(2) + ':00:00',
    errors='coerce'
)
# 删除原来的年月日小时列
for c in ['year', 'month', 'day', 'hour']:
    if c in df.columns:
        df.drop(c, axis=1, inplace=True)

# 4. 缺失值处理：首先可视化，然后按时间排序并前向填充
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing values heatmap (Before processing)')
plt.tight_layout()
plt.savefig("AirQuality(北京)/output/missing_heatmap_before.png", dpi=100, bbox_inches='tight')
plt.show()

# 按时间排序并前向填充
if 'time' in df.columns:
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
df.fillna(method='ffill', inplace=True)

# ==============================
# CO 单位转换：÷1000
# ==============================
# ==============================
# CO 单位转换：÷1000（按列名）
# ==============================

co_col = None

for col in df.columns:
    if col.lower() == 'co':
        co_col = col
        break

if co_col is not None:
    df[co_col] = pd.to_numeric(df[co_col], errors='coerce') / 1000
    print(f"{co_col} 已除以1000")
else:
    print("未找到 CO 列")


# 3. 风向信息编码为 16 方位角度，再转为 sin/cos
# 16 方位映射（角度），包含常见英文缩写和常见中文方向词
dir16 = {
    'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
    'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
    'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
    'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5,
    # 常见中文方向（简化到8/16方位的常用词）
    '北': 0.0, '北偏东': 22.5, '东北': 45.0, '东偏北': 67.5,
    '东': 90.0, '东偏南': 112.5, '东南': 135.0, '南偏东': 157.5,
    '南': 180.0, '南偏西': 202.5, '西南': 225.0, '西偏南': 247.5,
    '西': 270.0, '西偏北': 292.5, '西北': 315.0, '北偏西': 337.5,
    '静风': np.nan, 'calm': np.nan
}
# 尝试自动找到风向的列名（常见 cbwd, wd, wind_dir 等）
wind_col = None
for cand in df.columns:
    if cand.lower() in ('cbwd', 'wd', 'wind_dir', 'wind', 'direction'):
        wind_col = cand
        break
if wind_col is None:
    # 若未找到，尝试包含关键字
    for cand in df.columns:
        if 'wd' in cand.lower() or 'wind' in cand.lower() or 'dir' in cand.lower():
            wind_col = cand
            break

if wind_col is not None:
    # 先标准化为字符串并剔除空白
    wd_series = df[wind_col].where(df[wind_col].notnull(), other=np.nan)

    def map_wd_to_angle(x):
        if pd.isnull(x):
            return np.nan
        s = str(x).strip()
        # 先尝试英文缩写（大写）
        angle = dir16.get(s.upper()) if isinstance(s, str) else None
        if angle is not None:
            return angle
        # 再尝试数字角度字符串
        try:
            v = float(s)
            if 0 <= v <= 360:
                return v
        except:
            pass
        # 最后尝试中文/原始匹配（保持原样的键）
        angle = dir16.get(s)
        if angle is not None:
            return angle
        return np.nan

    df['wind_angle'] = wd_series.map(map_wd_to_angle)
    # 计算 sin/cos（对 NaN 结果保持 NaN）
    df['wind_sin'] = np.sin(np.deg2rad(df['wind_angle'].astype(float)))
    df['wind_cos'] = np.cos(np.deg2rad(df['wind_angle'].astype(float)))
else:
    print("未找到风向列，跳过风向编码步骤。")

# 5. AQI 编码：计算单一污染物 IAQI 并取最大为总 AQI（仅保存总 AQI）
def calc_iaqi_1h(pollutant, Cp):
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
        "CO_1h": [
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
        raise ValueError("不支持的污染物名称")
    
    bps = AQI_BREAKPOINTS[pollutant]


     # 1 正常区间内插值
    for BP_low, BP_high, IAQI_low, IAQI_high in AQI_BREAKPOINTS[pollutant]:
        if BP_low <= Cp <= BP_high:
            iaqi = (IAQI_high - IAQI_low) / (BP_high - BP_low) * (Cp - BP_low) + IAQI_low
            return round(iaqi)
        
    # 2 低于最小值
    if Cp < bps[0][0]:
        return 0

    # 3 高于最大值 → 线性外推
    BP_low, BP_high, IAQI_low, IAQI_high = bps[-1]

    slope = (IAQI_high - IAQI_low) / (BP_high - BP_low)

    iaqi = IAQI_high + slope * (Cp - BP_high)

    return round(iaqi)


# 尝试找到数据中对应的污染物列并映射到 AQI 表的 key
col_lower_map = {c.lower(): c for c in df.columns}
pollutant_columns = {}
lookup = {
    "SO2_1h": ["so2"],
    "NO2_1h": ["no2"],
    "CO_1h": ["co"],
    "O3_1h": ["o3"],
    "PM10_1h": ["pm10", "pm_10"],
    "PM25_1h": ["pm2.5", "pm25", "pm_2.5", "pm_2_5"]
}
for key, keywords in lookup.items():
    found = None
    for k in keywords:
        for col_low, col_orig in col_lower_map.items():
            if k in col_low:
                found = col_orig
                break
        if found:
            break
    if found:
        pollutant_columns[key] = found

# 为每个找到的污染物计算 IAQI 列（先数值化，避免异常字符串导致的错误）
iaqi_cols = []
for pkey, colname in pollutant_columns.items():
    iaqi_col = f"{pkey}_IAQI"
    iaqi_cols.append(iaqi_col)
    numeric = pd.to_numeric(df[colname], errors='coerce')
    df[iaqi_col] = numeric.apply(lambda v: calc_iaqi_1h(pkey, float(v)) if pd.notnull(v) else np.nan)

# 总 AQI 为各污染物 IAQI 的最大值
if iaqi_cols:
    df['AQI'] = df[iaqi_cols].max(axis=1)
    # 删除中间 IAQI 列（只保留总 AQI）
    df.drop(columns=iaqi_cols, inplace=True)
else:
    df['AQI'] = np.nan
    print("未找到任何可用于计算 IAQI 的污染物列。")

# 保存处理后的文件
df.to_csv("AirQuality(北京)/data/processed_PRSA.csv", index=False)
print("预处理完成，结果已保存为 processed_PRSA.csv")