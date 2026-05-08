# dataset_config.py  【全局唯一配置文件】
# 所有数据集的固定参数，所有脚本统一导入

# 1. 数据集特征映射（唯一来源）
FEATURE_MAP = {
    'etth1': 'S',
    'etth2': 'S',
    'airquality_bj': 'M',
    'airquality_italian': 'M',
    'traffic': 'S',
    'fd001': 'MS',
    'fd002': 'MS',
    'fd003': 'MS',
    'fd004': 'MS'
}

# 2. 生成倍数映射（唯一来源）
GENERATE_TIMES_MAP = {
    'etth1': 5,
    'etth2': 6,
    'airquality_bj': 3,
    'airquality_italian': 2,
    'traffic': 2,
    'fd001': 2,
    'fd002': 2,
    'fd003': 2,
    'fd004': 2
}

# 3. 支持的数据集列表
SUPPORTED_DATASETS = list(FEATURE_MAP.keys())

# 4. 大小写兼容映射（解决etth1/ETTh1不统一问题）
NAME_ALIAS = {
    'ETTh1': 'etth1',
    'ETTh2': 'etth2',
    'AirQuality(bj)': 'airquality_bj',
    'AirQuality(Italian)': 'airquality_italian',
    'Traffic': 'traffic',
    'FD001': 'fd001',
    'FD002': 'fd002',
    'FD003': 'fd003',
    'FD004': 'fd004'
}

# 工具函数：标准化数据集名称（自动转小写规范名）
def standardize_name(data_name):
    return NAME_ALIAS.get(data_name, data_name.lower())