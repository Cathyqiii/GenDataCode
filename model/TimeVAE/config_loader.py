import configparser
import os
import torch
from typing import List, Union

class ConfigLoader:
    """
    使用configparser读取配置文件的配置管理类
    替代原有的argparse-based Options类
    """

    def __init__(self, config_path: str):
        # 配置文件绝对路径
        self.config_path = os.path.abspath(config_path)

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # 启用 ExtendedInterpolation
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )

        # 读取配置文件
        self.config.read(self.config_path, encoding="utf-8")

        # 注入 current_dir（关键）
        # current_dir 表示「配置文件所在目录」
        config_dir = os.path.dirname(self.config_path)
        self.config["DEFAULT"]["current_dir"] = config_dir

        # 解析配置
        self._parse_config()

    def _parse_config(self):
        """解析配置文件中的所有参数"""

        self.ROOT_DIR =  os.path.abspath(self.config.get('DEFAULT', 'ROOT_DIR'))
        self.current_dir =  os.path.abspath(self.config.get('DEFAULT', 'current_dir'))
        self.BASE_DIR = os.path.abspath(self.config.get('DATASET', 'BASE_DIR'))
        self.DATASETS_DIR =  os.path.abspath(self.config.get('Paths', 'DATASETS_DIR'))
        self.OUTPUTS_DIR =  os.path.abspath(self.config.get('Paths', 'OUTPUTS_DIR'))
        self.GEN_DATA_DIR =  os.path.abspath(self.config.get('Paths', 'GEN_DATA_DIR'))
        self.MODELS_DIR =  os.path.abspath(self.config.get('Paths', 'MODELS_DIR'))
        self.TSNE_DIR =  os.path.abspath(self.config.get('Paths', 'TSNE_DIR'))
        self.SRC_DIR = os.path.abspath(self.config.get('Paths', 'SRC_DIR'))
        self.CONFIG_DIR = os.path.abspath(self.config.get('Paths', 'CONFIG_DIR'))
        self.HYPERPARAMETERS_FILE_PATH = os.path.abspath(self.config.get('Paths', 'HYPERPARAMETERS_FILE_PATH'))

# 使用示例
if __name__ == "__main__":
    # 创建配置加载器
    config_loader = ConfigLoader("config.ini")


