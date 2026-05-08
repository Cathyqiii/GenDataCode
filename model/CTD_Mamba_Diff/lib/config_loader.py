import ast
import configparser
import torch
from typing import List


class ConfigLoader:
    """
    使用 configparser 读取配置文件的配置管理类
    适配简化版配置文件
    """

    def __init__(self, config_path: str = "./config/etth1.conf"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding="utf-8")
        self.config_path = config_path

        # 解析配置
        self._parse_config()

    def _parse_config(self):
        """解析必要的配置参数"""

        # ---------------- Data Section ----------------
        self.data_name = self.config.get('Data', 'data_name')
        self.feature_mode = self.config.get('Data', 'feature_mode')
        self.data_dim = self.config.getint('Data', 'data_dim')
        self.seq_len = self.config.getint('Data', 'seq_len')
        self.condition_dim = self.config.getint('Data', 'condition_dim')

        # ---------------- Model Section ----------------
        self.vae_hidden_dim = self.config.getint('Model', 'vae_hidden_dim')
        self.vae_latent_dim = self.config.getint('Model', 'vae_latent_dim')
        self.embedding_dim = self.config.getint('Model', 'embedding_dim')
        self.mamba_dim = self.config.getint('Model', 'mamba_dim')
        self.hidden_dim = self.config.getint('Model', 'hidden_dim')
        self.scales = ast.literal_eval(self.config.get('Model', 'scales'))  # 用 ast.literal_eval 来处理 Python 数据类型
        self.diffusion_steps = self.config.getint('Model', 'DIFFUSION_STEPS')
        self.mamba_num_layers = self.config.getint('Model', 'MAMBA_NUM_LAYERS')
        # ---------------- Training Section ----------------
        self.batch_size = self.config.getint('Training', 'BATCH_SIZE')
        self.lr = self.config.getfloat('Training', 'LR')
        self.epochs = self.config.getint('Training', 'EPOCHS')
        self.grad_clip = self.config.getfloat('Training', 'grad_clip')

        # 读取学习率调度器及其参数
        self.lr_scheduler_type = self.config.get('Training', 'lr_scheduler')
        self.lr_scheduler_params = ast.literal_eval(self.config.get('Training', 'lr_scheduler_params'))  # 字符串转换为字典
        self.early_stop_patience = self.config.getint('Training', 'early_stop_patience')
        self.early_stop_delta = self.config.getfloat('Training', 'early_stop_delta')
        self.grad_accumulation_steps = self.config.getfloat('Training', 'grad_accumulation_steps')


        # ---------------- Hardware Section ----------------
        self.device = self.config.get('Hardware', 'device')
        self.gpu_index = self.config.getint('Hardware', 'gpu_index')
        self.gpu_ids = [self.gpu_index] if self.device.lower() == 'gpu' else []

        # ---------------- Paths Section ----------------
        self.condition_vae_ckpt = self.config.get('Paths', 'condition_vae_ckpt')
        self.checkpoint_dir = self.config.get('Paths', 'checkpoint_dir')
        self.generated_data_dir = self.config.get('Paths', 'generated_data_dir')

        # ---------------- Generation Section ----------------
        self.generate_times = self.config.getint('Generation', 'generate_times')

        # 设置设备
        self._setup_device()

    def _setup_device(self):
        """设置计算设备（GPU/CPU）"""
        if self.device.lower() == "gpu" and torch.cuda.is_available() and self.gpu_ids:
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = "cpu"

    def to_dict(self):
        """将配置参数转换为字典"""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                if attr_name in ['config', 'config_path']:
                    continue
                val = getattr(self, attr_name)
                if isinstance(val, list):
                    val = ','.join(map(str, val))
                config_dict[attr_name] = val
        return config_dict

    def update_config(self, section: str, key: str, value: str):
        """更新配置参数"""
        if self.config.has_section(section):
            self.config.set(section, key, value)
            self._parse_config()

    def save_config(self, config_path: str = None):
        """保存配置到文件"""
        if config_path is None:
            config_path = self.config_path
        with open(config_path, 'w') as f:
            self.config.write(f)
        print(f"配置已保存到: {config_path}")

    def __str__(self):
        """便于调试"""
        sections_str = []
        for section in self.config.sections():
            sections_str.append(f"[{section}]")
            for key, value in self.config.items(section):
                sections_str.append(f"  {key}: {value}")
            sections_str.append("")
        return "\n".join(sections_str)


# # ------------------ 测试代码 ------------------
# if __name__ == "__main__":
#     config_loader = ConfigLoader("../config/etth1.conf")
#
#     print("=" * 50)
#     print("CTD_Mamba_Diff 配置参数")
#     print("=" * 50)
#     print(f"数据集: {config_loader.data_name}")
#     print(f"序列长度: {config_loader.seq_len}")
#     print(f"批次大小: {config_loader.batch_size}")
#     print(f"学习率: {config_loader.lr}")
#     print(f"GPU IDs: {config_loader.gpu_ids}")
#     print(f"Mamba维度: {config_loader.mamba_dim}")
#     print(f"扩散步数: {config_loader.diffusion_steps}")
#     print(f"生成样本数: {config_loader.num_generated_samples}")
#
#     # 获取字典形式
#     opt_dict = config_loader.to_dict()
#     print(f"\n配置参数总数: {len(opt_dict)}")
#
#     print("\n前10个参数:")
#     for k, v in list(opt_dict.items())[:10]:
#         print(f"  {k}: {v}")
