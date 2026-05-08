import configparser
import os
import torch
from typing import List, Union

class ConfigLoader:
    """
    使用configparser读取配置文件的配置管理类
    替代原有的argparse-based Options类
    """
    
    def __init__(self, config_path: str = "./config/etth1.conf"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding="utf-8")
        self.config_path = config_path
        
        # 解析配置并设置属性
        self._parse_config()
    
    def _parse_config(self):
        """解析配置文件中的所有参数"""
        
        # Data Section
        self.data_name = self.config.get('Data', 'data_name')
        self.z_dim = self.config.getint('Data', 'z_dim')
        self.seq_len = self.config.getint('Data', 'seq_len')
        
        # Model Section
        self.model = self.config.get('Model', 'model')
        self.module = self.config.get('Model', 'module')
        self.hidden_dim = self.config.getint('Model', 'hidden_dim')
        self.num_layer = self.config.getint('Model', 'num_layer')
        
        # Training Section
        self.iteration = self.config.getint('Training', 'iteration')
        self.batch_size = self.config.getint('Training', 'batch_size')
        self.metric_iteration = self.config.getint('Training', 'metric_iteration')
        self.isTrain = self.config.getboolean('Training', 'isTrain')
        self.lr = self.config.getfloat('Training', 'lr')
        self.beta1 = self.config.getfloat('Training', 'beta1')
        self.w_gamma = self.config.getfloat('Training', 'w_gamma')
        self.w_es = self.config.getfloat('Training', 'w_es')
        self.w_e0 = self.config.getfloat('Training', 'w_e0')
        self.w_g = self.config.getfloat('Training', 'w_g')
        self.print_freq = self.config.getint('Training', 'print_freq')
        self.load_weights = self.config.getboolean('Training', 'load_weights')
        self.resume = self.config.get('Training', 'resume')
        self.display = self.config.getboolean('Training', 'display')
        
        # System Section
        self.workers = self.config.getint('System', 'workers')
        self.device = self.config.get('System', 'device')
        self.gpu_ids_str = self.config.get('System', 'gpu_ids')
        self.ngpu = self.config.getint('System', 'ngpu')
        self.outf = self.config.get('System', 'outf')
        self.name = self.config.get('System', 'name')
        self.manualseed = self.config.getint('System', 'manualseed')
        
        # Visdom Section
        self.display_server = self.config.get('Visdom', 'display_server')
        self.display_port = self.config.getint('Visdom', 'display_port')
        self.display_id = self.config.getint('Visdom', 'display_id')
        
        # 处理GPU IDs（字符串转列表）
        self.gpu_ids = self._parse_gpu_ids(self.gpu_ids_str)
        
        # 设置设备
        self._setup_device()
        
        # 创建输出目录并保存配置
        self._setup_output()
    
    def _parse_gpu_ids(self, gpu_ids_str: str) -> List[int]:
        """解析GPU ID字符串（如'0,1,2'）为整数列表"""
        if gpu_ids_str.strip() == '':
            return []
        
        str_ids = gpu_ids_str.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id.strip())
            if id >= 0:
                gpu_ids.append(id)
        return gpu_ids
    
    def _setup_device(self):
        """设置计算设备（GPU/CPU）"""
        if self.device == "gpu" and torch.cuda.is_available() and self.gpu_ids:
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = "cpu"
    
    def _setup_output(self):
        """创建输出目录并保存配置文件副本"""
        # 设置实验名称
        if self.name == "experiment_name":
            self.name = f"{self.model}/{self.data_name}"
        
        # 创建输出目录
        expr_dir = os.path.join(self.outf, self.name)
        os.makedirs(expr_dir, exist_ok=True)
        
        # # 保存配置文件副本
        # config_copy_path = os.path.join(expr_dir, "config_backup.ini")
        # with open(config_copy_path, 'w') as f:
        #     self.config.write(f)
        
        # 保存参数文本文件（与原Options类保持一致）
        # file_name = os.path.join(expr_dir, "opt.txt")
        # with open(file_name, "wt") as opt_file:
        #     opt_file.write("------------ Options -------------\n")
        #     for k, v in sorted(self.to_dict().items()):
        #         opt_file.write("%s: %s\n" % (str(k), str(v)))
        #     opt_file.write("-------------- End ----------------\n")
    
    def to_dict(self):
        """将配置参数转换为字典（与原Options类兼容）"""
        return {
            'data_name': self.data_name,
            'z_dim': self.z_dim,
            'seq_len': self.seq_len,
            'model': self.model,
            'module': self.module,
            'hidden_dim': self.hidden_dim,
            'num_layer': self.num_layer,
            'iteration': self.iteration,
            'batch_size': self.batch_size,
            'metric_iteration': self.metric_iteration,
            'isTrain': self.isTrain,
            'lr': self.lr,
            'beta1': self.beta1,
            'w_gamma': self.w_gamma,
            'w_es': self.w_es,
            'w_e0': self.w_e0,
            'w_g': self.w_g,
            'print_freq': self.print_freq,
            'load_weights': self.load_weights,
            'resume': self.resume,
            'display': self.display,
            'workers': self.workers,
            'device': self.device,
            'gpu_ids': self.gpu_ids,
            'ngpu': self.ngpu,
            'outf': self.outf,
            'name': self.name,
            'manualseed': self.manualseed,
            'display_server': self.display_server,
            'display_port': self.display_port,
            'display_id': self.display_id
        }
    
    def __str__(self):
        """字符串表示，便于调试"""
        return "\n".join([f"{k}: {v}" for k, v in self.to_dict().items()])


# 使用示例
if __name__ == "__main__":
    # 创建配置加载器
    config_loader = ConfigLoader("config.ini")
    
    # 访问配置参数
    print(f"数据集: {config_loader.data_name}")
    print(f"批次大小: {config_loader.batch_size}")
    print(f"学习率: {config_loader.lr}")
    print(f"GPU IDs: {config_loader.gpu_ids}")
    
    # 获取字典形式（与原代码兼容）
    opt_dict = config_loader.to_dict()
    print(f"\n字典形式的前5个参数:")
    for k, v in list(opt_dict.items())[:5]:
        print(f"  {k}: {v}")