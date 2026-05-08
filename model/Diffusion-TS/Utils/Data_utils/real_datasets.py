import os
import sys
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# 修正路径回溯逻辑，确保跨平台兼容性
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..', '..'))  # 适当回溯到仓库根
sys.path.append(_repo_root)

# 导入外部模块（确保这些模块路径正确）
from TSlib.lib.dataloader import real_data_loading
from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one
)
from Utils.masking_utils import noise_mask
from model.TimeVAE.lib.data import split_data, scale_data


class CustomDataset(Dataset):
    # ===================== 直接在这里放入 feature_map =====================
    feature_map = {
        'etth1': 'S',
        'etth2': 'S',
        'AirQuality(bj)': 'M',
        'AirQuality(Italian)': 'M',
        'Traffic': 'S',
        'FD001': 'MS',
        'FD002': 'MS',
        'FD003': 'MS',
        'FD004': 'MS'
    }

    def __init__(
            self,
            name,
            window=64,
            proportion=0.8,
            save2npy=True,
            neg_one_to_one=True,
            seed=123,
            period='train',
            output_dir='./OUTPUT',
            predict_length=None,
            missing_ratio=None,
            style='separate',
            distribution='geometric',
            mean_mask_length=3
    ):
        super().__init__()
        assert period in ['train', 'test']
        if period == 'train':
            assert not (predict_length is not None or missing_ratio is not None)

        self.name = name
        self.style = style
        self.distribution = distribution
        self.mean_mask_length = mean_mask_length
        self.missing_ratio = missing_ratio

        # ===================== 核心：自动获取当前数据集的 feature 类型 =====================
        self.feature_type = self.feature_map[name]

        # ===================== 加载数据，并传入 feature =====================
        train, _, test, train_g = real_data_loading(self.name, feature=self.feature_type)

        # 数据清洗
        train = self._clean_numeric_data(train)
        test = self._clean_numeric_data(test)
        train_g = self._clean_numeric_data(train_g)



        self.var_num = train_g.shape[-1]
        self.window = window
        self.period = period
        self.save2npy = save2npy

        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        # ===================== ✅ 完全删除标准化，直接使用原始数据 =====================
        # ✅ 模型训练数据 = 小样本 train_g
        self.train_data_g = train_g
        self.raw_data = train_g if period == 'train' else test
        self.samples = self.raw_data

        # 样本数量
        self.sample_num = self.samples.shape[0]

        # 测试集掩码
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones_like(self.samples)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError

    def _clean_numeric_data(self, data):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data[:, :, None]
        elif data.ndim != 3:
            raise ValueError(f"数据维度错误，期望2D/3D，实际{data.ndim}D")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        non_zero_cols = np.any(data != 0, axis=(0, 1))
        data = data[:, :, non_zero_cols]
        return data

    def __getsamples(self, data, proportion, seed):
        N, T, D = data.shape
        assert T >= self.window
        samples = []
        for n in range(N):
            for t in range(T - self.window + 1):
                samples.append(data[n, t:t + self.window, :])
        x = np.stack(samples, axis=0)
        train_data, test_data = self.divide(x, proportion, seed)
        if self.save2npy:
            if len(test_data) > 0:
                np.save(os.path.join(self.dir, f"{self.name}_gt_{self.window}_test.npy"), test_data)
            np.save(os.path.join(self.dir, f"{self.name}_gt_{self.window}_train.npy"), train_data)
        return train_data, test_data

    @staticmethod
    def divide(data, ratio, seed):
        size = data.shape[0]
        st0 = np.random.get_state()
        np.random.seed(seed)
        train_num = int(np.ceil(size * ratio))
        idx = np.arange(size)
        train_data = data[idx[:train_num]]
        test_data = data[idx[train_num:]]
        np.random.set_state(st0)
        return train_data, test_data

    def read_npy(self, filepath):
        assert filepath.endswith(".npy")
        data = np.load(filepath, allow_pickle=True)
        data = self._clean_numeric_data(data)
        return data

    def mask_data(self, seed):
        masks = np.ones_like(self.samples)
        st0 = np.random.get_state()
        np.random.seed(seed)
        for i in range(self.samples.shape[0]):
            masks[i] = noise_mask(self.samples[i], self.missing_ratio, self.mean_mask_length, self.style, self.distribution)
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        x = self.samples[ind]
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.period == 'test':
            m = self.masking[ind]
            m = np.asarray(m, dtype=np.float32)
            return torch.from_numpy(x).float(), torch.from_numpy(m).float()

        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num