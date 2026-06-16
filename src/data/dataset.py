"""PyTorch Dataset for Precipitation Forecasting."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple

class PrecipitationDataset(Dataset):
    """降水预报数据集"""

    def __init__(self, data_config: Dict, split: str = 'train'):
        super().__init__()
        self.data_config = data_config
        self.split = split
        self.indices = self._load_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        # 返回模拟数据用于测试
        era5_data = torch.randn(10, 64, 64)  # 10个ERA5变量
        gmcp_data = torch.randn(1, 128, 128)  # GMCP降水
        target = torch.randn(1, 128, 128)  # 目标降水

        return {
            'era5': era5_data,
            'gmcp': gmcp_data,
            'target': target
        }

    def _load_indices(self) -> List:
        return list(range(1000))  # 模拟1000个样本
