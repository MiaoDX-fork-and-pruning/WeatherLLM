"""Data Loader for Precipitation Forecasting."""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional

class PrecipitationDataLoader:
    """降水预报数据加载器"""

    def __init__(self, data_config: Dict, training_config: Dict, distributed: bool = False):
        self.data_config = data_config
        self.training_config = training_config
        self.distributed = distributed

        from src.data.dataset import PrecipitationDataset
        self.train_dataset = PrecipitationDataset(data_config, 'train')
        self.val_dataset = PrecipitationDataset(data_config, 'val')
        self.test_dataset = PrecipitationDataset(data_config, 'test')

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config.get('batch_size', 16),
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config.get('batch_size', 16),
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def get_test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_config.get('batch_size', 16),
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
