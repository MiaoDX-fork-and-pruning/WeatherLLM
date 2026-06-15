"""Checkpoint Management for PhyDiff-Net.

Utilities for saving and loading model checkpoints.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List
import shutil


class CheckpointManager:
    """模型检查点管理"""

    def __init__(self, checkpoint_dir: str = 'models/checkpoints', max_checkpoints: int = 5):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大检查点数量
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict, filename: Optional[str] = None) -> str:
        """
        保存检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            metrics: 评估指标
            filename: 文件名（可选）

        Returns:
            保存路径
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch:04d}.pt'

        save_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        torch.save(checkpoint, save_path)

        # 清理旧检查点
        self.cleanup_old_checkpoints()

        return str(save_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径

        Returns:
            检查点字典
        """
        return torch.load(checkpoint_path, map_location='cpu')

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新检查点

        Returns:
            最新检查点路径，如果没有则返回None
        """
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            return str(checkpoints[-1])
        return None

    def cleanup_old_checkpoints(self):
        """清理旧检查点"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()
