"""Device Management for PhyDiff-Net.

Utilities for managing computing devices.
"""

import torch
import torch.distributed as dist
from typing import Optional


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    获取计算设备

    Args:
        device_id: GPU设备ID

    Returns:
        计算设备
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        return torch.device('cuda')
    return torch.device('cpu')


def setup_distributed(rank: int, world_size: int):
    """
    设置分布式训练

    Args:
        rank: 进程ID
        world_size: 进程总数
    """
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式训练"""
    dist.destroy_process_group()


def get_gpu_info() -> dict:
    """
    获取GPU信息

    Returns:
        GPU信息字典
    """
    if not torch.cuda.is_available():
        return {'available': False}

    return {
        'available': True,
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(0),
        'memory_allocated': torch.cuda.memory_allocated(0),
        'memory_reserved': torch.cuda.memory_reserved(0)
    }
