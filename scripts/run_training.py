"""Training Launch Script for PhyDiff-Net.

Script to launch model training with proper configuration.
"""

import sys
sys.path.insert(0, 'e:/weather')

import argparse
import torch
from pathlib import Path
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device, get_gpu_info
from src.data.dataloader import PrecipitationDataLoader
from src.models.phydiff_net import PhyDiffNet
from src.training.trainer import PhyDiffTrainer

def main():
    parser = argparse.ArgumentParser(description='Train PhyDiff-Net')
    parser.add_argument('--model_config', type=str, default='src/configs/model_config.yaml')
    parser.add_argument('--data_config', type=str, default='src/configs/data_config.yaml')
    parser.add_argument('--training_config', type=str, default='src/configs/training_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    training_config = load_config(args.training_config)

    # 检查GPU
    gpu_info = get_gpu_info()
    print("=" * 60)
    print("PhyDiff-Net Training")
    print("=" * 60)
    print(f"GPU Available: {gpu_info.get('available', False)}")
    if gpu_info.get('available'):
        print(f"GPU Device: {gpu_info.get('device_name', 'Unknown')}")
        print(f"GPU Memory: {gpu_info.get('memory_allocated', 0) / 1e9:.2f} GB")
    print("=" * 60)

    # 设置设备
    device = get_device(args.gpu)

    # 创建数据加载器
    print("\n1. Creating data loader...")
    data_loader = PrecipitationDataLoader(data_config, training_config)
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    print(f"   Train samples: {len(data_loader.train_dataset)}")
    print(f"   Val samples: {len(data_loader.val_dataset)}")

    # 创建模型
    print("\n2. Creating model...")
    model = PhyDiffNet(model_config['model']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")

    # 创建训练器
    print("\n3. Creating trainer...")
    trainer = PhyDiffTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )

    # 恢复检查点
    if args.resume:
        print(f"\n4. Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 开始训练
    print("\n5. Starting training...")
    trainer.train()

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
