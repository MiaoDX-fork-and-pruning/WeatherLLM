"""Model Test Script for PhyDiff-Net.

Test script to verify the model can be instantiated and run forward pass.
"""

import sys
sys.path.insert(0, 'e:/weather')

import torch
from src.models.phydiff_net import PhyDiffNet
from src.utils.config import load_config

def test_model_instantiation():
    """测试模型实例化"""
    print("Testing model instantiation...")

    config = load_config('src/configs/model_config.yaml')
    model = PhyDiffNet(config["model"])

    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def test_forward_pass(model):
    """测试前向传播"""
    print("\nTesting forward pass...")

    batch_size = 2
    era5_channels = 10
    height, width = 64, 64

    # 创建模拟输入
    era5_data = torch.randn(batch_size, era5_channels, height, width)
    gmcp_data = torch.randn(batch_size, 1, height, width)  # GMCP at same spatial resolution

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(era5_data, gmcp_data)

    print(f"Forward pass successful!")
    print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else 'Dict'}")
    return output

def test_loss_computation():
    """测试损失计算"""
    print("\nTesting loss computation...")

    from src.models.losses.precipitation_loss import PrecipitationLoss

    config = load_config('src/configs/model_config.yaml')
    loss_fn = PrecipitationLoss(config['loss'])

    # 创建模拟预测和目标
    predictions = torch.randn(2, 1, 64, 64)
    targets = torch.randn(2, 1, 64, 64)

    # 计算损失
    losses = loss_fn(predictions, targets)

    print(f"Loss computation successful!")
    print(f"Total loss: {losses['total'].item():.4f}")
    return losses

def main():
    """主测试函数"""
    print("=" * 60)
    print("PhyDiff-Net Model Test")
    print("=" * 60)

    try:
        # 测试1: 模型实例化
        model = test_model_instantiation()

        # 测试2: 前向传播
        output = test_forward_pass(model)

        # 测试3: 损失计算
        losses = test_loss_computation()

        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
