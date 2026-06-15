"""Training Script for PhyDiff-Net.

Main training script for the precipitation forecasting model.
Supports single-GPU and distributed training with checkpoint resumption.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns:
        Parsed argument namespace containing config paths and training options.
    """
    parser = argparse.ArgumentParser(
        description='Train PhyDiff-Net precipitation forecasting model'
    )
    parser.add_argument(
        '--config', type=str, default='src/configs/training_config.yaml',
        help='Training config file path'
    )
    parser.add_argument(
        '--data_config', type=str, default='src/configs/data_config.yaml',
        help='Data config file path'
    )
    parser.add_argument(
        '--model_config', type=str, default='src/configs/model_config.yaml',
        help='Model config file path'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume training from checkpoint path'
    )
    parser.add_argument(
        '--distributed', action='store_true',
        help='Use distributed training with DDP'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device id to use (single-GPU mode)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize distributed training with NCCL backend.

    Sets the process group and pins the current CUDA device to the
    assigned rank.

    Args:
        rank: Unique process identifier for this rank.
        world_size: Total number of processes participating in training.
    """
    import torch.distributed as dist
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    logger.info(
        f"Distributed training initialized: rank {rank}/{world_size}"
    )


def main() -> None:
    """Main training entry point.

    Orchestrates the full training pipeline:
    1. Parse CLI arguments
    2. Set random seed for reproducibility
    3. Load training, data, and model configurations
    4. Set up compute device
    5. Create data loaders
    6. Instantiate the PhyDiff-Net model
    7. Launch training via PhyDiffTrainer
    8. Clean up distributed process group if applicable
    """
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    # Set random seed for reproducibility
    from src.utils.seed import set_seed
    set_seed(args.seed)

    # Load configurations
    from src.utils.config import load_config
    train_config = load_config(args.config)
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)

    logger.info("Configurations loaded successfully")

    # Set up device
    from src.utils.device import get_device
    device = get_device(args.gpu)
    logger.info(f"Using device: {device}")

    # Create data loaders
    from src.data.dataloader import PrecipitationDataLoader
    data_loader = PrecipitationDataLoader(
        data_config, train_config, args.distributed
    )

    # Create model
    from src.models.phydiff_net import PhyDiffNet
    model = PhyDiffNet(model_config).to(device)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f"Model parameters: {total_params:,} total, "
        f"{trainable_params:,} trainable"
    )

    # Create trainer
    from src.training.trainer import PhyDiffTrainer
    trainer = PhyDiffTrainer(
        model=model,
        config=train_config,
        train_loader=data_loader.get_train_loader(),
        val_loader=data_loader.get_val_loader(),
    )

    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from checkpoint: {args.resume}")

    # Run training
    trainer.train()

    # Clean up distributed training resources
    if args.distributed:
        import torch.distributed as dist
        dist.destroy_process_group()
        logger.info("Distributed training resources cleaned up")


if __name__ == '__main__':
    main()
