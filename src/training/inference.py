"""Inference Script for PhyDiff-Net.

Inference script for making precipitation predictions from ERA5 reanalysis
data. Supports single-sample and batch inference with configurable output
formats.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference.

    Returns:
        Parsed argument namespace with checkpoint, data, and output paths.
    """
    parser = argparse.ArgumentParser(
        description='Run inference with PhyDiff-Net'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint file (.pt)'
    )
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to input ERA5 data directory or file'
    )
    parser.add_argument(
        '--output_path', type=str, default='outputs/predictions',
        help='Directory to save prediction outputs'
    )
    parser.add_argument(
        '--config', type=str, default='src/configs/model_config.yaml',
        help='Model config file path'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device id to use'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for inference'
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    config: Dict,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained PhyDiff-Net model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        config: Model configuration dictionary.
        device: Target device to load the model onto.

    Returns:
        Model in evaluation mode.
    """
    from src.models.phydiff_net import PhyDiffNet

    model = PhyDiffNet(config).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path}: {e}"
        ) from e

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(
        f"Model loaded from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', 'unknown')})"
    )
    return model


def main() -> None:
    """Main inference entry point.

    Loads the trained model and ERA5 input data, runs inference, and saves
    the resulting precipitation predictions as NumPy arrays.
    """
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    # Set up device
    from src.utils.device import get_device
    device = get_device(args.gpu)
    logger.info(f"Using device: {device}")

    # Load configuration
    from src.utils.config import load_config
    config = load_config(args.config)

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load and preprocess input data
    from src.data.preprocessing import ERA5Preprocessor
    preprocessor = ERA5Preprocessor(config.get('data', {}))
    era5_data = preprocessor.load_data(args.data_path)

    logger.info(
        f"Input data loaded: shape={era5_data.shape}, "
        f"dtype={era5_data.dtype}"
    )

    # Run inference
    era5_tensor = torch.tensor(era5_data, dtype=torch.float32)
    if era5_tensor.dim() == 3:
        era5_tensor = era5_tensor.unsqueeze(0)
    era5_tensor = era5_tensor.to(device)

    with torch.no_grad():
        predictions = model.sample(era5_tensor, None)

    # Save results
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_np = predictions.cpu().numpy()
    output_file = output_path / 'predictions.npy'
    np.save(output_file, predictions_np)

    logger.info(
        f"Predictions saved to {output_file} "
        f"(shape={predictions_np.shape})"
    )


if __name__ == '__main__':
    main()
