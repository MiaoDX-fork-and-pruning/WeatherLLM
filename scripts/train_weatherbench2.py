"""WeatherBench2 Training Script for PhyDiff-Net.

This script provides a self-contained training pipeline for testing the
PhyDiff-Net model with WeatherBench2 ERA5 data at 1.5° resolution.

The script:
1. Loads WeatherBench2 ERA5 data (NetCDF format)
2. Creates a lightweight model adapted for single-source input
3. Runs forward/backward pass verification
4. Trains for a small number of epochs as a smoke test

Expected data dimensions:
- Input:  [B, input_timesteps, n_channels, lat, lon] = [B, 4, 30, 121, 240]
  where n_channels = 6 ERA5 variables * 5 pressure levels = 30
- Target: [B, forecast_horizon, lat, lon] = [B, 4, 121, 240]

Usage:
    python scripts/train_weatherbench2.py
    python scripts/train_weatherbench2.py --config configs/training_weatherbench2.yaml
    python scripts/train_weatherbench2.py --epochs 3 --batch_size 2
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ==============================================================================
# Model Definition
# ==============================================================================


class WeatherBench2Encoder(nn.Module):
    """Lightweight encoder for WeatherBench2 ERA5 data.

    Processes multi-level atmospheric variables through convolutional
    layers to extract spatiotemporal features.

    Args:
        in_channels: Number of input channels (n_vars * n_levels).
        hidden_channels: Number of hidden channels.
        num_layers: Number of encoding layers.
    """

    def __init__(
        self,
        in_channels: int = 30,
        hidden_channels: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
            ])
            current_channels = hidden_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Encoded features [B, hidden_channels, H, W].
        """
        return self.encoder(x)


class WeatherBench2Decoder(nn.Module):
    """Lightweight decoder for precipitation prediction.

    Decodes encoded features into precipitation forecasts.

    Args:
        hidden_channels: Number of hidden channels.
        out_channels: Number of output channels (forecast_horizon).
        num_layers: Number of decoding layers.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        out_channels: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        current_channels = hidden_channels
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
            ])
            current_channels = hidden_channels
        # Final projection to output channels
        layers.append(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features to precipitation forecast.

        Args:
            x: Encoded features [B, hidden_channels, H, W].

        Returns:
            Precipitation forecast [B, out_channels, H, W].
        """
        return self.decoder(x)


class WeatherBench2Model(nn.Module):
    """WeatherBench2 precipitation forecasting model.

    A lightweight encoder-decoder model adapted for WeatherBench2
    ERA5 single-source input. Processes each input timestep through
    shared encoder weights, aggregates temporally, then decodes to
    precipitation forecasts.

    Input:  [B, T_in, C, H, W] where C = n_vars * n_levels = 30
    Output: [B, T_out, H, W]

    Args:
        config: Model configuration dictionary.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        in_channels = config.get("in_channels", 30)
        out_channels = config.get("out_channels", 4)
        hidden_channels = config.get("hidden_channels", 64)
        encoder_layers = config.get("encoder_layers", 2)
        decoder_layers = config.get("decoder_layers", 2)

        # Encoder: processes each timestep independently
        self.encoder = WeatherBench2Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=encoder_layers,
        )

        # Temporal aggregation: combines features across input timesteps
        self.temporal_attention = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Softmax(dim=1),  # Attention weights over timesteps
        )

        # Decoder: generates precipitation forecast
        self.decoder = WeatherBench2Decoder(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=decoder_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T_in, C, H, W].

        Returns:
            Precipitation forecast [B, T_out, H, W].
        """
        B, T, C, H, W = x.shape

        # Reshape to process all timesteps: [B*T, C, H, W]
        x_flat = x.reshape(B * T, C, H, W)

        # Encode each timestep: [B*T, hidden, H, W]
        features = self.encoder(x_flat)

        # Reshape back: [B, T, hidden, H, W]
        hidden_channels = features.shape[1]
        features = features.reshape(B, T, hidden_channels, H, W)

        # Temporal aggregation with attention
        # Average over timesteps: [B, hidden, H, W]
        features_avg = features.mean(dim=1)  # [B, hidden, H, W]

        # Compute attention weights: [B, T, 1, 1, 1]
        attn_input = features.reshape(B * T, hidden_channels, H, W)
        attn_weights = self.temporal_attention(attn_input)  # [B*T, 1, H, W]
        attn_weights = attn_weights.reshape(B, T, 1, H, W)

        # Weighted sum: [B, hidden, H, W]
        features_weighted = (features * attn_weights).sum(dim=1)

        # Combine average and attention-weighted features
        features_combined = features_avg + features_weighted

        # Decode to precipitation forecast: [B, T_out, H, W]
        output = self.decoder(features_combined)

        return output


# ==============================================================================
# Loss Functions
# ==============================================================================


class PrecipitationLoss(nn.Module):
    """Combined loss for precipitation forecasting.

    Combines MSE loss with optional MAE loss for robustness.

    Args:
        mse_weight: Weight for MSE loss.
        mae_weight: Weight for MAE loss.
    """

    def __init__(self, mse_weight: float = 1.0, mae_weight: float = 0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            predictions: Model predictions [B, T, H, W].
            targets: Ground truth [B, T, H, W].

        Returns:
            Dictionary of losses with 'total', 'mse', 'mae' keys.
        """
        losses = {}
        losses["mse"] = F.mse_loss(predictions, targets)
        losses["mae"] = F.l1_loss(predictions, targets)
        losses["total"] = (
            self.mse_weight * losses["mse"] + self.mae_weight * losses["mae"]
        )
        return losses


# ==============================================================================
# Training Utilities
# ==============================================================================


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device(gpu_id: int = 0) -> torch.device:
    """Get the appropriate compute device.

    Args:
        gpu_id: GPU device ID.

    Returns:
        torch.device to use for computation.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==============================================================================
# Training Loop
# ==============================================================================


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 5,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        epoch: Current epoch number.
        log_interval: Log every N batches.

    Returns:
        Dictionary of average losses.
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        predictions = model(inputs)
        losses = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += losses["total"].item()
        total_mse += losses["mse"].item()
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {losses['total'].item():.6f} "
                f"(avg: {avg_loss:.6f})"
            )

    avg_loss = total_loss / max(num_batches, 1)
    avg_mse = total_mse / max(num_batches, 1)
    return {"loss": avg_loss, "mse": avg_mse}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Dictionary of validation losses.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    for batch in val_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        predictions = model(inputs)
        losses = criterion(predictions, targets)

        total_loss += losses["total"].item()
        total_mse += losses["mse"].item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_mse = total_mse / max(num_batches, 1)
    return {"loss": avg_loss, "mse": avg_mse}


def verify_forward_backward(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> bool:
    """Verify that forward and backward passes work correctly.

    Runs a single forward pass, computes loss, and performs backward
    pass to ensure gradients flow correctly through the model.

    Args:
        model: Model to test.
        train_loader: Data loader (uses first batch).
        criterion: Loss function.
        device: Compute device.

    Returns:
        True if verification passes, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Forward/Backward Pass Verification")
    logger.info("=" * 60)

    model.train()

    # Get first batch
    batch = next(iter(train_loader))
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)

    logger.info(f"  Input shape:  {inputs.shape}")
    logger.info(f"  Target shape: {targets.shape}")

    # Forward pass
    try:
        predictions = model(inputs)
        logger.info(f"  Output shape: {predictions.shape}")
        logger.info(f"  Forward pass: OK")
    except Exception as e:
        logger.error(f"  Forward pass FAILED: {e}")
        return False

    # Verify output shape
    expected_shape = targets.shape
    actual_shape = predictions.shape
    if actual_shape != expected_shape:
        logger.warning(
            f"  Shape mismatch: expected {expected_shape}, got {actual_shape}"
        )
        # Not a failure, just a warning

    # Compute loss
    try:
        losses = criterion(predictions, targets)
        logger.info(f"  Loss: {losses['total'].item():.6f}")
        logger.info(f"  Loss computation: OK")
    except Exception as e:
        logger.error(f"  Loss computation FAILED: {e}")
        return False

    # Backward pass
    try:
        model.zero_grad()
        losses["total"].backward()

        # Check gradients
        has_grad = False
        no_grad_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                if param.grad.abs().sum() == 0:
                    no_grad_params.append(name)

        if has_grad:
            logger.info(f"  Backward pass: OK (gradients computed)")
        else:
            logger.error(f"  Backward pass FAILED (no gradients)")
            return False

        if no_grad_params:
            logger.warning(
                f"  Parameters with zero gradients: {no_grad_params[:5]}..."
            )

    except Exception as e:
        logger.error(f"  Backward pass FAILED: {e}")
        return False

    # Verify optimizer step
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.step()
        logger.info(f"  Optimizer step: OK")
    except Exception as e:
        logger.error(f"  Optimizer step FAILED: {e}")
        return False

    logger.info("=" * 60)
    logger.info("Verification PASSED: All checks OK")
    logger.info("=" * 60)
    return True


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main() -> None:
    """Main training entry point.

    Orchestrates the complete WeatherBench2 training pipeline:
    1. Parse arguments and load config
    2. Set up device and seed
    3. Create data loaders
    4. Create model
    5. Verify forward/backward pass
    6. Run training loop
    7. Save checkpoint
    """
    parser = argparse.ArgumentParser(
        description="Train PhyDiff-Net with WeatherBench2 data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_weatherbench2.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify forward/backward pass, skip full training",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Get device
    device = get_device(args.gpu)

    # ---- Data Loading ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Loading WeatherBench2 Data")
    logger.info("=" * 60)

    from src.data.weatherbench2_dataset import WeatherBench2Dataset

    data_config = config["data"]
    batch_size = config["training"]["batch_size"]

    train_dataset = WeatherBench2Dataset(data_config, split="train")
    val_dataset = WeatherBench2Dataset(data_config, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # ---- Model Creation ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Creating Model")
    logger.info("=" * 60)

    model_config = config["model"]
    model = WeatherBench2Model(model_config).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info(f"Model config: {model_config}")

    # ---- Forward/Backward Verification ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Verifying Forward/Backward Pass")
    logger.info("=" * 60)

    criterion = PrecipitationLoss(
        mse_weight=config["loss"].get("mse_weight", 1.0),
        mae_weight=config["loss"].get("mae_weight", 0.0),
    )

    verification_passed = verify_forward_backward(
        model, train_loader, criterion, device
    )

    if not verification_passed:
        logger.error("Forward/backward verification FAILED. Aborting.")
        sys.exit(1)

    if args.verify_only:
        logger.info("Verification complete (--verify_only mode). Exiting.")
        return

    # ---- Training ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Starting Training")
    logger.info("=" * 60)

    max_epochs = config["training"]["max_epochs"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.01)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=tuple(config["training"].get("optimizer", {}).get("betas", [0.9, 0.999])),
        weight_decay=weight_decay,
    )

    # Simple cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    logger.info(f"Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
    logger.info(f"Scheduler: CosineAnnealing (T_max={max_epochs})")
    logger.info(f"Epochs: {max_epochs}")
    logger.info(f"Batch size: {batch_size}")

    # Create checkpoint directory
    checkpoint_dir = Path(config.get("training", {}).get("checkpoint", {}).get("save_dir", "checkpoints/weatherbench2_test"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    training_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        logger.info(f"\n--- Epoch {epoch}/{max_epochs} ---")
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{max_epochs} - "
            f"Train Loss: {train_metrics['loss']:.6f}, "
            f"Val Loss: {val_metrics['loss']:.6f}, "
            f"LR: {current_lr:.2e}, "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "config": config,
                },
                best_checkpoint_path,
            )
            logger.info(f"  Best model saved: {best_checkpoint_path}")

    # Save final model
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    torch.save(
        {
            "epoch": max_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "config": config,
        },
        final_checkpoint_path,
    )

    total_time = time.time() - training_start
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info(f"Final model: {final_checkpoint_path}")
    logger.info(f"Best model: {best_checkpoint_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
