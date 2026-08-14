"""GMCP-only training script for PhyDiff-Net.

Trains PhyDiff-Net using only GMCP precipitation data as input and target.
This is a self-contained script that can be used as a baseline while ERA5
data download is stalled.

Usage:
    python scripts/train_gmcp_only.py
    python scripts/train_gmcp_only.py --config configs/training_gmcp_only.yaml
    python scripts/train_gmcp_only.py --verify_only
    python scripts/train_gmcp_only.py --epochs 5 --batch_size 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gmcp_sequence_dataset import create_gmcp_dataloaders
from src.models.losses.gmcp_extreme_loss import GMCPExtremeLoss
from src.models.phydiff_net import PhyDiffNet
from src.utils.config import load_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training_gmcp_only.yaml"


class GMCPLoss(nn.Module):
    """Combined MSE + MAE loss for GMCP-only precipitation forecasting."""

    def __init__(self, mse_weight: float = 1.0, mae_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        losses["mse"] = F.mse_loss(predictions, targets)
        losses["mae"] = F.l1_loss(predictions, targets)
        losses["total"] = (
            self.mse_weight * losses["mse"] + self.mae_weight * losses["mae"]
        )
        return losses


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PhyDiff-Net with GMCP-only data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to training configuration YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Run forward/backward verification and exit.",
    )
    parser.add_argument(
        "--use_extreme_loss",
        action="store_true",
        help="Use GMCPExtremeLoss (normalized MSE + physical CSI/extreme "
        "weighting) instead of the plain MSE+MAE loss.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from (restores model, "
        "optimizer, scheduler, and epoch counter).",
    )
    return parser.parse_args()


def get_device(gpu_id: int = 0) -> torch.device:
    """Get the appropriate compute device."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(gpu_id))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def verify_forward_backward(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    """Verify that a forward and backward pass work."""
    model.train()
    batch = next(iter(train_loader))
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)

    # Forward
    outputs = model(gmcp_data=inputs)
    predictions = outputs["precipitation"]
    logger.info("Forward pass output shape: %s", predictions.shape)

    # Backward
    losses = criterion(predictions, targets)
    losses["total"].backward()
    logger.info("Backward pass succeeded. Loss: %.6f", losses["total"].item())


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(gmcp_data=inputs)
        predictions = outputs["precipitation"]
        losses = criterion(predictions, targets)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        total_mse += losses["mse"].item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            logger.info(
                "Epoch %d [%d/%d] Loss: %.6f (avg: %.6f)",
                epoch,
                batch_idx + 1,
                len(train_loader),
                losses["total"].item(),
                total_loss / num_batches,
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "mse": total_mse / max(num_batches, 1),
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            outputs = model(gmcp_data=inputs)
            predictions = outputs["precipitation"]
            losses = criterion(predictions, targets)

            total_loss += losses["total"].item()
            total_mse += losses["mse"].item()
            num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "mse": total_mse / max(num_batches, 1),
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    epochs = args.epochs if args.epochs is not None else training_config.get("max_epochs", 5)
    batch_size = args.batch_size if args.batch_size is not None else training_config.get("batch_size", 1)
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else training_config.get("learning_rate", 1e-4)
    )

    seed = training_config.get("seed", 42)
    set_seed(seed)

    device = get_device(training_config.get("gpu_id", 0))

    logger.info("Creating GMCP dataloaders...")
    train_loader, val_loader, test_loader = create_gmcp_dataloaders(
        data_config,
        batch_size=batch_size,
        num_workers=training_config.get("num_workers", 0),
    )

    logger.info("Building PhyDiff-Net (GMCP-only mode)...")
    model = PhyDiffNet(model_config).to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info(
        "Model parameters: total=%d, trainable=%d", total_params, trainable_params
    )

    if args.use_extreme_loss:
        # Pull normalization stats from the training dataset so the loss can
        # denormalize predictions back to physical mm/6h for threshold-based
        # CSI and extreme-event weighting.
        train_ds = train_loader.dataset
        criterion = GMCPExtremeLoss(
            norm_min=getattr(train_ds, "input_min", None),
            norm_max=getattr(train_ds, "input_max", None),
            normalize=data_config.get("normalize", "log_minmax"),
            mse_weight=training_config.get("mse_weight", 1.0),
            mae_weight=training_config.get("mae_weight", 0.5),
            csi_weight=training_config.get("csi_weight", 0.5),
            extreme_weight=training_config.get("extreme_weight", 1.0),
        )
        logger.info(
            "Using GMCPExtremeLoss: norm_min=%.4f, norm_max=%.4f",
            criterion.norm_min.item(), criterion.norm_max.item(),
        )
    else:
        criterion = GMCPLoss(
            mse_weight=training_config.get("mse_weight", 1.0),
            mae_weight=training_config.get("mae_weight", 0.5),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=training_config.get("min_lr", 1e-6),
    )

    if args.verify_only:
        verify_forward_backward(model, train_loader, criterion, device)
        return

    output_dir = Path(training_config.get("output_dir", "outputs/gmcp_only"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    best_val_loss = float("inf")
    start_epoch = 1
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info(
            "Resumed from %s: starting at epoch %d, best_val_loss=%.6f",
            args.resume, start_epoch, best_val_loss,
        )

    for epoch in range(start_epoch, epochs + 1):
        logger.info("Epoch %d/%d", epoch, epochs)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %d summary: train_loss=%.6f, train_mse=%.6f, "
            "val_loss=%.6f, val_mse=%.6f, lr=%.2e",
            epoch,
            train_metrics["loss"],
            train_metrics["mse"],
            val_metrics["loss"],
            val_metrics["mse"],
            optimizer.param_groups[0]["lr"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info("Saved best checkpoint to %s", checkpoint_path)

    logger.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
