"""Dual-source (ERA5 + GMCP) training script for PhyDiff-Net.

Trains the full PhyDiff-Net architecture with ERA5 atmospheric states as
the second input source, complementing GMCP precipitation history. This
targets the extreme-event weakness of the GMCP-only baseline: atmospheric
circulation (moisture transport, vertical motion precursors) is the key
physical information for predicting heavy rainfall.

Usage:
    python scripts/train_gmcp_era5.py --config configs/training_gmcp_era5.yaml
    python scripts/train_gmcp_era5.py --config ... --verify_only
    python scripts/train_gmcp_era5.py --config ... --resume outputs/.../best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gmcp_era5_dataset import create_gmcp_era5_dataloaders  # noqa: E402
from src.models.losses.gmcp_extreme_loss import GMCPExtremeLoss  # noqa: E402
from src.models.phydiff_net import PhyDiffNet  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training_gmcp_era5.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PhyDiff-Net with ERA5 + GMCP dual sources."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to training configuration YAML.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Run one forward/backward pass and exit.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Checkpoint to resume training from.",
    )
    return parser.parse_args()


def get_device(gpu_id: int) -> torch.device:
    """Return the compute device."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(gpu_id))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device,
    epoch: int,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    n = 0
    for batch_idx, batch in enumerate(loader):
        era5 = batch["era5"].to(device)
        gmcp = batch["input"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(era5_data=era5, gmcp_data=gmcp)
        pred = outputs["precipitation"]
        losses = criterion(pred, target)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        total_mse += losses["mse"].item()
        n += 1
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            logger.info(
                "Epoch %d [%d/%d] Loss: %.4f (avg: %.4f)",
                epoch, batch_idx + 1, len(loader),
                losses["total"].item(), total_loss / n,
            )
    return {"loss": total_loss / max(n, 1), "mse": total_mse / max(n, 1)}


@torch.no_grad()
def validate(model, loader, criterion, device) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n = 0
    for batch in loader:
        era5 = batch["era5"].to(device)
        gmcp = batch["input"].to(device)
        target = batch["target"].to(device)
        outputs = model(era5_data=era5, gmcp_data=gmcp)
        losses = criterion(outputs["precipitation"], target)
        total_loss += losses["total"].item()
        total_mse += losses["mse"].item()
        n += 1
    return {"loss": total_loss / max(n, 1), "mse": total_mse / max(n, 1)}


def main() -> None:
    """Run dual-source training."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    epochs = args.epochs or training_config.get("max_epochs", 5)
    batch_size = args.batch_size or training_config.get("batch_size", 1)
    lr = args.learning_rate or training_config.get("learning_rate", 1e-4)

    set_seed(training_config.get("seed", 42))
    device = get_device(training_config.get("gpu_id", 0))

    logger.info("Creating dual-source dataloaders...")
    train_loader, val_loader, _ = create_gmcp_era5_dataloaders(
        data_config, batch_size=batch_size,
        num_workers=training_config.get("num_workers", 0),
    )

    logger.info("Building PhyDiff-Net (dual-source mode)...")
    model = PhyDiffNet(model_config).to(device)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", total)

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
        eta_min=training_config.get("min_lr", 1e-6),
    )

    if args.verify_only:
        model.train()
        batch = next(iter(train_loader))
        era5 = batch["era5"].to(device)
        gmcp = batch["input"].to(device)
        target = batch["target"].to(device)
        outputs = model(era5_data=era5, gmcp_data=gmcp)
        logger.info("Forward output shape: %s", outputs["precipitation"].shape)
        losses = criterion(outputs["precipitation"], target)
        losses["total"].backward()
        logger.info("Backward OK. Loss: %.4f", losses["total"].item())
        return

    output_dir = Path(training_config.get("output_dir", "outputs/gmcp_era5"))
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
            "Resumed from %s: epoch %d, best_val_loss=%.4f",
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
            "Epoch %d summary: train_loss=%.4f, train_mse=%.6f, "
            "val_loss=%.4f, val_mse=%.6f, lr=%.2e",
            epoch, train_metrics["loss"], train_metrics["mse"],
            val_metrics["loss"], val_metrics["mse"],
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

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
