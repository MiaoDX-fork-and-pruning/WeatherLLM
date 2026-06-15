"""
PhyDiff-Net Trainer Module

This module implements the PhyDiffTrainer class for multi-stage progressive
training of the PhyDiff-Net precipitation forecasting model.

The trainer supports:
- Multi-stage training (pretrain, fusion_pretrain, finetune, extreme_enhance)
- Mixed precision training with automatic loss scaling
- Distributed training with DDP
- Checkpoint saving and restoration
- Comprehensive logging with TensorBoard/WandB
- Early stopping and learning rate scheduling
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Tracks the current state of training."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    best_epoch: int = 0
    stages_completed: list = field(default_factory=list)
    current_stage: str = ""
    training_loss_history: list = field(default_factory=list)
    validation_loss_history: list = field(default_factory=list)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate after annealing
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self) -> None:
        """Update learning rate for the current epoch."""
        self.current_epoch += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self) -> float:
        """Compute learning rate based on current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + torch.cos(torch.tensor(progress * 3.14159)).item()
            )

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._compute_lr()


class PhyDiffTrainer:
    """Main trainer class for PhyDiff-Net model.

    This class handles the complete training pipeline including:
    - Multi-stage progressive training
    - Mixed precision training
    - Distributed training support
    - Checkpoint management
    - Logging and monitoring

    Args:
        model: PhyDiff-Net model to train
        config: Training configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        rank: Process rank for distributed training
        world_size: Number of processes for distributed training
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Initialize model
        self.model = model.to(self.device)

        # Wrap model with DDP if distributed
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=config.get("distributed", {}).get(
                    "find_unused_parameters", False
                ),
            )

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training state
        self.state = TrainingState()

        # Initialize components
        self._init_optimizer()
        self._init_scheduler()
        self._init_mixed_precision()
        self._init_loss_function()
        self._init_logging()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PhyDiffTrainer initialized on device: {self.device}, "
            f"rank: {rank}/{world_size}"
        )

    def _init_optimizer(self) -> None:
        """Initialize optimizer with parameter groups."""
        opt_config = self.config.get("training", {}).get("optimizer", {})

        # Create parameter groups with different learning rates
        param_groups = [
            {"params": self.model.parameters(), "lr": self.config["training"]["learning_rate"]}
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config["training"]["learning_rate"],
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            eps=opt_config.get("eps", 1e-8),
            weight_decay=self.config["training"].get("weight_decay", 0.01),
        )

    def _init_scheduler(self) -> None:
        """Initialize learning rate scheduler."""
        sched_config = self.config.get("training", {}).get("scheduler", {})
        training_config = self.config.get("training", {})

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=training_config.get("warmup_epochs", 10),
            total_epochs=training_config.get("max_epochs", 400),
            base_lr=training_config.get("learning_rate", 1e-4),
            min_lr=sched_config.get("min_lr", 1e-6),
        )

    def _init_mixed_precision(self) -> None:
        """Initialize mixed precision training."""
        mp_config = self.config.get("training", {}).get("mixed_precision", {})
        self.use_amp = mp_config.get("enabled", True)
        self.scaler = GradScaler(enabled=self.use_amp)

    def _init_loss_function(self) -> None:
        """Initialize loss function."""
        # Import loss function from model or use default
        try:
            from src.models.losses import PrecipitationLoss

            self.criterion = PrecipitationLoss(self.config)
        except ImportError:
            logger.warning("PrecipitationLoss not found, using combined loss")
            self.criterion = self._create_default_loss()

    def _create_default_loss(self) -> nn.Module:
        """Create default combined loss function."""
        loss_config = self.config.get("loss", {})

        class CombinedLoss(nn.Module):
            """Combined loss function for precipitation forecasting."""

            def __init__(self, config):
                super().__init__()
                self.mse_loss = nn.MSELoss()
                self.huber_loss = nn.HuberLoss(delta=10.0)
                self.weights = {
                    "mse": config.get("mse_weight", 0.2),
                    "huber": config.get("huber_weight", 0.1),
                }

            def forward(self, predictions, targets, metadata=None):
                """Compute combined loss."""
                losses = {}
                losses["mse"] = self.mse_loss(predictions, targets)
                losses["huber"] = self.huber_loss(predictions, targets)

                total_loss = (
                    self.weights["mse"] * losses["mse"]
                    + self.weights["huber"] * losses["huber"]
                )
                losses["total"] = total_loss
                return losses

        return CombinedLoss(loss_config)

    def _init_logging(self) -> None:
        """Initialize logging components."""
        log_config = self.config.get("training", {}).get("logging", {})
        self.log_dir = Path(log_config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_every_n_steps = log_config.get("log_every_n_steps", 50)

        # TensorBoard writer
        if log_config.get("tensorboard", True) and self.is_main_process:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        else:
            self.writer = None

    def set_stage(self, stage_name: str) -> None:
        """Set current training stage and update optimizer.

        Args:
            stage_name: Name of the training stage
                ('pretrain', 'fusion_pretrain', 'finetune', 'extreme_enhance')
        """
        stages = self.config.get("training", {}).get("stages", {})
        if stage_name not in stages:
            raise ValueError(
                f"Unknown stage: {stage_name}. Available: {list(stages.keys())}"
            )

        stage_config = stages[stage_name]
        self.state.current_stage = stage_name

        # Update learning rate for this stage
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = stage_config["lr"]

        logger.info(
            f"Stage set to: {stage_name}, LR: {stage_config['lr']}, "
            f"Epochs: {stage_config['epochs']}"
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_losses = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                predictions = self.model(batch["input"])
                losses = self.criterion(predictions, batch["target"], batch.get("metadata"))

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()

            # Gradient clipping
            if self.config["training"].get("gradient_clip", 0) > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip"],
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item()
            num_batches += 1

            # Log progress
            if self.is_main_process and batch_idx % self.log_every_n_steps == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self._log_metrics(losses, global_step, prefix="train/")

        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        self.state.training_loss_history.append(avg_losses.get("total", 0.0))

        return avg_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        val_losses = {}
        num_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)

            with autocast(enabled=self.use_amp):
                predictions = self.model(batch["input"])
                losses = self.criterion(predictions, batch["target"], batch.get("metadata"))

            for key, value in losses.items():
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += value.item()
            num_batches += 1

        avg_losses = {key: value / num_batches for key, value in val_losses.items()}
        self.state.validation_loss_history.append(avg_losses.get("total", 0.0))

        if self.is_main_process:
            self._log_metrics(avg_losses, epoch, prefix="val/")

        return avg_losses

    def train(
        self,
        start_stage: str = "pretrain",
        resume_checkpoint: Optional[str] = None,
    ) -> TrainingState:
        """Run the complete multi-stage training pipeline.

        Args:
            start_stage: Stage to start training from
            resume_checkpoint: Path to checkpoint to resume from

        Returns:
            Final training state
        """
        # Resume from checkpoint if provided
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        stages = self.config.get("training", {}).get("stages", {})
        stage_order = ["pretrain", "fusion_pretrain", "finetune", "extreme_enhance"]

        # Find starting stage index
        start_idx = stage_order.index(start_stage) if start_stage in stage_order else 0

        for stage_name in stage_order[start_idx:]:
            stage_config = stages[stage_name]
            self.set_stage(stage_name)

            logger.info(f"\n{'='*60}")
            logger.info(f"Starting stage: {stage_name}")
            logger.info(f"Description: {stage_config.get('description', '')}")
            logger.info(f"Epochs: {stage_config['epochs']}, LR: {stage_config['lr']}")
            logger.info(f"{'='*60}\n")

            for epoch in range(stage_config["epochs"]):
                self.state.epoch += 1

                # Train
                train_losses = self.train_epoch(self.state.epoch)
                logger.info(
                    f"Epoch {self.state.epoch}/{stage_config['epochs']} "
                    f"[{stage_name}] - Train Loss: {train_losses.get('total', 0):.4f}"
                )

                # Validate
                if self.state.epoch % self.config.get("validation", {}).get(
                    "validate_every_n_epochs", 5
                ) == 0:
                    val_losses = self.validate(self.state.epoch)
                    logger.info(
                        f"Epoch {self.state.epoch} - Val Loss: "
                        f"{val_losses.get('total', 0):.4f}"
                    )

                    # Early stopping check
                    if self._check_early_stopping(val_losses):
                        logger.info("Early stopping triggered")
                        break

                # Save checkpoint
                if self.state.epoch % self.config.get("checkpoint", {}).get(
                    "save_every_n_epochs", 10
                ) == 0:
                    self.save_checkpoint(
                        self.checkpoint_dir / f"checkpoint_epoch_{self.state.epoch}.pt"
                    )

                # Update scheduler
                self.scheduler.step()

            self.state.stages_completed.append(stage_name)
            logger.info(f"Completed stage: {stage_name}")

        # Save final checkpoint
        self.save_checkpoint(self.checkpoint_dir / "final_model.pt")

        if self.writer:
            self.writer.close()

        return self.state

    def _to_device(self, data: Any) -> Any:
        """Move data to the appropriate device.

        Args:
            data: Input data (tensor, dict, or list)

        Returns:
            Data moved to device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {key: self._to_device(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._to_device(item) for item in data]
        return data

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        """Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            prefix: Prefix for metric names
        """
        if self.writer is None:
            return

        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}{key}", value, step)

        # Log learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("learning_rate", current_lr, step)

    def _check_early_stopping(self, val_losses: Dict[str, float]) -> bool:
        """Check if early stopping criteria is met.

        Args:
            val_losses: Validation loss dictionary

        Returns:
            True if training should stop
        """
        val_config = self.config.get("validation", {})
        patience = val_config.get("early_stopping_patience", 30)
        monitor_metric = val_config.get("monitor_metric", "val_extreme_csi")

        current_metric = val_losses.get(monitor_metric.replace("val_", ""), 0.0)

        if current_metric > self.state.best_metric:
            self.state.best_metric = current_metric
            self.state.best_epoch = self.state.epoch
            return False

        return (self.state.epoch - self.state.best_epoch) >= patience

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        if not self.is_main_process:
            return

        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model_state_dict": (
                self.model.module.state_dict()
                if isinstance(self.model, DDP)
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": {
                "current_epoch": self.scheduler.current_epoch,
            },
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "stages_completed": self.state.stages_completed,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scaler state
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore training state
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.best_metric = checkpoint.get("best_metric", float("-inf"))
        self.state.best_epoch = checkpoint.get("best_epoch", 0)
        self.state.stages_completed = checkpoint.get("stages_completed", [])

        # Restore scheduler state
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.current_epoch = checkpoint["scheduler_state_dict"].get(
                "current_epoch", 0
            )

        logger.info(f"Checkpoint loaded: {path} (epoch {self.state.epoch})")

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints beyond the keep limit."""
        keep_last_n = self.config.get("checkpoint", {}).get("keep_last_n_checkpoints", 5)

        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > keep_last_n:
            for old_checkpoint in checkpoints[:-keep_last_n]:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
