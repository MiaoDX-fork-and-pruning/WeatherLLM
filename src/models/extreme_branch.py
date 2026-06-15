"""Extreme Event Aware Branch for PhyDiff-Net.

This module handles extreme precipitation events with specialized
encoding and prediction heads. It includes:
- Multi-level extreme event detection
- Specialized encoder for extreme patterns
- Intensity and extent prediction heads

Author: weather-model-trainer
Date: 2026-06-15
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtremeDetector(nn.Module):
    """Multi-level extreme event detector.

    Detects extreme precipitation events at different intensity levels
    using learnable thresholds and adaptive detection mechanisms.

    Args:
        in_channels: Number of input channels.
        hidden_dim: Hidden dimension for detection networks.
        thresholds: Dictionary mapping level names to intensity thresholds
                   (mm/6h). Default: {'heavy': 25.0, 'very_heavy': 50.0,
                   'extreme': 100.0}.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if thresholds is None:
            thresholds = {"heavy": 25.0, "very_heavy": 50.0, "extreme": 100.0}

        self.thresholds = thresholds
        self.level_names = list(thresholds.keys())

        # Detection network
        self.detection_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, len(self.level_names), kernel_size=1),
        )

        # Learnable threshold adjustments
        self.threshold_adjustments = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0))
            for name in self.level_names
        })

    def forward(
        self, precipitation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Detect extreme events at multiple levels.

        Args:
            precipitation: Precipitation field [B, 1, H, W].

        Returns:
            Dictionary mapping level names to boolean masks [B, 1, H, W].
        """
        # Generate detection logits
        logits = self.detection_net(precipitation)

        # Apply learned thresholds
        masks = {}
        for i, (name, threshold) in enumerate(self.thresholds.items()):
            adjusted_threshold = threshold + self.threshold_adjustments[name]
            masks[name] = logits[:, i:i+1, :, :] > adjusted_threshold

        return masks

    def get_combined_mask(
        self, masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get combined extreme event mask (any level).

        Args:
            masks: Dictionary of level masks from forward().

        Returns:
            Combined boolean mask [B, 1, H, W].
        """
        combined = None
        for mask in masks.values():
            if combined is None:
                combined = mask
            else:
                combined = combined | mask
        return combined


class ExtremeEncoder(nn.Module):
    """Specialized encoder for extreme precipitation patterns.

    This encoder focuses on capturing the unique characteristics of
    extreme precipitation events, which differ from normal precipitation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_layers: Number of encoding layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels

        # Multi-scale feature extraction
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])

        # Layer-wise encoding with residual connections
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "conv1": nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                "norm1": nn.GroupNorm(8, out_channels),
                "conv2": nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                "norm2": nn.GroupNorm(8, out_channels),
                "attention": CrossScaleAttention(out_channels, num_heads=4),
                "dropout": nn.Dropout(dropout),
            }))

        # Final projection
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        extreme_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode extreme precipitation features.

        Args:
            features: Input features [B, C, H, W].
            extreme_mask: Optional mask for extreme regions [B, 1, H, W].

        Returns:
            Encoded extreme features [B, out_channels, H, W].
        """
        # Apply mask if provided
        if extreme_mask is not None:
            features = features * extreme_mask

        # Multi-scale feature extraction
        multi_scale = [conv(features) for conv in self.scale_convs]
        h = sum(multi_scale) / len(multi_scale)

        # Layer-wise encoding
        for layer in self.layers:
            residual = h
            h = layer["norm1"](h)
            h = F.silu(h)
            h = layer["conv1"](h)
            h = layer["norm2"](h)
            h = F.silu(h)
            h = layer["dropout"](h)
            h = layer["conv2"](h)
            h = layer["attention"](h)
            h = h + residual

        return self.final_conv(h)


class CrossScaleAttention(nn.Module):
    """Cross-scale attention for capturing multi-scale patterns.

    Args:
        channels: Number of channels.
        num_heads: Number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W].
        """
        B, C, H, W = x.shape
        h = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        h_norm = self.norm(h)
        h_attn, _ = self.attention(h_norm, h_norm, h_norm)
        h = h + h_attn
        return h.transpose(1, 2).view(B, C, H, W)


class IntensityHead(nn.Module):
    """Prediction head for extreme precipitation intensity.

    This head predicts the intensity of extreme precipitation events,
    using specialized loss functions for heavy rain events.

    Args:
        in_channels: Number of input channels.
        hidden_dim: Hidden dimension.
        out_channels: Number of output channels (default 1 for intensity).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        out_channels: int = 1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Intensity prediction with non-negative output
        self.intensity_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1),
            nn.ReLU(),  # Ensure non-negative intensity
        )

        # Uncertainty estimation
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1),
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict extreme precipitation intensity.

        Args:
            features: Encoded features [B, C, H, W].

        Returns:
            Tuple of (intensity prediction, uncertainty).
        """
        encoded = self.encoder(features)
        intensity = self.intensity_conv(encoded)
        uncertainty = self.uncertainty_conv(encoded)
        return intensity, uncertainty


class ExtentHead(nn.Module):
    """Prediction head for extreme precipitation spatial extent.

    This head predicts the spatial extent (area coverage) of extreme
    precipitation events.

    Args:
        in_channels: Number of input channels.
        hidden_dim: Hidden dimension.
        num_classes: Number of extent classes (e.g., point, small, medium, large).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_classes: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Extent classification
        self.extent_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1),
        )

        # Boundary refinement
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1, groups=num_classes),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict extreme precipitation extent.

        Args:
            features: Encoded features [B, C, H, W].

        Returns:
            Extent logits [B, num_classes, H, W].
        """
        encoded = self.encoder(features)
        extent_logits = self.extent_conv(encoded)

        # Apply boundary refinement
        extent_probs = F.softmax(extent_logits, dim=1)
        extent_probs = self.boundary_conv(extent_probs)

        return extent_probs


class ExtremeEventLoss(nn.Module):
    """Loss function for extreme precipitation events.

    This loss combines multiple objectives to optimize extreme event
    prediction: MSE, CSI, and quantile loss.

    Args:
        weights: Dictionary mapping level names to loss weights.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if weights is None:
            weights = {"heavy": 1.0, "very_heavy": 2.0, "extreme": 5.0}

        self.weights = weights
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        extreme_masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute extreme event loss.

        Args:
            predictions: Predicted precipitation [B, 1, H, W].
            targets: Target precipitation [B, 1, H, W].
            extreme_masks: Dictionary of extreme event masks per level.

        Returns:
            Tuple of (total loss, per-level losses dictionary).
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        per_level_losses = {}

        for level, weight in self.weights.items():
            if level not in extreme_masks:
                continue

            mask = extreme_masks[level]

            if mask.sum() == 0:
                per_level_losses[level] = 0.0
                continue

            # Apply mask
            pred_masked = predictions[mask.expand_as(predictions)]
            target_masked = targets[mask.expand_as(targets)]

            # MSE loss
            mse = F.mse_loss(pred_masked, target_masked)

            # CSI loss (1 - CSI)
            csi_loss = 1.0 - self._compute_csi(predictions, targets, mask)

            # Quantile loss for heavy events
            quantile_loss = self._quantile_loss(pred_masked, target_masked, quantile=0.95)

            # Combined loss
            level_loss = mse + 0.5 * csi_loss + 0.3 * quantile_loss
            total_loss = total_loss + weight * level_loss

            per_level_losses[level] = level_loss.item()

        return total_loss, per_level_losses

    def _compute_csi(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Compute Critical Success Index.

        Args:
            pred: Predicted field.
            target: Target field.
            mask: Extreme event mask.
            threshold: Threshold for binary classification.

        Returns:
            CSI value.
        """
        pred_binary = ((pred > threshold) & mask).float()
        target_binary = ((target > threshold) & mask).float()

        hits = (pred_binary * target_binary).sum()
        false_alarms = (pred_binary * (1 - target_binary)).sum()
        misses = ((1 - pred_binary) * target_binary).sum()

        csi = hits / (hits + false_alarms + misses + 1e-8)
        return csi

    def _quantile_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        quantile: float = 0.95,
    ) -> torch.Tensor:
        """Compute quantile loss.

        Args:
            pred: Predicted values.
            target: Target values.
            quantile: Quantile to optimize.

        Returns:
            Quantile loss value.
        """
        errors = target - pred
        loss = torch.max(quantile * errors, (quantile - 1) * errors)
        return loss.mean()


class ExtremeEventBranch(nn.Module):
    """Extreme event aware branch for precipitation forecasting.

    This branch specializes in handling extreme precipitation events
    through dedicated detection, encoding, and prediction mechanisms.

    Args:
        in_channels: Number of input channels.
        hidden_dim: Hidden dimension.
        dropout: Dropout rate.

    Example:
        >>> branch = ExtremeEventBranch(in_channels=256, hidden_dim=256)
        >>> features = torch.randn(2, 256, 128, 128)
        >>> intensity, extent, masks = branch(features)
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Extreme event detector
        self.extreme_detector = ExtremeDetector(
            in_channels=1,
            hidden_dim=64,
            thresholds={"heavy": 25.0, "very_heavy": 50.0, "extreme": 100.0},
        )

        # Extreme event encoder
        self.extreme_encoder = ExtremeEncoder(
            in_channels=in_channels,
            out_channels=hidden_dim,
            num_layers=4,
            dropout=dropout,
        )

        # Intensity prediction head
        self.intensity_head = IntensityHead(
            in_channels=hidden_dim,
            hidden_dim=128,
            out_channels=1,
        )

        # Extent prediction head
        self.extent_head = ExtentHead(
            in_channels=hidden_dim,
            hidden_dim=128,
            num_classes=4,
        )

        # Feature fusion
        self.feature_fusion = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

        # Loss function
        self.loss_fn = ExtremeEventLoss()

    def forward(
        self,
        features: torch.Tensor,
        precipitation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for extreme event processing.

        Args:
            features: Input features [B, C, H, W].
            precipitation: Optional precipitation field for detection [B, 1, H, W].
                          If None, uses the mean of features.

        Returns:
            Tuple of:
            - intensity: Predicted intensity [B, 1, H, W]
            - extent: Predicted extent [B, num_classes, H, W]
            - extreme_masks: Dictionary of extreme event masks
        """
        # Get precipitation for detection
        if precipitation is None:
            precipitation = features.mean(dim=1, keepdim=True)

        # Detect extreme events
        extreme_masks = self.extreme_detector(precipitation)
        combined_mask = self.extreme_detector.get_combined_mask(extreme_masks)

        # Encode extreme features
        extreme_features = self.extreme_encoder(features, combined_mask)

        # Fuse with original features
        fused = torch.cat([features, extreme_features], dim=1)
        fused = self.feature_fusion(fused)

        # Predict intensity and extent
        intensity, uncertainty = self.intensity_head(fused)
        extent = self.extent_head(fused)

        return intensity, extent, extreme_masks

    def compute_loss(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        extreme_masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for extreme event branch.

        Args:
            predictions: Tuple of (intensity, extent) predictions.
            targets: Target precipitation field [B, 1, H, W].
            extreme_masks: Extreme event masks from detection.

        Returns:
            Tuple of (total loss, per-level losses).
        """
        intensity_pred, extent_pred = predictions

        # Intensity loss
        intensity_loss, per_level_losses = self.loss_fn(
            intensity_pred, targets, extreme_masks
        )

        return intensity_loss, per_level_losses
