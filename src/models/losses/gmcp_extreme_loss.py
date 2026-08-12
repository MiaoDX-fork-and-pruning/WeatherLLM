"""GMCP extreme-event-aware loss for PhyDiff-Net training.

This loss bridges the gap between the normalized space the model operates in
(log_minmax) and the physical space (mm/6h) where precipitation thresholds
for CSI and extreme-event weighting are defined.

Components:
    - MSE + MAE in normalized space (regression accuracy).
    - Multi-threshold CSI loss in physical space (detection skill).
    - Extreme-event weighted MSE in physical space (counteracts the severe
      class imbalance documented in gmcp_analysis_20260630.md: >=25 mm/6h
      is only 0.19% of grid points).

The denormalization is differentiable and consistent with
``GMCPSequenceDataset._normalize_input`` / ``_denormalize_input``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GMCPExtremeLoss(nn.Module):
    """Extreme-event-aware loss for GMCP-only precipitation forecasting.

    Computes regression losses in normalized space and threshold-based
    detection/extreme losses in physical (denormalized) space, then combines
    them with configurable weights.

    Args:
        norm_min: Minimum of the log1p-transformed data (log_minmax lower
            bound). Required when ``normalize="log_minmax"``.
        norm_max: Maximum of the log1p-transformed data (log_minmax upper
            bound). Required when ``normalize="log_minmax"``.
        normalize: Normalization scheme used by the dataset. Currently
            supports ``"log_minmax"`` and ``None``.
        thresholds: Precipitation thresholds (mm/6h) for CSI computation.
        extreme_thresholds: Thresholds (mm/6h) defining extreme-event masks.
            Errors at grid points exceeding these are up-weighted.
        extreme_weights: Per-threshold weight multipliers for extreme MSE.
            Must align with ``extreme_thresholds`` in length.
        mse_weight: Weight for normalized-space MSE.
        mae_weight: Weight for normalized-space MAE.
        csi_weight: Weight for physical-space CSI loss (1 - CSI).
        extreme_weight: Weight for physical-space extreme-event MSE.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        norm_min: Optional[float] = None,
        norm_max: Optional[float] = None,
        normalize: Optional[str] = "log_minmax",
        thresholds: Optional[List[float]] = None,
        extreme_thresholds: Optional[List[float]] = None,
        extreme_weights: Optional[List[float]] = None,
        mse_weight: float = 1.0,
        mae_weight: float = 0.5,
        csi_weight: float = 0.5,
        extreme_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.eps = eps

        if normalize == "log_minmax":
            if norm_min is None or norm_max is None:
                raise ValueError(
                    "norm_min and norm_max are required for log_minmax"
                )
            self.register_buffer("norm_min", torch.tensor(float(norm_min)))
            self.register_buffer("norm_max", torch.tensor(float(norm_max)))
        elif normalize is None:
            self.register_buffer("norm_min", torch.tensor(0.0))
            self.register_buffer("norm_max", torch.tensor(1.0))
        else:
            raise ValueError(f"Unsupported normalization: {normalize}")

        self.thresholds = thresholds or [0.1, 5.0, 10.0, 25.0, 50.0]
        self.extreme_thresholds = extreme_thresholds or [25.0, 50.0, 100.0]
        self.extreme_weights = extreme_weights or [2.0, 4.0, 8.0]
        if len(self.extreme_weights) != len(self.extreme_thresholds):
            raise ValueError(
                "extreme_weights must match extreme_thresholds in length"
            )

        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.csi_weight = csi_weight
        self.extreme_weight = extreme_weight

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized predictions/targets back to physical mm/6h.

        Inverts the log_minmax transform: ``x -> expm1(x*(max-min)+min)``.
        When ``normalize`` is None, returns x unchanged.

        Args:
            x: Normalized tensor.

        Returns:
            Tensor in physical precipitation units (mm/6h).
        """
        if self.normalize == "log_minmax":
            log_x = x * (self.norm_max - self.norm_min) + self.norm_min
            return torch.expm1(log_x)
        return x

    def _csi_loss_single(
        self,
        pred_phys: torch.Tensor,
        target_phys: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Compute 1 - CSI for a single threshold in physical space."""
        pred_bin = (pred_phys > threshold).float()
        target_bin = (target_phys > threshold).float()
        hits = (pred_bin * target_bin).sum()
        false_alarms = (pred_bin * (1 - target_bin)).sum()
        misses = ((1 - pred_bin) * target_bin).sum()
        csi = hits / (hits + false_alarms + misses + self.eps)
        return 1.0 - csi

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined extreme-event-aware loss.

        Args:
            predictions: Model predictions in normalized space [B, T, H, W].
            targets: Ground truth in normalized space, same shape.

        Returns:
            Dictionary with per-component losses and the weighted ``total``.
        """
        losses: Dict[str, torch.Tensor] = {}

        # Regression losses in normalized space.
        losses["mse"] = F.mse_loss(predictions, targets)
        losses["mae"] = F.l1_loss(predictions, targets)

        # Denormalize to physical space for threshold-based losses.
        pred_phys = self.denormalize(predictions)
        target_phys = self.denormalize(targets)

        # Multi-threshold CSI loss (detection skill across intensities).
        csi_losses = [
            self._csi_loss_single(pred_phys, target_phys, t)
            for t in self.thresholds
        ]
        losses["csi"] = torch.stack(csi_losses).mean()

        # Extreme-event weighted MSE in physical space.
        # Build a per-gridpoint weight map: base weight 1, multiplied by the
        # extreme weight at each threshold the target exceeds.
        weight_map = torch.ones_like(target_phys)
        for thr, w in zip(self.extreme_thresholds, self.extreme_weights):
            weight_map = torch.where(
                target_phys > thr, weight_map * w, weight_map
            )
        sq_err = (pred_phys - target_phys) ** 2
        losses["extreme"] = (sq_err * weight_map).mean()

        total = (
            self.mse_weight * losses["mse"]
            + self.mae_weight * losses["mae"]
            + self.csi_weight * losses["csi"]
            + self.extreme_weight * losses["extreme"]
        )
        losses["total"] = total
        return losses
