"""CSI Loss for Precipitation Forecasting.

Critical Success Index (CSI) loss for evaluating precipitation predictions.
CSI is a key metric for evaluating the accuracy of precipitation forecasts,
particularly for extreme events. The loss is defined as 1 - CSI, so that
minimizing the loss corresponds to maximizing CSI.
"""

import torch
import torch.nn as nn
from typing import List


class CSILoss(nn.Module):
    """Critical Success Index (CSI) loss for precipitation thresholds.

    CSI measures the fraction of observed and/or forecast events that were
    correctly predicted, ignoring correct negatives. It is defined as:

        CSI = hits / (hits + false_alarms + misses)

    The loss is computed as 1 - CSI, so minimizing the loss maximizes CSI.

    This implementation computes CSI across multiple precipitation thresholds
    and returns the weighted average loss.

    Attributes:
        thresholds: List of precipitation thresholds (mm) for CSI computation.
        weights: Importance weights for each threshold.
    """

    def __init__(
        self,
        thresholds: List[float] = None,
        weights: List[float] = None,
    ):
        """Initialize CSILoss.

        Args:
            thresholds: Precipitation thresholds in mm. Default uses standard
                meteorological thresholds: 0.1 (drizzle), 5.0 (light rain),
                10.0 (moderate rain), 25.0 (heavy rain), 50.0 (very heavy rain).
            weights: Importance weights for each threshold. If None, equal
                weights are used.
        """
        super().__init__()
        if thresholds is None:
            thresholds = [0.1, 5.0, 10.0, 25.0, 50.0]
        self.thresholds = thresholds

        if weights is None:
            weights = [1.0] * len(thresholds)
        self.register_buffer(
            "weights_tensor",
            torch.tensor(weights, dtype=torch.float32),
        )

    def compute_single_threshold_csi(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Compute CSI for a single precipitation threshold.

        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W].
            targets: Ground truth values, same shape as predictions.
            threshold: Precipitation threshold in mm.

        Returns:
            CSI value as a scalar tensor.
        """
        pred_binary = (predictions > threshold).float()
        target_binary = (targets > threshold).float()

        hits = (pred_binary * target_binary).sum()
        false_alarms = (pred_binary * (1 - target_binary)).sum()
        misses = ((1 - pred_binary) * target_binary).sum()

        csi = hits / (hits + false_alarms + misses + 1e-8)
        return csi

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted average CSI loss across all thresholds.

        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W].
                Values should be in precipitation units (mm).
            targets: Ground truth precipitation, same shape as predictions.

        Returns:
            Weighted average CSI loss (1 - CSI) as a scalar tensor.
            Minimizing this loss maximizes the overall CSI.
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        weights = self.weights_tensor

        for i, threshold in enumerate(self.thresholds):
            csi = self.compute_single_threshold_csi(
                predictions, targets, threshold
            )
            csi_loss = 1.0 - csi
            total_loss = total_loss + weights[i] * csi_loss

        # Normalize by sum of weights
        total_loss = total_loss / weights.sum()
        return total_loss
