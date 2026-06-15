"""Precipitation Loss Function for PhyDiff-Net.

Multi-task combination loss for precipitation forecasting.
Combines regression losses, classification losses, distribution losses,
physical constraints, and extreme event penalties into a unified
training objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from src.models.losses.csi_loss import CSILoss
from src.models.losses.physics_loss import PhysicsConstraintLoss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in precipitation detection.

    Focal Loss down-weights well-classified examples and focuses training
    on hard negatives. It is defined as:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the correct class.

    This is particularly useful for precipitation forecasting where
    non-precipitation pixels vastly outnumber precipitation pixels.

    Attributes:
        gamma: Focusing parameter. Higher values increase focus on
            hard examples. Default: 2.0.
        alpha: Balancing factor for positive class. Default: 0.25.
        reduction: Loss reduction method ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        """Initialize FocalLoss.

        Args:
            gamma: Focusing parameter. gamma=0 recovers standard BCE.
            alpha: Balancing factor for the positive class (0-1).
            reduction: Specifies reduction: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            predictions: Predicted probabilities [B, ...] (after sigmoid).
            targets: Binary ground truth labels [B, ...].

        Returns:
            Focal loss scalar tensor.
        """
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )

        # Compute p_t
        probs = torch.sigmoid(predictions)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class QuantileLoss(nn.Module):
    """Quantile Loss (pinball loss) for probabilistic precipitation forecasting.

    Quantile loss evaluates how well the predicted quantiles match the
    observed distribution. It is defined as:

        QL(q, y, y_hat) = q * max(y - y_hat, 0) + (1 - q) * max(y_hat - y, 0)

    where q is the target quantile.

    This loss is useful for capturing the full distribution of precipitation
    uncertainty, not just the mean.

    Attributes:
        quantiles: List of target quantiles to evaluate.
    """

    def __init__(
        self,
        quantiles: List[float] = None,
    ):
        """Initialize QuantileLoss.

        Args:
            quantiles: List of quantile levels (0-1). Default uses
                [0.1, 0.5, 0.9] representing 10th, 50th, and 90th percentiles.
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    def single_quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        """Compute loss for a single quantile.

        Args:
            predictions: Predicted values [B, ...].
            targets: Observed values [B, ...].
            quantile: Target quantile level (0-1).

        Returns:
            Scalar loss tensor for this quantile.
        """
        errors = targets - predictions
        loss = torch.max(
            quantile * errors,
            (quantile - 1.0) * errors,
        )
        return loss.mean()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute average quantile loss across all target quantiles.

        Args:
            predictions: Predicted values [B, C, H, W] or [B, H, W].
            targets: Observed values, same shape as predictions.

        Returns:
            Average quantile loss across all quantiles.
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        for quantile in self.quantiles:
            total_loss = total_loss + self.single_quantile_loss(
                predictions, targets, quantile
            )
        return total_loss / len(self.quantiles)


class ExtremeEventLoss(nn.Module):
    """Loss function specialized for extreme precipitation events.

    Applies higher weights to errors on extreme precipitation events,
    ensuring the model does not underestimate rare but impactful events.
    Uses multi-level weighting based on event severity.

    Attributes:
        weights: Dictionary mapping severity levels to loss weights.
        thresholds: Dictionary mapping severity levels to precipitation
            thresholds (mm).
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        thresholds: Dict[str, float] = None,
    ):
        """Initialize ExtremeEventLoss.

        Args:
            weights: Loss weights per severity level. Default:
                {'heavy': 1.0, 'very_heavy': 2.0, 'extreme': 5.0}.
            thresholds: Precipitation thresholds per severity level (mm/6h).
                Default: {'heavy': 25.0, 'very_heavy': 50.0, 'extreme': 100.0}.
        """
        super().__init__()
        if weights is None:
            weights = {"heavy": 1.0, "very_heavy": 2.0, "extreme": 5.0}
        if thresholds is None:
            thresholds = {"heavy": 25.0, "very_heavy": 50.0, "extreme": 100.0}

        self.weights = weights
        self.thresholds = thresholds

    def _compute_level_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss for a specific severity level.

        Uses MSE + CSI + quantile loss combination.

        Args:
            predictions: Predicted values (masked to extreme region).
            targets: Observed values (masked to extreme region).

        Returns:
            Combined loss for this severity level.
        """
        if predictions.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)

        # MSE component
        mse_loss = F.mse_loss(predictions, targets)

        # CSI component (simplified: fraction of correctly predicted extremes)
        pred_binary = (predictions > 0.0).float()
        target_binary = (targets > 0.0).float()
        hits = (pred_binary * target_binary).sum()
        false_alarms = (pred_binary * (1 - target_binary)).sum()
        misses = ((1 - pred_binary) * target_binary).sum()
        csi = hits / (hits + false_alarms + misses + 1e-8)
        csi_loss = 1.0 - csi

        # Quantile component (95th percentile)
        errors = targets - predictions
        quantile_loss = torch.max(
            0.95 * errors, 0.05 * errors
        ).mean()

        return mse_loss + 0.5 * csi_loss + 0.3 * quantile_loss

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        extreme_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute weighted extreme event loss.

        Args:
            predictions: Model predictions [B, C, H, W].
            targets: Ground truth [B, C, H, W].
            extreme_mask: Optional dictionary mapping severity levels to
                boolean masks. If None, masks are computed from targets.

        Returns:
            Weighted sum of losses across all severity levels.
        """
        # Ensure channel dimension
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)
            targets = targets.unsqueeze(1)

        total_loss = torch.tensor(0.0, device=predictions.device)

        # Generate masks from targets if not provided
        if extreme_mask is None:
            extreme_mask = {}
            for level, threshold in self.thresholds.items():
                extreme_mask[level] = (targets > threshold).any(dim=1, keepdim=True)

        for level, weight in self.weights.items():
            level_mask = extreme_mask.get(level)
            if level_mask is None:
                continue

            if level_mask.sum() > 0:
                # level_mask is [B, 1, H, W] - expand to match targets
                expanded_mask = level_mask.expand_as(targets)
                level_loss = self._compute_level_loss(
                    predictions[expanded_mask],
                    targets[expanded_mask],
                )
                total_loss = total_loss + weight * level_loss

        return total_loss


class PrecipitationLoss(nn.Module):
    """Multi-task combination loss for precipitation forecasting.

    Combines multiple loss components to optimize the PhyDiff-Net model
    for precipitation prediction:

    1. **MSE Loss**: Basic regression accuracy.
    2. **Huber Loss**: Robust regression with outlier tolerance.
    3. **Focal Loss**: Classification of precipitation vs non-precipitation.
    4. **CSI Loss**: Threshold-based detection accuracy.
    5. **Quantile Loss**: Distributional accuracy across quantiles.
    6. **Physics Loss**: Physical consistency constraints.
    7. **Extreme Event Loss**: Specialized handling of extreme events.

    The total loss is a weighted combination:

        L_total = w_mse * L_mse + w_huber * L_huber + w_focal * L_focal
                  + w_csi * L_csi + w_quantile * L_quantile
                  + w_physics * L_physics + w_extreme * L_extreme

    Attributes:
        mse_loss: Mean Squared Error loss module.
        huber_loss: Huber loss module for robust regression.
        focal_loss: Focal loss for precipitation detection.
        csi_loss: CSI loss for threshold-based evaluation.
        quantile_loss: Quantile loss for distributional accuracy.
        physics_loss: Physics constraint loss.
        extreme_loss: Extreme event loss.
        weights: Dictionary of loss component weights.
    """

    def __init__(self, config: Dict = None):
        """Initialize PrecipitationLoss.

        Args:
            config: Configuration dictionary with optional overrides:
                - 'mse_weight': Weight for MSE loss (default: 0.2).
                - 'huber_weight': Weight for Huber loss (default: 0.1).
                - 'focal_weight': Weight for Focal loss (default: 0.15).
                - 'csi_weight': Weight for CSI loss (default: 0.25).
                - 'quantile_weight': Weight for Quantile loss (default: 0.1).
                - 'physics_weight': Weight for Physics loss (default: 0.1).
                - 'extreme_weight': Weight for Extreme loss (default: 0.1).
                - 'csi_thresholds': CSI thresholds (default: [0.1, 5, 10, 25, 50]).
                - 'focal_gamma': Focal loss gamma (default: 2.0).
                - 'focal_alpha': Focal loss alpha (default: 0.25).
                - 'quantiles': Quantile levels (default: [0.1, 0.5, 0.9]).
                - 'huber_delta': Huber loss delta (default: 10.0).
        """
        super().__init__()
        if config is None:
            config = {}

        # Base regression losses
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(
            delta=config.get("huber_delta", 10.0)
        )

        # Precipitation-specific losses
        self.focal_loss = FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=config.get("focal_alpha", 0.25),
        )
        self.csi_loss = CSILoss(
            thresholds=config.get(
                "csi_thresholds", [0.1, 5.0, 10.0, 25.0, 50.0]
            )
        )
        self.quantile_loss = QuantileLoss(
            quantiles=config.get("quantiles", [0.1, 0.5, 0.9])
        )

        # Physics and extreme event losses
        self.physics_loss = PhysicsConstraintLoss()
        self.extreme_loss = ExtremeEventLoss()

        # Loss weights (default from design doc)
        self.weights = {
            "mse": config.get("mse_weight", 0.2),
            "huber": config.get("huber_weight", 0.1),
            "focal": config.get("focal_weight", 0.15),
            "csi": config.get("csi_weight", 0.25),
            "quantile": config.get("quantile_weight", 0.1),
            "physics": config.get("physics_weight", 0.1),
            "extreme": config.get("extreme_weight", 0.1),
        }

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task precipitation loss.

        Args:
            predictions: Model predictions [B, T, H, W] or [B, C, H, W]
                or [B, H, W]. Values should be precipitation amounts (mm).
            targets: Ground truth precipitation, same shape as predictions.
            metadata: Optional dictionary containing:
                - 'extreme_events': Dictionary of extreme event masks.
                - 'moisture': Moisture fields for physics constraints.

        Returns:
            Dictionary containing individual loss components and total loss:
                - 'mse': MSE loss.
                - 'huber': Huber loss.
                - 'focal': Focal loss.
                - 'csi': CSI loss.
                - 'quantile': Quantile loss.
                - 'physics': Physics constraint loss.
                - 'extreme': Extreme event loss.
                - 'total': Weighted total loss.
        """
        if metadata is None:
            metadata = {}

        losses = {}

        # 1. Base regression losses
        losses["mse"] = self.mse_loss(predictions, targets)
        losses["huber"] = self.huber_loss(predictions, targets)

        # 2. Classification loss (precipitation vs non-precipitation)
        pred_binary = (predictions > 0.1).float()
        target_binary = (targets > 0.1).float()
        losses["focal"] = self.focal_loss(pred_binary, target_binary)

        # 3. CSI loss (threshold-based detection)
        losses["csi"] = self.csi_loss(predictions, targets)

        # 4. Quantile loss (distributional accuracy)
        losses["quantile"] = self.quantile_loss(predictions, targets)

        # 5. Physics constraint loss
        losses["physics"] = self.physics_loss(predictions, metadata)

        # 6. Extreme event loss
        extreme_events = metadata.get("extreme_events")
        if extreme_events is not None:
            losses["extreme"] = self.extreme_loss(
                predictions, targets, extreme_events
            )
        else:
            losses["extreme"] = self.extreme_loss(predictions, targets)

        # Weighted total loss
        total_loss = (
            self.weights["mse"] * losses["mse"]
            + self.weights["huber"] * losses["huber"]
            + self.weights["focal"] * losses["focal"]
            + self.weights["csi"] * losses["csi"]
            + self.weights["quantile"] * losses["quantile"]
            + self.weights["physics"] * losses["physics"]
            + self.weights["extreme"] * losses["extreme"]
        )

        losses["total"] = total_loss
        return losses
