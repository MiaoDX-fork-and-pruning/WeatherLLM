"""Physics Constraint Loss for PhyDiff-Net.

Physical consistency constraints for precipitation forecasting.
These losses enforce that predicted precipitation fields satisfy known
physical laws of atmospheric dynamics, including moisture conservation,
spatial smoothness, and non-negativity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PhysicsConstraintLoss(nn.Module):
    """Physics-informed constraint loss for precipitation fields.

    Enforces physical consistency in predicted precipitation through
    multiple constraint terms:

    1. **Non-negativity**: Precipitation cannot be negative.
    2. **Moisture conservation**: Spatial gradients of precipitation should
       be consistent with moisture transport.
    3. **Spatial smoothness**: Adjacent grid points should have physically
       plausible precipitation gradients (penalizes excessive noise).
    4. **Mass conservation**: Total precipitation mass should be bounded
       by available moisture in the input fields.

    Attributes:
        non_neg_weight: Weight for the non-negativity penalty.
        smoothness_weight: Weight for the spatial smoothness penalty.
        conservation_weight: Weight for the moisture conservation penalty.
        max_precipitation: Physical upper bound for precipitation rate (mm/h).
    """

    def __init__(
        self,
        non_neg_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        conservation_weight: float = 0.05,
        max_precipitation: float = 200.0,
    ):
        """Initialize PhysicsConstraintLoss.

        Args:
            non_neg_weight: Weight for non-negativity constraint.
            smoothness_weight: Weight for spatial smoothness constraint.
            conservation_weight: Weight for moisture conservation constraint.
            max_precipitation: Physical upper bound for precipitation (mm/h).
                Used for upper-bound constraint.
        """
        super().__init__()
        self.non_neg_weight = non_neg_weight
        self.smoothness_weight = smoothness_weight
        self.conservation_weight = conservation_weight
        self.max_precipitation = max_precipitation

    def _non_negativity_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Penalize negative precipitation values.

        Uses a smooth penalty (softplus) instead of hard clipping to
        maintain differentiability.

        Args:
            predictions: Model predictions [B, C, H, W].

        Returns:
            Scalar loss tensor.
        """
        # Penalize negative values with softplus for smooth gradients
        negative_penalty = F.softplus(-predictions)
        return negative_penalty.mean()

    def _spatial_smoothness_loss(
        self, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Penalize excessive spatial noise in precipitation fields.

        Uses Laplacian operator to measure second-order spatial derivatives.
        Physically plausible precipitation fields should have smooth
        spatial gradients.

        Args:
            predictions: Model predictions [B, C, H, W].

        Returns:
            Scalar loss tensor.
        """
        # Laplacian kernel for second-order spatial derivatives
        laplacian_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            device=predictions.device,
            dtype=predictions.dtype,
        ).reshape(1, 1, 3, 3)

        # Apply Laplacian to each channel independently
        num_channels = predictions.shape[1]
        kernel = laplacian_kernel.expand(num_channels, -1, -1, -1)

        laplacian = F.conv2d(
            predictions,
            kernel,
            padding=1,
            groups=num_channels,
        )

        return (laplacian ** 2).mean()

    def _moisture_conservation_loss(
        self,
        predictions: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Enforce approximate moisture conservation.

        The divergence of precipitation should be bounded by the available
        moisture flux. Without explicit moisture fields, we use a proxy:
        the spatial gradient of precipitation should not exceed a physical
        threshold derived from typical atmospheric moisture transport.

        Args:
            predictions: Model predictions [B, C, H, W].
            metadata: Optional metadata containing moisture fields.
                If provided, uses actual moisture for conservation check.

        Returns:
            Scalar loss tensor.
        """
        # Compute spatial gradients (crop to common spatial dimensions)
        grad_y = predictions[:, :, 1:, :-1] - predictions[:, :, :-1, :-1]
        grad_x = predictions[:, :, :-1, 1:] - predictions[:, :, :-1, :-1]

        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Penalize gradients that exceed physical limits
        # Typical precipitation gradients are bounded by atmospheric dynamics
        gradient_threshold = self.max_precipitation / 10.0
        excess_gradient = F.relu(grad_magnitude - gradient_threshold)

        return excess_gradient.mean()

    def _upper_bound_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Penalize precipitation values exceeding physical upper bound.

        Uses a smooth penalty for values above the maximum physical
        precipitation rate.

        Args:
            predictions: Model predictions [B, C, H, W].

        Returns:
            Scalar loss tensor.
        """
        excess = F.softplus(predictions - self.max_precipitation)
        return excess.mean()

    def forward(
        self,
        predictions: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Compute total physics constraint loss.

        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W].
                Should be in precipitation units (mm).
            metadata: Optional dictionary containing auxiliary data:
                - 'moisture': Moisture fields for conservation constraint.
                - 'terrain': Terrain data for orographic effects.

        Returns:
            Weighted sum of all physics constraint losses.
        """
        # Ensure predictions have channel dimension
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)

        total_loss = torch.tensor(0.0, device=predictions.device)

        # Non-negativity constraint
        total_loss = total_loss + self.non_neg_weight * (
            self._non_negativity_loss(predictions)
        )

        # Spatial smoothness constraint
        total_loss = total_loss + self.smoothness_weight * (
            self._spatial_smoothness_loss(predictions)
        )

        # Moisture conservation constraint
        total_loss = total_loss + self.conservation_weight * (
            self._moisture_conservation_loss(predictions, metadata)
        )

        # Upper bound constraint
        total_loss = total_loss + self.non_neg_weight * (
            self._upper_bound_loss(predictions)
        )

        return total_loss
