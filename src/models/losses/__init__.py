"""Loss functions for PhyDiff-Net precipitation forecasting."""

from src.models.losses.precipitation_loss import PrecipitationLoss
from src.models.losses.csi_loss import CSILoss
from src.models.losses.physics_loss import PhysicsConstraintLoss

__all__ = [
    "PrecipitationLoss",
    "CSILoss",
    "PhysicsConstraintLoss",
]
