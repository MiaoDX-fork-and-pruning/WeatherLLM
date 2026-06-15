"""
PhyDiff-Net Evaluation Module.

Provides precipitation forecast evaluation metrics and visualization tools
for assessing the performance of physics-guided diffusion-based
precipitation prediction models.
"""

from .metrics import PrecipitationMetrics
from .visualization import PrecipitationVisualizer

__all__ = ["PrecipitationMetrics", "PrecipitationVisualizer"]
