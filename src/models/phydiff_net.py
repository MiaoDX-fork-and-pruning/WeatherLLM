"""PhyDiff-Net: Physics-guided Diffusion Network for Precipitation Forecasting.

This module implements the main PhyDiff-Net architecture, integrating:
- Multi-scale spatiotemporal encoder
- Physics-constrained diffusion module
- Extreme event aware branch
- Spatiotemporal heterogeneity modeling
- Multi-task output head

The architecture is designed for high-resolution precipitation forecasting
over China, achieving state-of-the-art performance through physics-guided
diffusion and extreme event specialization.

Author: weather-model-trainer
Date: 2026-06-15
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import PhysicsConstrainedDiffusion
from .extreme_branch import ExtremeEventBranch
from .heterogeneity import SpatiotemporalHeterogeneity


class SpatialAttention(nn.Module):
    """Spatial attention mechanism.

    Args:
        channels: Number of channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Attended tensor [B, C, H, W].
        """
        B, C, _, _ = x.shape
        attn = self.attention(x).view(B, C, 1, 1)
        return x * attn


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time series features.

    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply temporal attention.

        Args:
            features: Input features [B, C, H, W].
            timestamps: Optional timestamps [B, T].

        Returns:
            Attended features [B, C, H, W].
        """
        B, C, H, W = features.shape

        # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        features_flat = features.view(B, C, H * W).transpose(1, 2)

        # Self-attention
        attn_out, _ = self.attention(features_flat, features_flat, features_flat)
        attn_out = self.norm(attn_out + features_flat)

        # Reshape back
        return attn_out.transpose(1, 2).view(B, C, H, W)


class CrossResolutionFusion(nn.Module):
    """Cross-resolution fusion for ERA5 and GMCP data.

    This module fuses features from different resolutions (ERA5 at 0.25°
    and GMCP at 0.1°) using learnable upsampling and attention.

    Args:
        era5_channels: Number of ERA5 channels.
        gmcp_channels: Number of GMCP channels.
        hidden_dim: Hidden dimension.
    """

    def __init__(
        self,
        era5_channels: int = 19,
        gmcp_channels: int = 1,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # ERA5 processing (lower resolution)
        self.era5_encoder = nn.Sequential(
            nn.Conv2d(era5_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
        )

        # GMCP processing (higher resolution)
        self.gmcp_encoder = nn.Sequential(
            nn.Conv2d(gmcp_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
        )

        # Cross-resolution attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim // 2, num_heads=4, dropout=0.1, batch_first=True
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        era5_features: torch.Tensor,
        gmcp_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse features from different resolutions.

        Args:
            era5_features: ERA5 features [B, C_era5, H, W].
            gmcp_features: GMCP features [B, C_gmcp, H', W'].

        Returns:
            Fused features [B, hidden_dim, H, W].
        """
        B = era5_features.shape[0]

        # Process each source
        era5_enc = self.era5_encoder(era5_features)
        gmcp_enc = self.gmcp_encoder(gmcp_features)

        # Upsample GMCP to match ERA5 spatial dimensions
        gmcp_upsampled = F.interpolate(
            gmcp_enc,
            size=(era5_enc.shape[2], era5_enc.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape for attention
        H, W = era5_enc.shape[2], era5_enc.shape[3]
        era5_flat = era5_enc.view(B, -1, H * W).transpose(1, 2)  # [B, H*W, C]
        gmcp_flat = gmcp_upsampled.view(B, -1, H * W).transpose(1, 2)

        # Cross attention
        cross_out, _ = self.cross_attention(era5_flat, gmcp_flat, gmcp_flat)

        # Reshape back
        cross_out = cross_out.transpose(1, 2).view(B, -1, H, W)

        # Concatenate and fuse
        fused = torch.cat([era5_enc, cross_out], dim=1)
        return self.fusion(fused)


class MultiScaleEncoder(nn.Module):
    """Multi-scale spatiotemporal encoder.

    This encoder extracts features from ERA5 and GMCP data at multiple
    scales, capturing both large-scale atmospheric patterns and local
    precipitation details.

    Args:
        in_channels_era5: Number of ERA5 input channels.
        in_channels_gmcp: Number of GMCP input channels.
        hidden_dim: Hidden dimension.
        num_scales: Number of scale branches.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels_era5: int = 19,
        in_channels_gmcp: int = 1,
        hidden_dim: int = 256,
        num_scales: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        # Multi-scale convolution branches
        kernel_sizes = [3, 5, 7][:num_scales]
        self.scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels_era5,
                    hidden_dim // num_scales,
                    kernel_size=k,
                    padding=k // 2,
                ),
                nn.GroupNorm(8, hidden_dim // num_scales),
                nn.SiLU(),
                nn.Dropout2d(dropout),
            )
            for k in kernel_sizes
        ])

        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim, num_heads=4)

        # Spatial attention
        self.spatial_attention = SpatialAttention(hidden_dim, reduction=16)

        # Cross-resolution fusion
        self.cross_resolution_fusion = CrossResolutionFusion(
            era5_channels=in_channels_era5,
            gmcp_channels=in_channels_gmcp,
            hidden_dim=hidden_dim,
        )

        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
        )

    def forward(
        self,
        era5_data: torch.Tensor,
        gmcp_data: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode multi-scale spatiotemporal features.

        Args:
            era5_data: ERA5 input data [B, C_era5, H, W].
            gmcp_data: GMCP input data [B, C_gmcp, H', W'].
            timestamps: Optional timestamps [B, T].

        Returns:
            Multi-scale features [B, hidden_dim, H, W].
        """
        # Multi-scale feature extraction
        multi_scale_features = []
        for branch in self.scale_branches:
            scale_feat = branch(era5_data)
            multi_scale_features.append(scale_feat)

        # Concatenate multi-scale features
        concat_features = torch.cat(multi_scale_features, dim=1)

        # Temporal attention
        temporal_feat = self.temporal_attention(concat_features, timestamps)

        # Spatial attention
        spatial_feat = self.spatial_attention(temporal_feat)

        # Cross-resolution fusion (uses raw ERA5 data for resolution alignment)
        cross_fused = self.cross_resolution_fusion(era5_data, gmcp_data)

        # Combine processed features with cross-resolution features
        fused_features = spatial_feat + cross_fused

        # Refinement
        refined = self.refinement(fused_features) + fused_features

        return refined


class MultiTaskOutputHead(nn.Module):
    """Multi-task output head for precipitation forecasting.

    This head generates predictions for multiple tasks:
    - Primary precipitation prediction
    - Uncertainty estimation
    - Auxiliary tasks (e.g., precipitation type)

    Args:
        hidden_dim: Hidden dimension.
        output_channels: Number of output channels.
        forecast_horizon: Number of forecast time steps.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_channels: int = 1,
        forecast_horizon: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # Feature processing
        self.processing = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Precipitation prediction head
        self.precip_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, forecast_horizon * output_channels, kernel_size=1),
            nn.ReLU(),  # Non-negative precipitation
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, forecast_horizon * output_channels, kernel_size=1),
            nn.Softplus(),  # Positive uncertainty
        )

        # Precipitation type classification (optional)
        self.type_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, forecast_horizon * 3, kernel_size=1),
            # 0: no precip, 1: rain, 2: snow
        )

    def forward(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate multi-task predictions.

        Args:
            features: Encoded features [B, C, H, W].

        Returns:
            Dictionary containing:
            - 'precipitation': Predicted precipitation [B, T, H, W]
            - 'uncertainty': Uncertainty estimates [B, T, H, W]
            - 'precip_type': Precipitation type logits [B, T, 3, H, W]
        """
        B, C, H, W = features.shape

        # Process features
        processed = self.processing(features)

        # Generate predictions
        precip = self.precip_head(processed)
        precip = precip.view(B, self.forecast_horizon, 1, H, W)

        uncertainty = self.uncertainty_head(processed)
        uncertainty = uncertainty.view(B, self.forecast_horizon, 1, H, W)

        precip_type = self.type_head(processed)
        precip_type = precip_type.view(B, self.forecast_horizon, 3, H, W)

        return {
            "precipitation": precip.squeeze(2),  # [B, T, H, W]
            "uncertainty": uncertainty.squeeze(2),
            "precip_type": precip_type,
        }


class PhyDiffNet(nn.Module):
    """PhyDiff-Net: Physics-guided Diffusion Network.

    This is the main model architecture for high-resolution precipitation
    forecasting over China. It integrates:

    1. Multi-scale spatiotemporal encoder for feature extraction
    2. Physics-constrained diffusion for physically consistent generation
    3. Extreme event aware branch for extreme precipitation handling
    4. Spatiotemporal heterogeneity modeling for regional patterns
    5. Multi-task output head for comprehensive predictions

    Args:
        config: Model configuration dictionary containing:
            - encoder: Encoder configuration
            - diffusion: Diffusion module configuration
            - extreme_branch: Extreme branch configuration
            - heterogeneity: Heterogeneity module configuration
            - output: Output head configuration

    Example:
        >>> config = {
        ...     'encoder': {'in_channels': 19, 'hidden_dim': 256},
        ...     'diffusion': {'hidden_dim': 256, 'num_steps': 1000},
        ...     'extreme_branch': {'hidden_dim': 256},
        ...     'heterogeneity': {'hidden_dim': 256},
        ...     'output': {'hidden_dim': 256, 'forecast_horizon': 4}
        ... }
        >>> model = PhyDiffNet(config)
        >>> era5 = torch.randn(2, 19, 32, 32)
        >>> gmcp = torch.randn(2, 1, 128, 128)
        >>> output = model(era5, gmcp)
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Multi-scale encoder
        encoder_config = config.get("encoder", {})
        self.encoder = MultiScaleEncoder(
            in_channels_era5=encoder_config.get("in_channels", 19),
            in_channels_gmcp=encoder_config.get("gmcp_channels", 1),
            hidden_dim=encoder_config.get("hidden_dim", 256),
            num_scales=encoder_config.get("num_scales", 3),
            dropout=encoder_config.get("dropout", 0.1),
        )

        # Physics-constrained diffusion module
        diffusion_config = config.get("diffusion", {})
        self.diffusion = PhysicsConstrainedDiffusion(
            hidden_dim=diffusion_config.get("hidden_dim", 256),
            num_diffusion_steps=diffusion_config.get("num_steps", 1000),
            dropout=diffusion_config.get("dropout", 0.1),
        )

        # Extreme event aware branch
        extreme_config = config.get("extreme_branch", {})
        self.extreme_branch = ExtremeEventBranch(
            in_channels=encoder_config.get("hidden_dim", 256),
            hidden_dim=extreme_config.get("hidden_dim", 256),
            dropout=extreme_config.get("dropout", 0.1),
        )

        # Spatiotemporal heterogeneity modeling
        heterogeneity_config = config.get("heterogeneity", {})
        self.heterogeneity = SpatiotemporalHeterogeneity(
            hidden_dim=heterogeneity_config.get("hidden_dim", 256),
            num_heads=heterogeneity_config.get("num_heads", 8),
            num_regions=heterogeneity_config.get("num_regions", 4),
            num_frequencies=heterogeneity_config.get("num_frequencies", 8),
            dropout=heterogeneity_config.get("dropout", 0.1),
        )

        # Multi-task output head
        output_config = config.get("output", {})
        self.output_head = MultiTaskOutputHead(
            hidden_dim=output_config.get("hidden_dim", 256),
            output_channels=output_config.get("output_channels", 1),
            forecast_horizon=output_config.get("forecast_horizon", 4),
            dropout=output_config.get("dropout", 0.1),
        )

        # Feature fusion before output
        hidden_dim = encoder_config.get("hidden_dim", 256)
        self.extreme_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        era5_data: torch.Tensor,
        gmcp_data: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            era5_data: ERA5 atmospheric data [B, C_era5, H, W].
            gmcp_data: GMCP precipitation data [B, C_gmcp, H', W'].
            timestamps: Optional timestamps [B, T].
            return_intermediate: If True, return intermediate features.

        Returns:
            Dictionary containing:
            - 'precipitation': Predicted precipitation [B, T, H, W]
            - 'uncertainty': Uncertainty estimates [B, T, H, W]
            - 'extreme_masks': Extreme event masks
            - Optional intermediate features
        """
        # Encode features
        encoded_features = self.encoder(era5_data, gmcp_data, timestamps)

        # Heterogeneity modeling
        heterogeneity_features = self.heterogeneity(encoded_features, timestamps)

        # Extreme event processing
        extreme_intensity, extreme_extent, extreme_masks = self.extreme_branch(
            encoded_features, gmcp_data.mean(dim=1, keepdim=True)
        )

        # Fuse features from different branches
        extreme_feat = self.extreme_proj(extreme_intensity)
        fused_features = torch.cat([
            encoded_features,
            heterogeneity_features,
            extreme_feat,
        ], dim=1)
        fused_features = self.feature_fusion(fused_features)

        # Generate output
        output = self.output_head(fused_features)

        # Add extreme event information
        output["extreme_masks"] = extreme_masks
        output["extreme_intensity"] = extreme_intensity
        output["extreme_extent"] = extreme_extent

        if return_intermediate:
            output["encoded_features"] = encoded_features
            output["heterogeneity_features"] = heterogeneity_features

        return output

    @torch.no_grad()
    def sample(
        self,
        era5_data: torch.Tensor,
        gmcp_data: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        use_ddim: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Generate precipitation predictions via diffusion sampling.

        Args:
            era5_data: ERA5 atmospheric data [B, C_era5, H, W].
            gmcp_data: GMCP precipitation data [B, C_gmcp, H', W'].
            timestamps: Optional timestamps [B, T].
            num_samples: Number of samples to generate.
            use_ddim: If True, use DDIM sampling (faster).

        Returns:
            Dictionary containing generated predictions.
        """
        B = era5_data.shape[0]

        # Encode features (condition)
        encoded_features = self.encoder(era5_data, gmcp_data, timestamps)

        # Heterogeneity features
        heterogeneity_features = self.heterogeneity(encoded_features, timestamps)

        # Get spatial dimensions
        H, W = encoded_features.shape[2], encoded_features.shape[3]

        # Generate via diffusion
        shape = (B, 1, H, W)

        if use_ddim:
            precipitation = self.diffusion.ddim_sample(
                condition=encoded_features,
                shape=shape,
                device=era5_data.device,
                ddim_steps=50,
            )
        else:
            precipitation = self.diffusion.sample(
                condition=encoded_features,
                shape=shape,
                device=era5_data.device,
            )

        # Process with extreme branch
        extreme_intensity, extreme_extent, extreme_masks = self.extreme_branch(
            encoded_features, precipitation
        )

        return {
            "precipitation": precipitation,
            "extreme_masks": extreme_masks,
            "extreme_intensity": extreme_intensity,
            "extreme_extent": extreme_extent,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        physics_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss for training.

        Args:
            predictions: Model predictions from forward().
            targets: Target precipitation [B, T, H, W].
            physics_loss: Physics constraint loss from diffusion.

        Returns:
            Dictionary of losses.
        """
        losses = {}

        # Primary precipitation loss (MSE)
        precip_pred = predictions["precipitation"]
        if precip_pred.dim() == 4 and targets.dim() == 4:
            mse_loss = F.mse_loss(precip_pred, targets)
        else:
            mse_loss = F.mse_loss(precip_pred, targets)
        losses["mse"] = mse_loss

        # Uncertainty-aware loss
        if "uncertainty" in predictions:
            uncertainty = predictions["uncertainty"]
            nll_loss = 0.5 * (
                torch.log(uncertainty + 1e-8) +
                (precip_pred - targets) ** 2 / (uncertainty + 1e-8)
            ).mean()
            losses["nll"] = nll_loss

        # Extreme event loss
        extreme_masks = predictions.get("extreme_masks", {})
        if extreme_masks:
            extreme_loss = torch.tensor(0.0, device=precip_pred.device)
            for level, mask in extreme_masks.items():
                if mask.sum() > 0:
                    level_loss = F.mse_loss(
                        precip_pred[mask.expand_as(precip_pred)],
                        targets[mask.expand_as(targets)]
                    )
                    extreme_loss = extreme_loss + level_loss
            losses["extreme"] = extreme_loss / max(len(extreme_masks), 1)

        # Physics constraint loss
        losses["physics"] = physics_loss

        # Heterogeneity regularization
        heterogeneity_features = predictions.get("heterogeneity_features")
        if heterogeneity_features is not None:
            het_loss = self.heterogeneity.compute_heterogeneity_loss(
                heterogeneity_features
            )
            losses["heterogeneity"] = het_loss

        # Total loss with weights
        total_loss = (
            0.4 * losses.get("mse", 0) +
            0.1 * losses.get("nll", 0) +
            0.2 * losses.get("extreme", 0) +
            0.2 * losses.get("physics", 0) +
            0.1 * losses.get("heterogeneity", 0)
        )
        losses["total"] = total_loss

        return losses
