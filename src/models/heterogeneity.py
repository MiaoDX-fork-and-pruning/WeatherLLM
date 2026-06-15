"""Spatiotemporal Heterogeneity Module for PhyDiff-Net.

This module models the spatiotemporal heterogeneity of precipitation fields,
including spatial non-stationarity and temporal variability characteristics.

The key components are:
- Spatial heterogeneity modeling for regional precipitation patterns
- Temporal variability modeling for multi-scale time variations
- Adaptive feature modulation for dynamic adjustment

Author: weather-model-trainer
Date: 2026-06-15
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionEncoder(nn.Module):
    """Encoder for region-specific precipitation patterns.

    This encoder captures the unique precipitation characteristics
    of different geographical regions.

    Args:
        in_channels: Number of input channels.
        hidden_dim: Hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode region-specific features.

        Args:
            features: Input features [B, C, H, W].

        Returns:
            Encoded features [B, hidden_dim, H, W].
        """
        return self.encoder(features)


class BoundarySmoothing(nn.Module):
    """Boundary smoothing for region transitions.

    This module smooths the transitions between different regions
    to avoid artificial discontinuities.

    Args:
        kernel_size: Size of the smoothing kernel.
        sigma: Standard deviation for Gaussian smoothing.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create Gaussian kernel
        self.register_buffer(
            "kernel",
            self._create_gaussian_kernel(kernel_size, sigma),
        )

    def _create_gaussian_kernel(
        self, kernel_size: int, sigma: float
    ) -> torch.Tensor:
        """Create 2D Gaussian kernel.

        Args:
            kernel_size: Size of the kernel.
            sigma: Standard deviation.

        Returns:
            Gaussian kernel tensor.
        """
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        x = x ** 2
        kernel_1d = torch.exp(-x / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply boundary smoothing.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Smoothed tensor [B, C, H, W].
        """
        C = x.shape[1]
        kernel = self.kernel.expand(C, -1, -1, -1)
        padding = self.kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=C)


class SpatialHeterogeneity(nn.Module):
    """Spatial non-stationarity modeling module.

    This module captures the spatial heterogeneity of precipitation
    fields by modeling regional variations and their interactions.

    Args:
        hidden_dim: Hidden dimension.
        num_regions: Number of geographical regions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_regions: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_regions = num_regions

        # Region-specific encoders
        self.region_encoders = nn.ModuleList([
            RegionEncoder(hidden_dim, hidden_dim // num_regions, dropout)
            for _ in range(num_regions)
        ])

        # Boundary smoothing
        self.boundary_smoothing = BoundarySmoothing(kernel_size=5, sigma=1.0)

        # Region weight predictor
        self.region_weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_regions),
        )

        # Spatial attention for feature modulation
        region_dim = hidden_dim // num_regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(region_dim, region_dim // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(region_dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Output projection to restore full hidden_dim
        self.output_proj = nn.Conv2d(region_dim, hidden_dim, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Model spatial heterogeneity.

        Args:
            features: Input features [B, C, H, W].

        Returns:
            Heterogeneity features [B, C, H, W].
        """
        B, C, H, W = features.shape

        # Predict region weights
        region_weights = self.region_weight_predictor(features)  # [B, num_regions]
        region_weights = F.softmax(region_weights, dim=-1)

        # Extract region-specific features
        region_features = []
        for i, encoder in enumerate(self.region_encoders):
            region_feat = encoder(features)
            # Apply region weight
            weight = region_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1)
            region_features.append(region_feat * weight)

        # Combine region features
        heterogeneity_feature = sum(region_features)

        # Apply spatial attention
        spatial_attn = self.spatial_attention(heterogeneity_feature)
        heterogeneity_feature = heterogeneity_feature * spatial_attn

        # Apply boundary smoothing
        heterogeneity_feature = self.boundary_smoothing(heterogeneity_feature)

        # Project back to full hidden_dim
        heterogeneity_feature = self.output_proj(heterogeneity_feature)

        return heterogeneity_feature


class TemporalEncoding(nn.Module):
    """Temporal position encoding using sinusoidal functions.

    Args:
        hidden_dim: Hidden dimension.
        max_len: Maximum sequence length.
    """

    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_dim]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add temporal encoding.

        Args:
            x: Input tensor [B, T, C].

        Returns:
            Encoded tensor [B, T, C].
        """
        return x + self.pe[:, :x.size(1), :]


class FrequencyEncoder(nn.Module):
    """Encoder for temporal frequency components.

    This module captures periodic patterns at different frequencies
    using learnable frequency decomposition.

    Args:
        hidden_dim: Hidden dimension.
        num_frequencies: Number of frequency components.
    """

    def __init__(self, hidden_dim: int, num_frequencies: int = 8):
        super().__init__()
        self.num_frequencies = num_frequencies

        # Frequency-specific projections
        self.frequency_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // num_frequencies)
            for _ in range(num_frequencies)
        ])

        # Learnable frequency weights
        self.frequency_weights = nn.Parameter(torch.ones(num_frequencies) / num_frequencies)

        # Output projection
        self.output_proj = nn.Linear(
            hidden_dim // num_frequencies * num_frequencies, hidden_dim
        )

    def forward(
        self, features: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Encode frequency components.

        Args:
            features: Temporal features [B, T, C].
            timestamps: Timestamp indices [B, T].

        Returns:
            Frequency-encoded features [B, T, C].
        """
        B, T, C = features.shape
        freq_features = []

        for i in range(self.num_frequencies):
            # Create sinusoidal basis
            freq = (i + 1) * 0.1
            phase = 2 * torch.pi * freq * timestamps.float()
            sin_part = torch.sin(phase).unsqueeze(-1)
            cos_part = torch.cos(phase).unsqueeze(-1)

            # Combine with features
            freq_input = features * (sin_part + cos_part)
            freq_feat = self.frequency_projections[i](freq_input)
            freq_features.append(freq_feat)

        # Concatenate and project
        concat_features = torch.cat(freq_features, dim=-1)
        output = self.output_proj(concat_features)

        # Apply frequency weights
        weights = F.softmax(self.frequency_weights, dim=0)
        weighted_features = sum(
            w * f for w, f in zip(weights, freq_features)
        )

        return weighted_features


class TemporalVariability(nn.Module):
    """Temporal variability modeling module.

    This module captures the temporal dynamics of precipitation fields
    at multiple time scales.

    Args:
        hidden_dim: Hidden dimension.
        num_frequencies: Number of frequency components.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_frequencies: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Temporal encoding
        self.temporal_encoding = TemporalEncoding(hidden_dim)

        # Frequency encoder
        self.frequency_encoder = FrequencyEncoder(hidden_dim, num_frequencies)

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Temporal convolution for local patterns
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Model temporal variability.

        Args:
            features: Input features [B, C, H, W].
            timestamps: Optional timestamp indices [B, T].

        Returns:
            Temporal variability features [B, C, H, W].
        """
        B, C, H, W = features.shape

        # Reshape for temporal processing: [B, C, H, W] -> [B, H*W, C]
        features_flat = features.view(B, C, H * W).transpose(1, 2)

        # Add temporal encoding (use position along spatial dimension)
        features_encoded = self.temporal_encoding(features_flat)

        # Apply temporal attention
        attn_out, _ = self.temporal_attention(
            features_encoded, features_encoded, features_encoded
        )

        # Apply temporal convolution
        conv_in = attn_out.transpose(1, 2)  # [B, C, H*W]
        conv_out = self.temporal_conv(conv_in).transpose(1, 2)  # [B, H*W, C]

        # Gated fusion
        gate_input = torch.cat([attn_out, conv_out], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * attn_out + (1 - gate) * conv_out

        # Layer normalization
        fused = self.norm(fused)

        # Reshape back to spatial format
        output = fused.transpose(1, 2).view(B, C, H, W)

        return output


class AdaptiveModulation(nn.Module):
    """Adaptive feature modulation module.

    This module dynamically adjusts features based on spatial and
    temporal heterogeneity information.

    Args:
        hidden_dim: Hidden dimension.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Spatial modulation network
        self.spatial_mod = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Temporal modulation network
        self.temporal_mod = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Feature transformation
        self.transform = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # Residual scaling
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        features: torch.Tensor,
        spatial_het: torch.Tensor,
        temporal_var: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive modulation.

        Args:
            features: Input features [B, C, H, W].
            spatial_het: Spatial heterogeneity features [B, C, H, W].
            temporal_var: Temporal variability features [B, C, H, W].

        Returns:
            Modulated features [B, C, H, W].
        """
        # Spatial modulation
        spatial_input = torch.cat([features, spatial_het], dim=1)
        spatial_gate = self.spatial_mod(spatial_input)
        spatial_modulated = features * spatial_gate

        # Temporal modulation
        temporal_input = torch.cat([spatial_modulated, temporal_var], dim=1)
        temporal_gate = self.temporal_mod(temporal_input)
        temporal_modulated = spatial_modulated * temporal_gate

        # Feature transformation
        transformed = self.transform(temporal_modulated)

        # Residual connection with learnable scaling
        output = features + self.residual_scale * transformed

        return output


class SpatiotemporalHeterogeneity(nn.Module):
    """Spatiotemporal heterogeneity modeling module.

    This module comprehensively models the spatiotemporal heterogeneity
    of precipitation fields, capturing both spatial non-stationarity
    and temporal variability.

    The module is designed to:
    1. Capture regional variations in precipitation patterns
    2. Model temporal dynamics at multiple scales
    3. Adaptively modulate features based on heterogeneity

    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        num_regions: Number of geographical regions.
        num_frequencies: Number of frequency components.
        dropout: Dropout rate.

    Example:
        >>> heterogeneity = SpatiotemporalHeterogeneity(hidden_dim=256)
        >>> features = torch.randn(2, 256, 128, 128)
        >>> output = heterogeneity(features)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_regions: int = 4,
        num_frequencies: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Spatial heterogeneity modeling
        self.spatial_heterogeneity = SpatialHeterogeneity(
            hidden_dim=hidden_dim,
            num_regions=num_regions,
            dropout=dropout,
        )

        # Temporal variability modeling
        self.temporal_variability = TemporalVariability(
            hidden_dim=hidden_dim,
            num_frequencies=num_frequencies,
            num_heads=num_heads // 2,
            dropout=dropout,
        )

        # Adaptive feature modulation
        self.adaptive_modulation = AdaptiveModulation(hidden_dim)

        # Feature aggregation
        self.aggregation = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Model spatiotemporal heterogeneity.

        Args:
            features: Input features [B, C, H, W].
            timestamps: Optional timestamp indices [B, T].

        Returns:
            Heterogeneity-aware features [B, C, H, W].
        """
        # Spatial heterogeneity
        spatial_het = self.spatial_heterogeneity(features)

        # Temporal variability
        temporal_var = self.temporal_variability(features, timestamps)

        # Adaptive modulation
        modulated = self.adaptive_modulation(features, spatial_het, temporal_var)

        # Aggregate all features
        aggregated = torch.cat([features, spatial_het, temporal_var], dim=1)
        aggregated = self.aggregation(aggregated)

        # Final output with residual
        output = self.output_proj(aggregated + modulated)

        return output

    def compute_heterogeneity_loss(
        self,
        features: torch.Tensor,
        region_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute heterogeneity regularization loss.

        This loss encourages diverse representations across regions
        while maintaining coherence within regions.

        Args:
            features: Features [B, C, H, W].
            region_labels: Optional region labels [B, 1, H, W].

        Returns:
            Heterogeneity loss scalar.
        """
        B, C, H, W = features.shape

        # Spatial diversity loss
        # Encourage different regions to have different feature distributions
        features_flat = features.view(B, C, -1)  # [B, C, H*W]
        features_mean = features_flat.mean(dim=-1, keepdim=True)  # [B, C, 1]
        features_var = features_flat.var(dim=-1, keepdim=True)  # [B, C, 1]

        # Minimize variance within same region, maximize between regions
        diversity_loss = -features_var.mean()

        # Smoothness loss to prevent abrupt changes
        grad_x = features[:, :, :, 1:] - features[:, :, :, :-1]
        grad_y = features[:, :, 1:, :] - features[:, :, :-1, :]
        smoothness_loss = (grad_x ** 2 + grad_y ** 2).mean()

        return diversity_loss + 0.1 * smoothness_loss
