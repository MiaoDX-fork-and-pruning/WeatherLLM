"""Multi-Scale Spatiotemporal Encoder for PhyDiff-Net.

This module extracts multi-scale spatiotemporal features from ERA5 and GMCP
input data, capturing both large-scale circulation backgrounds and local
precipitation details through adaptive convolution, spatiotemporal attention,
and cross-resolution fusion mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for dynamic time-step importance weighting.

    Computes attention scores across the temporal dimension to emphasize
    the most informative time steps for precipitation prediction.

    Args:
        hidden_dim: Dimension of input features.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, timestamps=None):
        """Apply temporal attention to the input tensor.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, T, C, H, W].
            timestamps: Optional timestamp embeddings of shape [B, T, D].

        Returns:
            Temporally attended features with the same shape as input.
        """
        if x.dim() == 4:
            # [B, C, H, W] -> treat as single time step
            return x

        B, T, C, H, W = x.shape
        residual = x

        # Reshape for attention: [B*T, H*W, C]
        x_reshaped = x.reshape(B * T, C, H * W).permute(0, 2, 1)

        q = self.query(x_reshaped).reshape(B * T, H * W, self.num_heads, self.head_dim)
        k = self.key(x_reshaped).reshape(B * T, H * W, self.num_heads, self.head_dim)
        v = self.value(x_reshaped).reshape(B * T, H * W, self.num_heads, self.head_dim)

        # Add positional information from timestamps if available
        if timestamps is not None:
            time_emb = timestamps.reshape(B * T, 1, self.num_heads, self.head_dim)
            q = q + time_emb
            k = k + time_emb

        # Compute attention: [B*T, H*W, H*W, num_heads]
        q = q.permute(0, 2, 1, 3)  # [B*T, num_heads, H*W, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # [B*T, num_heads, H*W, head_dim]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B * T, H * W, C)

        out = self.out_proj(attn_out)
        out = out.reshape(B, T, C, H, W)
        out = self.layer_norm(out + residual)

        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for adaptive spatial feature weighting.

    Applies channel-wise attention across spatial locations to focus on
    regions most relevant for precipitation forecasting.

    Args:
        hidden_dim: Number of input/output channels.
        reduction: Channel reduction ratio for efficiency.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim, reduction=16, dropout=0.1):
        super().__init__()
        reduced_dim = max(hidden_dim // reduction, 16)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(hidden_dim, reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply spatial attention to the input tensor.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Spatially attended features of shape [B, C, H, W].
        """
        residual = x

        # Channel attention
        channel_weights = self.channel_attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_weights

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        x = x * spatial_weights

        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


class CrossResolutionFusion(nn.Module):
    """Cross-resolution fusion module for ERA5 and GMCP data.

    Explicitly models the resolution difference between ERA5 (0.25 degrees)
    and GMCP (0.1 degrees) data through learnable upsampling and cross-attention
    to produce a unified high-resolution feature representation.

    Args:
        era5_channels: Number of ERA5 input channels.
        gmcp_channels: Number of GMCP input channels.
        hidden_dim: Output feature dimension.
        era5_resolution: ERA5 spatial resolution in degrees.
        gmcp_resolution: GMCP spatial resolution in degrees.
        num_heads: Number of attention heads for cross-attention.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        era5_channels,
        gmcp_channels,
        hidden_dim,
        era5_resolution=0.25,
        gmcp_resolution=0.1,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.era5_resolution = era5_resolution
        self.gmcp_resolution = gmcp_resolution
        self.scale_factor = era5_resolution / gmcp_resolution

        # ERA5 upsampling path
        self.era5_proj = nn.Conv2d(era5_channels, hidden_dim, kernel_size=1)
        self.era5_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # GMCP projection path
        self.gmcp_proj = nn.Sequential(
            nn.Conv2d(gmcp_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, era5_features, gmcp_data):
        """Fuse ERA5 and GMCP features across resolutions.

        Args:
            era5_features: ERA5 features of shape [B, C_era5, H_low, W_low].
            gmcp_data: GMCP data of shape [B, C_gmcp, H, W].

        Returns:
            Fused features of shape [B, hidden_dim, H, W].
        """
        B, _, H, W = gmcp_data.shape

        # Project and upsample ERA5 features to GMCP resolution
        era5_proj = self.era5_proj(era5_features)
        era5_upsampled = self.era5_upsample(era5_proj)

        # Resize to exact target dimensions if needed
        if era5_upsampled.shape[2:] != (H, W):
            era5_upsampled = F.interpolate(
                era5_upsampled, size=(H, W), mode="bilinear", align_corners=False
            )

        # Project GMCP data
        gmcp_proj = self.gmcp_proj(gmcp_data)

        # Cross-attention: ERA5 queries attend to GMCP keys/values
        era5_flat = era5_upsampled.flatten(2).permute(0, 2, 1)  # [B, H*W, D]
        gmcp_flat = gmcp_proj.flatten(2).permute(0, 2, 1)      # [B, H*W, D]

        cross_out, _ = self.cross_attn(
            query=era5_flat, key=gmcp_flat, value=gmcp_flat
        )
        cross_out = cross_out.permute(0, 2, 1).reshape(B, -1, H, W)

        # Concatenate and fuse
        fused = torch.cat([cross_out, gmcp_proj], dim=1)
        output = self.fusion(fused)

        return output


class MultiScaleEncoder(nn.Module):
    """Multi-Scale Spatiotemporal Encoder for PhyDiff-Net.

    Extracts multi-scale features from ERA5 and GMCP inputs using adaptive
    convolution kernels (3x3, 5x5, 7x7), spatiotemporal attention, and
    cross-resolution fusion to capture both large-scale circulation and
    local precipitation details.

    Args:
        in_channels: Number of ERA5 input channels (default: 10).
        gmcp_channels: Number of GMCP input channels (default: 1).
        hidden_dim: Hidden feature dimension (default: 256).
        num_scales: Number of multi-scale convolution branches (default: 3).
        num_heads: Number of attention heads (default: 8).
        dropout: Dropout probability (default: 0.1).
    """

    KERNEL_SIZES = [3, 5, 7]

    def __init__(
        self,
        in_channels=10,
        gmcp_channels=1,
        hidden_dim=256,
        num_scales=3,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.num_scales = min(num_scales, len(self.KERNEL_SIZES))
        branch_dim = hidden_dim // self.num_scales

        # Multi-scale convolution branches with different kernel sizes
        self.scale_branches = nn.ModuleList()
        for i in range(self.num_scales):
            kernel_size = self.KERNEL_SIZES[i]
            padding = kernel_size // 2
            self.scale_branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, branch_dim,
                        kernel_size=kernel_size, stride=1, padding=padding,
                    ),
                    nn.GroupNorm(8, branch_dim),
                    nn.SiLU(),
                    nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, branch_dim),
                    nn.SiLU(),
                )
            )

        # Projection to match hidden_dim after concatenation
        self.scale_proj = nn.Sequential(
            nn.Conv2d(branch_dim * self.num_scales, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Spatiotemporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout
        )
        self.spatial_attention = SpatialAttention(
            hidden_dim, dropout=dropout
        )

        # Cross-resolution fusion
        self.cross_resolution_fusion = CrossResolutionFusion(
            era5_channels=hidden_dim,
            gmcp_channels=gmcp_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, era5_data, gmcp_data, timestamps=None):
        """Extract multi-scale spatiotemporal features.

        Args:
            era5_data: ERA5 input tensor of shape [B, C_era5, H, W].
            gmcp_data: GMCP input tensor of shape [B, C_gmcp, H_high, W_high].
            timestamps: Optional timestamp embeddings of shape [B, T, D].

        Returns:
            Multi-scale fused features of shape [B, hidden_dim, H_high, W_high].
        """
        # Multi-scale feature extraction
        multi_scale_features = []
        for branch in self.scale_branches:
            scale_feat = branch(era5_data)
            multi_scale_features.append(scale_feat)

        # Concatenate and project multi-scale features
        concat_features = torch.cat(multi_scale_features, dim=1)
        features = self.scale_proj(concat_features)

        # Spatiotemporal attention
        features = self.temporal_attention(features, timestamps)
        features = self.spatial_attention(features)

        # Cross-resolution fusion with GMCP data
        fused_features = self.cross_resolution_fusion(features, gmcp_data)
        fused_features = self.dropout(fused_features)

        return fused_features
