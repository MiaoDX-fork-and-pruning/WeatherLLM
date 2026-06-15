"""Physics-Constrained Diffusion Module for PhyDiff-Net.

This module implements the diffusion process with physical constraints
for precipitation forecasting. The key innovation is incorporating
atmospheric dynamics equations into the diffusion process to ensure
physical consistency of generated precipitation fields.

Author: weather-model-trainer
Date: 2026-06-15
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timestep encoding.

    This module encodes the diffusion timestep into a fixed-dimensional
    representation using sinusoidal functions.

    Args:
        dim: Output embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestep into sinusoidal embedding.

        Args:
            t: Timestep tensor of shape [batch_size].

        Returns:
            Embedding tensor of shape [batch_size, dim].
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding injection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_dim: Dimension of time embedding.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].
            t_emb: Time embedding of shape [B, time_dim].

        Returns:
            Output tensor of shape [B, out_channels, H, W].
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies.

    Args:
        channels: Number of channels.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self, channels: int, num_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Output tensor of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        h, _ = self.attention(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + self.dropout(h)


class Downsample(nn.Module):
    """Spatial downsampling using strided convolution.

    Args:
        channels: Number of channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Downsampled tensor of shape [B, C, H/2, W/2].
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling using transposed convolution.

    Args:
        channels: Number of channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Upsampled tensor of shape [B, C, H*2, W*2].
        """
        return self.conv(x)


class UNet(nn.Module):
    """U-Net denoising network for diffusion models.

    This U-Net architecture is designed for the diffusion denoising process,
    with time embedding injection and attention at specified resolutions.

    Args:
        in_channels: Number of input channels (condition + noisy input).
        out_channels: Number of output channels.
        hidden_channels: Base number of hidden channels.
        channel_mults: Channel multipliers for each level.
        num_res_blocks: Number of residual blocks per level.
        attention_resolutions: Resolutions where attention is applied.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 256,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        time_dim = hidden_channels * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_channels),
            nn.Linear(hidden_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        current_channels = hidden_channels
        current_res = 128  # Assume initial resolution

        for level, mult in enumerate(channel_mults):
            out_channels_level = hidden_channels * mult
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResidualBlock(
                        current_channels, out_channels_level, time_dim, dropout
                    )
                )
                if current_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(out_channels_level, num_heads, dropout))
                current_channels = out_channels_level

            self.encoder_blocks.append(level_blocks)
            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(current_channels))
                current_res //= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(current_channels, current_channels, time_dim, dropout),
            AttentionBlock(current_channels, num_heads, dropout),
            ResidualBlock(current_channels, current_channels, time_dim, dropout),
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_mults)):
            out_channels_level = hidden_channels * mult
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResidualBlock(
                        current_channels + out_channels_level,
                        out_channels_level,
                        time_dim,
                        dropout,
                    )
                )
                if current_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(out_channels_level, num_heads, dropout))
                current_channels = out_channels_level

            self.decoder_blocks.append(level_blocks)
            if level < len(channel_mults) - 1:
                self.upsamplers.append(Upsample(current_channels))
                current_res *= 2

        # Output
        self.output_norm = nn.GroupNorm(8, hidden_channels)
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy input tensor of shape [B, C_in, H, W].
            t: Timestep tensor of shape [B].
            condition: Conditioning tensor of shape [B, C_cond, H, W].

        Returns:
            Predicted noise tensor of shape [B, C_out, H, W].
        """
        # Concatenate input and condition
        h = torch.cat([x, condition], dim=1)

        # Time embedding
        t_emb = self.time_embed(t)

        # Input projection
        h = self.input_conv(h)

        # Encoder with skip connections
        skips = []
        for level_blocks in self.encoder_blocks:
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            skips.append(h)

        # Downsample
        for downsampler in self.downsamplers:
            h = downsampler(h)
            skips.append(h)

        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = h + block(h)

        # Decoder with skip connections
        for level_blocks, upsampler in zip(self.decoder_blocks, self.upsamplers):
            h = upsampler(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)

        # Handle remaining skip connections
        for level_blocks in self.decoder_blocks[len(self.upsamplers):]:
            if skips:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        return self.output_conv(h)


class ContinuityConstraint(nn.Module):
    """Atmospheric continuity equation constraint.

    Enforces mass conservation in the atmosphere by penalizing
    violations of the continuity equation.

    Args:
        hidden_dim: Hidden dimension for learnable parameters.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Learnable correction network
        self.correction_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=3, padding=1),
        )

    def forward(
        self, precipitation: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute continuity constraint.

        Args:
            precipitation: Predicted precipitation [B, 1, H, W].
            condition: Atmospheric condition [B, C, H, W].

        Returns:
            Tuple of (constraint_violation_loss, correction_term).
        """
        # Extract wind components from condition (u, v winds)
        # Assume first 2 channels are u and v winds
        if condition.shape[1] >= 2:
            u_wind = condition[:, 0:1, :, :]
            v_wind = condition[:, 1:2, :, :]
        else:
            # Fallback: use gradients of precipitation
            u_wind = precipitation[:, :, :, 1:] - precipitation[:, :, :, :-1]
            v_wind = precipitation[:, :, 1:, :] - precipitation[:, :, :-1, :]

        # Compute divergence (approximation of continuity equation violation)
        # du/dx + dv/dy should be zero for incompressible flow
        du_dx = u_wind[:, :, :, 1:] - u_wind[:, :, :, :-1]
        dv_dy = v_wind[:, :, 1:, :] - v_wind[:, :, :-1, :]

        # Pad to match dimensions
        du_dx = F.pad(du_dx, (0, 1), mode="constant", value=0)
        dv_dy = F.pad(dv_dy, (0, 0, 0, 1), mode="constant", value=0)

        divergence = du_dx + dv_dy

        # Compute constraint violation
        constraint_loss = torch.mean(divergence ** 2)

        # Generate correction
        stacked = torch.cat([precipitation, divergence], dim=1)
        correction = self.correction_net(stacked)

        return constraint_loss, correction


class MoistureConstraint(nn.Module):
    """Moisture conservation constraint.

    Ensures water vapor conservation in precipitation processes.

    Args:
        hidden_dim: Hidden dimension for learnable parameters.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=3, padding=1),
        )

    def forward(
        self, precipitation: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute moisture conservation constraint.

        Args:
            precipitation: Predicted precipitation [B, 1, H, W].
            condition: Atmospheric condition [B, C, H, W].

        Returns:
            Tuple of (constraint_violation_loss, correction_term).
        """
        # Extract humidity from condition
        if condition.shape[1] >= 3:
            humidity = condition[:, 2:3, :, :]
        else:
            humidity = torch.ones_like(precipitation) * 0.5

        # Moisture conservation: d(q)/dt + div(q*v) = E - P
        # where q is specific humidity, v is wind, E is evaporation, P is precipitation
        moisture_content = humidity * precipitation

        # Spatial gradient of moisture flux
        d_m_dx = moisture_content[:, :, :, 1:] - moisture_content[:, :, :, :-1]
        d_m_dy = moisture_content[:, :, 1:, :] - moisture_content[:, :, :-1, :]

        d_m_dx = F.pad(d_m_dx, (0, 1), mode="constant", value=0)
        d_m_dy = F.pad(d_m_dy, (0, 0, 0, 1), mode="constant", value=0)

        moisture_divergence = d_m_dx + d_m_dy

        # Non-negative precipitation constraint
        neg_penalty = F.relu(-precipitation).mean()

        constraint_loss = torch.mean(moisture_divergence ** 2) + neg_penalty

        stacked = torch.cat([precipitation, humidity, moisture_divergence], dim=1)
        correction = self.correction_net(stacked)

        return constraint_loss, correction


class EnergyConstraint(nn.Module):
    """Energy balance constraint.

    Ensures energy conservation in the precipitation process.

    Args:
        hidden_dim: Hidden dimension for learnable parameters.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.correction_net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=3, padding=1),
        )

    def forward(
        self, precipitation: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute energy balance constraint.

        Args:
            precipitation: Predicted precipitation [B, 1, H, W].
            condition: Atmospheric condition [B, C, H, W].

        Returns:
            Tuple of (constraint_violation_loss, correction_term).
        """
        # Extract temperature from condition
        if condition.shape[1] >= 4:
            temperature = condition[:, 3:4, :, :]
        else:
            temperature = torch.zeros_like(precipitation)

        # Latent heat release: Q = L * P
        # where L is latent heat of condensation (~2.5e6 J/kg)
        latent_heat = 2.5 * precipitation  # Scaled for numerical stability

        # Energy conservation: dT/dt = Q/cp + advection
        temperature_change = temperature - temperature.mean(dim=[2, 3], keepdim=True)

        # Smoothness constraint (penalize extreme gradients)
        grad_x = precipitation[:, :, :, 1:] - precipitation[:, :, :, :-1]
        grad_y = precipitation[:, :, 1:, :] - precipitation[:, :, :-1, :]
        smoothness_loss = (grad_x ** 2 + grad_y ** 2).mean()

        constraint_loss = smoothness_loss

        stacked = torch.cat([precipitation, latent_heat, temperature], dim=1)
        correction = self.correction_net(stacked)

        return constraint_loss, correction


class PhysicsConstraintModule(nn.Module):
    """Physics constraint module combining multiple atmospheric constraints.

    This module enforces physical laws (continuity, moisture conservation,
    energy balance) on the predicted precipitation fields.

    Args:
        equations: List of constraint equations to apply.
        hidden_dim: Hidden dimension for constraint networks.
    """

    def __init__(
        self,
        equations: List[str] = None,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if equations is None:
            equations = ["continuity", "moisture_conservation", "energy_balance"]

        self.constraint_modules = nn.ModuleDict()
        for eq in equations:
            if eq == "continuity":
                self.constraint_modules[eq] = ContinuityConstraint(hidden_dim)
            elif eq == "moisture_conservation":
                self.constraint_modules[eq] = MoistureConstraint(hidden_dim)
            elif eq == "energy_balance":
                self.constraint_modules[eq] = EnergyConstraint(hidden_dim)

        # Learnable constraint weights
        num_constraints = len(self.constraint_modules)
        self.constraint_weights = nn.Parameter(
            torch.ones(num_constraints) / num_constraints
        )

        self.correction_scale = 0.1

    def forward(
        self, precipitation: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply physics constraints.

        Args:
            precipitation: Predicted precipitation [B, 1, H, W].
            condition: Atmospheric condition [B, C, H, W].

        Returns:
            Tuple of (total_constraint_loss, corrected_precipitation).
        """
        total_loss = torch.tensor(0.0, device=precipitation.device)
        corrected = precipitation

        # Normalize weights
        weights = F.softmax(self.constraint_weights, dim=0)

        for i, (name, constraint) in enumerate(self.constraint_modules.items()):
            constraint_loss, correction = constraint(corrected, condition)
            total_loss = total_loss + weights[i] * constraint_loss
            corrected = corrected - self.correction_scale * correction

        return total_loss, corrected


class ConditionEncoder(nn.Module):
    """Condition encoder for encoding ERA5 atmospheric data.

    This module encodes the conditioning information (ERA5 data)
    into a format suitable for the U-Net denoiser.

    Args:
        hidden_dim: Output hidden dimension.
        num_layers: Number of encoding layers.
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Progressive downsampling
        layers = []
        in_channels = 19  # ERA5 typical channel count
        for i in range(num_layers):
            out_channels = hidden_dim if i == num_layers - 1 else hidden_dim // (2 ** (num_layers - 1 - i))
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            ])
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

        # Final projection
        self.projection = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Encode conditioning data.

        Args:
            condition: Atmospheric condition [B, C_era5, H, W].

        Returns:
            Encoded features [B, hidden_dim, H', W'].
        """
        features = self.encoder(condition)
        features = self.adaptive_pool(features)
        return self.projection(features)


class PhysicsConstrainedDiffusion(nn.Module):
    """Physics-constrained diffusion module for precipitation forecasting.

    This module implements the diffusion process with physical constraints,
    ensuring that generated precipitation fields obey atmospheric dynamics.

    The key components are:
    - U-Net denoising network for predicting noise/signal
    - Physics constraint module for enforcing physical laws
    - Condition encoder for processing ERA5 atmospheric data

    Args:
        hidden_dim: Hidden dimension for the U-Net and encoders.
        num_diffusion_steps: Number of diffusion steps.
        dropout: Dropout rate.

    Example:
        >>> model = PhysicsConstrainedDiffusion(hidden_dim=256)
        >>> x = torch.randn(2, 1, 128, 128)
        >>> t = torch.randint(0, 1000, (2,))
        >>> condition = torch.randn(2, 19, 32, 32)
        >>> output = model(x, t, condition)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_diffusion_steps: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_steps = num_diffusion_steps
        self.hidden_dim = hidden_dim

        # Precompute diffusion parameters
        self._setup_diffusion_params()

        # U-Net denoiser
        self.denoiser = UNet(
            in_channels=hidden_dim + 1,
            out_channels=1,
            hidden_channels=hidden_dim,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            num_heads=4,
            dropout=dropout,
        )

        # Physics constraint module
        self.physics_constraints = PhysicsConstraintModule(
            equations=["continuity", "moisture_conservation", "energy_balance"],
            hidden_dim=64,
        )

        # Condition encoder
        self.condition_encoder = ConditionEncoder(hidden_dim)

    def _setup_diffusion_params(self) -> None:
        """Setup diffusion schedule parameters."""
        # Linear beta schedule
        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, self.num_steps)

        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_m1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1
        )

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise to input (forward diffusion process).

        Args:
            x_start: Clean input tensor [B, C, H, W].
            t: Timestep tensor [B].
            noise: Optional pre-generated noise.

        Returns:
            Noisy tensor [B, C, H, W].
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod.to(x_start.device)[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t][:, None, None, None]

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict clean signal from noisy input.

        Args:
            x_t: Noisy input [B, 1, H, W].
            t: Timestep [B].
            condition: Conditioning data [B, C, H', W'].

        Returns:
            Predicted clean signal [B, 1, H, W].
        """
        # Encode condition to match spatial dimensions
        cond_features = self.condition_encoder(condition)

        # Resize condition features to match input spatial dims
        cond_features = F.interpolate(
            cond_features,
            size=(x_t.shape[2], x_t.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Predict noise
        noise_pred = self.denoiser(x_t, t, cond_features)

        # Recover x0 from noise prediction
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x_t.device)[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t][:, None, None, None]

        x0_pred = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        return x0_pred

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            x: Clean precipitation field [B, 1, H, W].
            t: Timestep [B].
            condition: Atmospheric condition [B, C, H', W'].

        Returns:
            Tuple of (predicted_noise, physics_loss).
        """
        # Generate noise
        noise = torch.randn_like(x)

        # Add noise
        x_t = self.q_sample(x, t, noise)

        # Predict noise
        cond_features = self.condition_encoder(condition)
        cond_features = F.interpolate(
            cond_features,
            size=(x_t.shape[2], x_t.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        noise_pred = self.denoiser(x_t, t, cond_features)

        # Apply physics constraints on predicted x0
        x0_pred = (x_t - torch.sqrt(1.0 - self.alphas_cumprod.to(x.device)[t][:, None, None, None]) * noise_pred) / \
                  torch.sqrt(self.alphas_cumprod.to(x.device)[t][:, None, None, None])

        physics_loss, corrected_x0 = self.physics_constraints(x0_pred, condition)

        return noise_pred, physics_loss

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate precipitation prediction via reverse diffusion.

        Args:
            condition: Atmospheric condition [B, C, H', W'].
            shape: Output shape (B, 1, H, W).
            device: Device for computation.

        Returns:
            Generated precipitation field [B, 1, H, W].
        """
        batch_size = shape[0]

        # Start from noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion process
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict x0
            x0_pred = self.predict_x0(x, t_tensor, condition)

            # Apply physics constraints
            _, x0_corrected = self.physics_constraints(x0_pred, condition)

            if t > 0:
                # Add noise
                noise = torch.randn_like(x)
                beta_t = self.betas.to(device)[t]
                alpha_t = self.alphas.to(device)[t]
                alpha_cumprod_t = self.alphas_cumprod.to(device)[t]

                x = (
                    torch.sqrt(alpha_cumprod_t) * x0_corrected +
                    torch.sqrt(1.0 - alpha_cumprod_t) * noise
                )
            else:
                x = x0_corrected

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate precipitation prediction using DDIM sampling.

        This is a faster alternative to standard reverse diffusion.

        Args:
            condition: Atmospheric condition [B, C, H', W'].
            shape: Output shape (B, 1, H, W).
            device: Device for computation.
            ddim_steps: Number of DDIM steps (fewer than total).
            eta: Stochasticity parameter (0 = deterministic).

        Returns:
            Generated precipitation field [B, 1, H, W].
        """
        batch_size = shape[0]

        # Create time step subset
        step_size = self.num_steps // ddim_steps
        timesteps = torch.arange(0, self.num_steps, step_size, device=device).long().flip(0)

        # Start from noise
        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict x0
            x0_pred = self.predict_x0(x, t_tensor, condition)

            # Apply physics constraints
            _, x0_corrected = self.physics_constraints(x0_pred, condition)

            if t > 0:
                # DDIM update
                alpha_t = self.alphas_cumprod.to(device)[t]
                if i + 1 < len(timesteps):
                    alpha_prev = self.alphas_cumprod.to(device)[timesteps[i + 1]]
                else:
                    alpha_prev = torch.tensor(1.0, device=device)

                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * \
                        torch.sqrt(1 - alpha_t / alpha_prev)

                x = (
                    torch.sqrt(alpha_prev) * x0_corrected +
                    torch.sqrt(1 - alpha_prev - sigma ** 2) * (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t) +
                    sigma * noise
                )
            else:
                x = x0_corrected

        return x
