"""Unit tests for PhyDiff-Net GMCP-only mode."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "e:/weather")

import torch

from src.models.phydiff_net import PhyDiffNet


def _get_gmcp_only_config(
    input_timesteps: int = 4,
    forecast_horizon: int = 4,
    hidden_dim: int = 64,
) -> dict:
    return {
        "use_era5": False,
        "hidden_dim": hidden_dim,
        "encoder_spatial_size": [64, 64],
        "encoder": {
            "in_channels": 19,
            "gmcp_channels": input_timesteps,
            "hidden_dim": hidden_dim,
            "num_scales": 2,
            "dropout": 0.0,
        },
        "diffusion": {
            "hidden_dim": hidden_dim,
            "num_steps": 10,
            "dropout": 0.0,
        },
        "extreme_branch": {
            "hidden_dim": hidden_dim,
            "num_layers": 2,
            "dropout": 0.0,
        },
        "heterogeneity": {
            "hidden_dim": hidden_dim,
            "num_heads": 4,
            "num_regions": 4,
            "num_frequencies": 4,
            "dropout": 0.0,
        },
        "output": {
            "hidden_dim": hidden_dim,
            "output_channels": 1,
            "forecast_horizon": forecast_horizon,
            "dropout": 0.0,
        },
    }


def test_phydiff_net_gmcp_only_instantiation() -> None:
    """PhyDiffNet should instantiate in GMCP-only mode."""
    config = _get_gmcp_only_config()
    model = PhyDiffNet(config)
    assert not model.use_era5


def test_phydiff_net_gmcp_only_forward() -> None:
    """PhyDiffNet should produce correct output shape in GMCP-only mode."""
    config = _get_gmcp_only_config(input_timesteps=4, forecast_horizon=4)
    model = PhyDiffNet(config)

    batch_size = 2
    height, width = 128, 128
    gmcp_input = torch.randn(batch_size, 4, height, width)

    output = model(gmcp_data=gmcp_input)
    assert "precipitation" in output
    assert output["precipitation"].shape == (batch_size, 4, height, width)


def test_phydiff_net_era5_mode_unchanged() -> None:
    """PhyDiffNet should still support ERA5+GMCP mode (backward compatible)."""
    config = _get_gmcp_only_config()
    config["use_era5"] = True
    model = PhyDiffNet(config)
    assert model.use_era5

    batch_size = 2
    era5 = torch.randn(batch_size, 19, 16, 16)
    gmcp = torch.randn(batch_size, 4, 32, 32)
    output = model(era5, gmcp)
    assert output["precipitation"].shape[0] == batch_size


def test_phydiff_net_gmcp_only_backward() -> None:
    """Backward pass should work in GMCP-only mode."""
    config = _get_gmcp_only_config(input_timesteps=4, forecast_horizon=4)
    model = PhyDiffNet(config)

    gmcp_input = torch.randn(1, 4, 128, 128)
    targets = torch.randn(1, 4, 128, 128)

    output = model(gmcp_data=gmcp_input)
    loss = torch.nn.functional.mse_loss(output["precipitation"], targets)
    loss.backward()

    assert loss.item() >= 0.0


def test_phydiff_net_missing_gmcp_raises() -> None:
    """Forward pass should raise when gmcp_data is missing."""
    config = _get_gmcp_only_config()
    model = PhyDiffNet(config)
    try:
        model()
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_phydiff_net_gmcp_only_sample() -> None:
    """Diffusion sampling should work in GMCP-only mode."""
    config = _get_gmcp_only_config(input_timesteps=4, forecast_horizon=4)
    model = PhyDiffNet(config)
    model.eval()

    gmcp_input = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        output = model.sample(gmcp_data=gmcp_input, use_ddim=False)
    assert "precipitation" in output
    assert output["precipitation"].shape == (1, 1, 128, 128)
