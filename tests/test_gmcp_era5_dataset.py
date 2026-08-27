"""Tests for GMCPERA5Dataset (synthetic fixtures, no real data needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.data.gmcp_era5_dataset import (
    ERA5_CHANNELS,
    ERA5_NUM_CHANNELS,
    GMCPERA5Dataset,
)


def _make_era5_file(path: Path, start: str, periods: int) -> None:
    """Write a small synthetic ERA5 file matching the WeatherBench2 layout."""
    times = xr.date_range(start, periods=periods, freq="6h")
    lats = np.arange(18.0, 54.1, 1.5)  # ascending, China range
    lons = np.arange(73.5, 135.1, 1.5)
    levels = [1000, 850, 500, 200, 100]
    rng = np.random.default_rng(0)

    data_vars = {}
    for var in ["u_component_of_wind", "v_component_of_wind", "temperature",
                "specific_humidity", "vertical_velocity", "geopotential"]:
        data_vars[var] = (("time", "level", "longitude", "latitude"),
                          rng.normal(size=(periods, len(levels), len(lons), len(lats))).astype(np.float32))
    for var in ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
                "surface_pressure", "total_column_water_vapour",
                "total_precipitation_6hr", "total_precipitation_24hr"]:
        data_vars[var] = (("time", "longitude", "latitude"),
                          rng.normal(size=(periods, len(lons), len(lats))).astype(np.float32))

    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "level": levels,
                "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)


def _make_gmcp_file(path: Path, start: str, periods: int) -> None:
    """Write a small synthetic preprocessed GMCP 6h file (China crop)."""
    times = xr.date_range(start, periods=periods, freq="6h",
                          calendar="standard", use_cftime=False)
    # GMCP latitude is descending.
    lats = np.arange(53.95, 18.0, -0.1)[:360]
    lons = np.arange(73.05, 135.0, 0.1)[:620]
    rng = np.random.default_rng(1)
    precip = rng.exponential(scale=0.5, size=(periods, len(lats), len(lons))).astype(np.float32)

    ds = xr.Dataset(
        {"precipitation_rate": (("time", "latitude", "longitude"), precip)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)


@pytest.fixture
def fixture_dir(tmp_path: Path) -> dict:
    """Create synthetic ERA5 + GMCP fixture files."""
    era5_file = tmp_path / "era5_test.nc"
    gmcp_dir = tmp_path / "gmcp_6h"
    gmcp_dir.mkdir()
    _make_era5_file(era5_file, "2018-01-01", 40)
    _make_gmcp_file(gmcp_dir / "gmcp_6h_2018.nc", "2018-01-01", 40)
    return {
        "era5_path": str(era5_file),
        "gmcp_path": str(gmcp_dir),
        "start_date": "2018-01-01",
        "end_date": "2018-01-10",
        "input_timesteps": 4,
        "forecast_horizon": 2,
        "normalize": "log_minmax",
    }


def test_channel_layout():
    """Channel count and ordering must match the documented layout."""
    assert ERA5_NUM_CHANNELS == 17
    assert ERA5_CHANNELS[0] == "u_component_of_wind_850"
    assert ERA5_CHANNELS[11] == "specific_humidity_200"
    assert ERA5_CHANNELS[12] == "2m_temperature"
    assert ERA5_CHANNELS[-1] == "total_column_water_vapour"


def test_sample_shapes(fixture_dir):
    """A sample must return ERA5 [C,H,W], GMCP input [T_in,1,H,W], target [T,H,W]."""
    ds = GMCPERA5Dataset(fixture_dir)
    sample = ds[0]
    assert sample["era5"].shape == (17, ds.era5_lat_count, ds.era5_lon_count)
    assert sample["input"].shape[0] == 4
    assert sample["input"].shape[1] == 1
    assert sample["target"].shape[0] == 2
    # ERA5 is z-scored.
    assert abs(float(sample["era5"].mean())) < 1.0


def test_window_count(fixture_dir):
    """Number of samples = times - input_timesteps - forecast_horizon."""
    ds = GMCPERA5Dataset(fixture_dir)
    # 40 aligned times; window needs 4 history + 2 forecast.
    assert len(ds) == 40 - 4 - 2 + 1


def test_time_alignment(fixture_dir):
    """The ERA5 anchor time must equal the last GMCP history window's label."""
    ds = GMCPERA5Dataset(fixture_dir)
    ei, gi = ds.indices[0]
    era5_t = ds._era5_times[ei]
    gmcp_t = ds._gmcp_times[gi]
    assert era5_t == gmcp_t


def test_targets_are_future_windows(fixture_dir):
    """Targets must be the GMCP windows strictly after the anchor."""
    ds = GMCPERA5Dataset(fixture_dir)
    ei, gi = ds.indices[0]
    raw = ds.gmcp["precipitation_rate"].values
    # Denormalized target must equal raw future windows.
    log_min = ds.input_min
    log_max = ds.input_max
    target_norm = ds[0]["target"].numpy()
    expected = np.log1p(raw[gi + 1: gi + 3])
    expected = (expected - log_min) / max(log_max - log_min, 1e-8)
    assert np.allclose(target_norm, expected, atol=1e-4)


def test_missing_gmcp_raises(tmp_path):
    """A missing GMCP directory must raise FileNotFoundError."""
    era5_file = tmp_path / "era5.nc"
    _make_era5_file(era5_file, "2018-01-01", 10)
    with pytest.raises(FileNotFoundError):
        GMCPERA5Dataset({
            "era5_path": str(era5_file),
            "gmcp_path": str(tmp_path / "nope"),
            "start_date": "2018-01-01",
            "end_date": "2018-01-05",
        })
