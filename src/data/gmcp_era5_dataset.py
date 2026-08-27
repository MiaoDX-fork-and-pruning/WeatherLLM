"""Joint ERA5 + GMCP dual-source dataset for PhyDiff-Net training.

Pairs ERA5 atmospheric states (WeatherBench2, 1.5°, 6h) with GMCP
high-resolution 6-hourly cumulative precipitation to train the full
dual-source PhyDiff-Net architecture.

Alignment conventions:
    - GMCP 6h windows are right-labeled: time=t means precipitation
      accumulated over (t-6h, t].
    - ERA5 states are instantaneous at time t.
    - A sample uses the ERA5 state at time t plus the 12 GMCP windows
      ending at t to predict the next 4 GMCP windows (24h ahead).

ERA5 channels (17 total):
    - 4 pressure-level vars (u, v, temperature, specific_humidity) at
      850/500/200 hPa = 12 channels.
    - 5 surface vars (2m_temperature, 10m_u, 10m_v, surface_pressure,
      total_column_water_vapour) = 5 channels.

Both sources are cropped to the China region. ERA5 is z-score normalized
per channel; GMCP follows the dataset's log_minmax convention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from src.data.gmcp_reader import CHINA_REGION

logger = logging.getLogger(__name__)

# Pressure-level variables and levels used as ERA5 channels.
ERA5_PRESSURE_VARS = ["u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity"]
ERA5_PRESSURE_LEVELS = [850, 500, 200]

# Surface variables appended after the pressure-level channels.
ERA5_SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_column_water_vapour",
]

ERA5_CHANNELS: List[str] = [
    f"{var}_{level}"
    for var in ERA5_PRESSURE_VARS
    for level in ERA5_PRESSURE_LEVELS
] + ERA5_SURFACE_VARS

ERA5_NUM_CHANNELS = len(ERA5_CHANNELS)  # 17


class GMCPERA5Dataset(Dataset):
    """Dual-source dataset pairing ERA5 states with GMCP precipitation.

    Args:
        config: Configuration dictionary containing:
            - era5_path: Path to the ERA5 NetCDF file (WeatherBench2 format).
            - gmcp_path: Path to preprocessed 6-hourly GMCP NetCDF directory
              (containing gmcp_6h_YYYY.nc files).
            - start_date / end_date: Period bounds 'YYYY-MM-DD'.
            - input_timesteps: Number of GMCP history windows (default 12).
            - forecast_horizon: Number of GMCP forecast windows (default 4).
            - normalize: GMCP normalization method ("log_minmax" or None).
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.era5_path = Path(config["era5_path"])
        self.gmcp_path = Path(config["gmcp_path"])
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.input_timesteps = config.get("input_timesteps", 12)
        self.forecast_horizon = config.get("forecast_horizon", 4)
        self.normalize = config.get("normalize", "log_minmax")

        self.era5 = self._load_era5()
        self.gmcp = self._load_gmcp()
        self._align_times()
        self._compute_gmcp_normalization_stats()

        logger.info(
            "GMCPERA5Dataset: %d samples, ERA5 %dx%d x %d ch, GMCP %dx%d",
            len(self),
            self.era5_lat_count,
            self.era5_lon_count,
            ERA5_NUM_CHANNELS,
            len(self.gmcp.latitude),
            len(self.gmcp.longitude),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_era5(self) -> xr.Dataset:
        """Load ERA5, crop to China, and stack the 17 channels.

        ``era5_path`` may be a single NetCDF file or a directory containing
        yearly files named ``era5_YYYY.nc`` (they are concatenated along
        time before cropping, so only the China-region slice is computed).
        """
        if not self.era5_path.exists():
            raise FileNotFoundError(f"ERA5 path not found: {self.era5_path}")

        if self.era5_path.is_dir():
            files = sorted(self.era5_path.glob("era5_*.nc"))
            if not files:
                raise FileNotFoundError(
                    f"No era5_*.nc files found in {self.era5_path}"
                )
            if len(files) == 1:
                ds = xr.open_dataset(files[0])
            else:
                ds = xr.concat([xr.open_dataset(f) for f in files], dim="time")
        else:
            ds = xr.open_dataset(self.era5_path)

        # Crop to the China region. ERA5 latitude is ascending (-90 -> 90).
        ds = ds.sel(
            latitude=slice(CHINA_REGION["lat_min"], CHINA_REGION["lat_max"]),
            longitude=slice(CHINA_REGION["lon_min"], CHINA_REGION["lon_max"]),
        )

        # Time filter.
        ds = ds.sel(time=slice(self.start_date, self.end_date))
        if len(ds.time) == 0:
            raise ValueError(
                f"No ERA5 times in range {self.start_date}..{self.end_date}"
            )

        # Select requested pressure levels.
        ds = ds.sel(level=ERA5_PRESSURE_LEVELS)

        self.era5_lat_count = len(ds.latitude)
        self.era5_lon_count = len(ds.longitude)

        # Stack channels: pressure vars first (var-major, level-minor), then surface.
        # Drop the 'level' coordinate so concat with surface vars (no level) works.
        channels = []
        for var in ERA5_PRESSURE_VARS:
            for level in ERA5_PRESSURE_LEVELS:
                channels.append(ds[var].sel(level=level).drop_vars("level"))
        for var in ERA5_SURFACE_VARS:
            channels.append(ds[var])

        stacked = xr.concat(channels, dim="channel")
        stacked = stacked.assign_coords(channel=ERA5_CHANNELS)

        # Transpose to [time, channel, lat, lon] and load into memory.
        arr = stacked.transpose("time", "channel", "latitude", "longitude")
        arr = arr.compute()

        # Z-score normalize per channel using the loaded period.
        arr = (arr - arr.mean(dim="time")) / (arr.std(dim="time") + 1e-8)
        self._era5_channels = ERA5_CHANNELS
        return arr.to_dataset(name="era5_state")

    def _load_gmcp(self) -> xr.Dataset:
        """Load preprocessed GMCP 6h files for the period."""
        start_year = int(self.start_date[:4])
        end_year = int(self.end_date[:4])

        files = [
            self.gmcp_path / f"gmcp_6h_{year}.nc"
            for year in range(start_year, end_year + 1)
            if (self.gmcp_path / f"gmcp_6h_{year}.nc").exists()
        ]
        if not files:
            raise FileNotFoundError(
                f"No preprocessed GMCP files found in {self.gmcp_path} "
                f"for {self.start_date}..{self.end_date}"
            )

        if len(files) == 1:
            ds = xr.open_dataset(files[0])
        else:
            ds = xr.concat([xr.open_dataset(f) for f in files], dim="time")
        ds = ds.sel(time=slice(self.start_date, self.end_date))
        return ds

    def _align_times(self) -> None:
        """Intersect ERA5 and GMCP time coordinates for windowing."""
        era5_times = set(self.era5.time.values)
        gmcp_times = self.gmcp.time.values

        usable = [
            t for t in gmcp_times
            if t in era5_times  # anchor: GMCP window right edge == ERA5 state time
        ]
        # A window needs input_timesteps GMCP windows ending at the anchor
        # and forecast_horizon windows after it, all present in GMCP.
        gmcp_index = {t: i for i, t in enumerate(gmcp_times)}
        era5_index = {t: i for i, t in enumerate(self.era5.time.values)}

        self.indices: List[Tuple[int, int]] = []  # (era5_idx, gmcp_idx)
        for t in usable:
            gi = gmcp_index[t]
            ei = era5_index[t]
            if gi - self.input_timesteps + 1 < 0:
                continue
            if gi + self.forecast_horizon >= len(gmcp_times):
                continue
            self.indices.append((ei, gi))

        self._gmcp_times = gmcp_times
        self._era5_times = self.era5.time.values

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _compute_gmcp_normalization_stats(self) -> None:
        """Estimate log_minmax bounds for GMCP from a small sample."""
        if self.normalize is None:
            self.input_min = None
            self.input_max = None
            return

        n = len(self.gmcp.time)
        sample_idx = np.linspace(0, n - 1, min(20, n), dtype=int)
        sample = self.gmcp["precipitation_rate"].isel(time=sample_idx).values
        log_sample = np.log1p(np.nan_to_num(sample))
        self.input_min = float(np.nanmin(log_sample))
        self.input_max = float(np.nanmax(log_sample))

    def _normalize_gmcp(self, x: np.ndarray) -> np.ndarray:
        if self.normalize is None:
            return np.nan_to_num(x, nan=0.0)
        log_x = np.log1p(np.nan_to_num(x, nan=0.0))
        return (log_x - self.input_min) / max(self.input_max - self.input_min, 1e-8)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        era5_idx, gmcp_idx = self.indices[idx]

        # ERA5 state at the anchor time: [C, H_era5, W_era5].
        era5_arr = self.era5["era5_state"].isel(time=era5_idx).values.astype(np.float32)

        # GMCP history windows ending at the anchor: [T_in, H, W].
        hist = self.gmcp["precipitation_rate"].isel(
            time=slice(gmcp_idx - self.input_timesteps + 1, gmcp_idx + 1)
        ).values.astype(np.float32)
        # Target windows after the anchor: [T_out, H, W].
        tgt = self.gmcp["precipitation_rate"].isel(
            time=slice(gmcp_idx + 1, gmcp_idx + 1 + self.forecast_horizon)
        ).values.astype(np.float32)

        hist_n = self._normalize_gmcp(hist)
        tgt_n = self._normalize_gmcp(tgt)

        # GMCP input with channel dim: [T_in, 1, H, W] (matches GMCP-only mode).
        hist_n = hist_n[:, None, :, :]

        return {
            "era5": torch.from_numpy(era5_arr.copy()),
            "input": torch.from_numpy(hist_n.copy()),
            "target": torch.from_numpy(tgt_n.copy()),
        }


def create_gmcp_era5_dataloaders(
    config: Dict,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train/val/test DataLoaders for the dual-source dataset.

    Args:
        config: Data configuration with era5_path, gmcp_path and splits.
        batch_size: Batch size for all loaders.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    base = {
        "era5_path": config["era5_path"],
        "gmcp_path": config["gmcp_path"],
        "input_timesteps": config.get("input_timesteps", 12),
        "forecast_horizon": config.get("forecast_horizon", 4),
        "normalize": config.get("normalize", "log_minmax"),
    }
    splits = config["splits"]

    loaders = []
    for split_name in ["train", "val", "test"]:
        ds = GMCPERA5Dataset({**base, **splits[split_name]})
        loaders.append(
            torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=(split_name == "train"),
            )
        )
    return tuple(loaders)
