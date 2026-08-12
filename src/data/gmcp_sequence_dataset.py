"""GMCP-only sequence dataset for precipitation forecasting.

Provides sliding-window samples of 6-hourly cumulative GMCP precipitation
for training PhyDiff-Net in GMCP-only mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from src.data.gmcp_reader import CHINA_REGION, load_gmcp_6h_period

logger = logging.getLogger(__name__)


class GMCPSequenceDataset(Dataset):
    """PyTorch Dataset for GMCP-only precipitation forecasting.

    Loads 6-hourly cumulative GMCP precipitation and creates sliding-window
    samples. The dataset can read preprocessed yearly NetCDF files (fast) or
    load directly from hourly GMCP files (slow, for small-scale tests).

    Args:
        config: Configuration dictionary containing:
            - data_path: Path to preprocessed 6-hourly NetCDF directory, or
              path to raw hourly GMCP data.
            - start_date: Start date ``YYYY-MM-DD``.
            - end_date: End date ``YYYY-MM-DD``.
            - input_timesteps: Number of input time steps.
            - forecast_horizon: Number of forecast time steps.
            - normalize: Normalization method ("log_minmax", "zscore", or None).
            - use_preprocessed: Whether to look for ``gmcp_6h_YYYY.nc`` files.
            - region: Optional region dict; defaults to China region.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.data_path = Path(config["data_path"])
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.input_timesteps = config.get("input_timesteps", 12)
        self.forecast_horizon = config.get("forecast_horizon", 4)
        self.normalize = config.get("normalize", None)
        self.use_preprocessed = config.get("use_preprocessed", True)
        self.region = config.get("region", CHINA_REGION)

        self.ds = self._load_data()
        self.n_times = len(self.ds.time)
        self._create_indices()
        self._compute_normalization_stats()

        logger.info(
            "GMCPSequenceDataset: %d samples, input shape [%d, 1, %d, %d], "
            "target shape [%d, %d, %d]",
            len(self),
            self.input_timesteps,
            len(self.ds.latitude),
            len(self.ds.longitude),
            self.forecast_horizon,
            len(self.ds.latitude),
            len(self.ds.longitude),
        )

    def _load_data(self) -> xr.Dataset:
        """Load 6-hourly cumulative GMCP data."""
        if self.use_preprocessed:
            ds = self._load_preprocessed()
            if ds is not None:
                return ds
            logger.warning(
                "Preprocessed files not found in %s; falling back to raw GMCP data.",
                self.data_path,
            )

        return load_gmcp_6h_period(
            data_path=self.data_path,
            start_date=self.start_date,
            end_date=self.end_date,
            region=self.region,
        )

    def _load_preprocessed(self) -> xr.Dataset | None:
        """Load preprocessed yearly NetCDF files if they exist."""
        start_year = int(self.start_date[:4])
        end_year = int(self.end_date[:4])

        files: List[Path] = []
        for year in range(start_year, end_year + 1):
            path = self.data_path / f"gmcp_6h_{year}.nc"
            if path.exists():
                files.append(path)

        if not files:
            return None

        if len(files) == 1:
            ds = xr.open_dataset(files[0])
        else:
            datasets = [xr.open_dataset(path) for path in files]
            ds = xr.concat(datasets, dim="time")
        ds = ds.sel(time=slice(self.start_date, self.end_date))
        return ds

    def _create_indices(self) -> None:
        """Create sliding window indices."""
        window_size = self.input_timesteps + self.forecast_horizon
        if self.n_times < window_size:
            raise ValueError(
                f"Not enough time steps ({self.n_times}) for window size {window_size}"
            )
        self.indices = list(range(self.n_times - window_size + 1))

    def _compute_normalization_stats(self) -> None:
        """Compute normalization statistics from a sample of the data."""
        if self.normalize is None:
            self.input_min = None
            self.input_max = None
            self.input_mean = None
            self.input_std = None
            return

        sample_size = min(20, self.n_times)
        sample_indices = np.linspace(0, self.n_times - 1, sample_size, dtype=int)
        sample = self.ds["precipitation_rate"].isel(time=sample_indices).values

        if self.normalize == "log_minmax":
            # log(x + 1) then min-max
            log_sample = np.log1p(sample)
            self.input_min = float(np.nanmin(log_sample))
            self.input_max = float(np.nanmax(log_sample))
            self.input_mean = None
            self.input_std = None
        elif self.normalize == "zscore":
            self.input_mean = float(np.nanmean(sample))
            self.input_std = float(np.nanstd(sample))
            self.input_min = None
            self.input_max = None
        else:
            raise ValueError(f"Unsupported normalization: {self.normalize}")

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input array."""
        if self.normalize is None:
            return x
        if self.normalize == "log_minmax":
            log_x = np.log1p(x)
            return (log_x - self.input_min) / max(self.input_max - self.input_min, 1e-8)
        if self.normalize == "zscore":
            return (x - self.input_mean) / max(self.input_std, 1e-8)
        return x

    def _denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Denormalize input array."""
        if self.normalize is None:
            return x
        if self.normalize == "log_minmax":
            log_x = x * max(self.input_max - self.input_min, 1e-8) + self.input_min
            return np.expm1(log_x)
        if self.normalize == "zscore":
            return x * self.input_std + self.input_mean
        return x

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.indices[idx]
        input_end = start + self.input_timesteps
        target_end = input_end + self.forecast_horizon

        input_data = self.ds["precipitation_rate"].isel(
            time=slice(start, input_end)
        ).values.astype(np.float32)
        target_data = self.ds["precipitation_rate"].isel(
            time=slice(input_end, target_end)
        ).values.astype(np.float32)

        # Replace NaN with 0
        input_data = np.nan_to_num(input_data, nan=0.0)
        target_data = np.nan_to_num(target_data, nan=0.0)

        input_data = self._normalize_input(input_data)
        target_data = self._normalize_input(target_data)

        # Add channel dimension: [T_in, H, W] -> [T_in, 1, H, W]
        input_data = input_data[:, None, :, :]

        return {
            "input": torch.from_numpy(input_data.copy()),
            "target": torch.from_numpy(target_data.copy()),
        }

    def close(self) -> None:
        """Close the underlying xarray dataset."""
        if hasattr(self, "ds") and self.ds is not None:
            self.ds.close()


def create_gmcp_dataloaders(
    config: Dict,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test DataLoaders for GMCP data.

    Args:
        config: Data configuration dictionary with train/val/test periods.
        batch_size: Batch size for all DataLoaders.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    base_config = {
        "data_path": config["data_path"],
        "input_timesteps": config.get("input_timesteps", 12),
        "forecast_horizon": config.get("forecast_horizon", 4),
        "normalize": config.get("normalize", None),
        "use_preprocessed": config.get("use_preprocessed", True),
        "region": config.get("region", CHINA_REGION),
    }

    splits = config["splits"]
    train_config = {**base_config, **splits["train"]}
    val_config = {**base_config, **splits["val"]}
    test_config = {**base_config, **splits["test"]}

    train_dataset = GMCPSequenceDataset(train_config)
    val_dataset = GMCPSequenceDataset(val_config)
    test_dataset = GMCPSequenceDataset(test_config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
