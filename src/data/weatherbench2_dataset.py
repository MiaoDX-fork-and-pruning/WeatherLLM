"""WeatherBench2 Dataset for PhyDiff-Net Training.

This module provides a PyTorch Dataset for loading WeatherBench2 ERA5 data
in NetCDF format, with support for sliding window sampling, normalization,
and train/val splitting.

Data format (WeatherBench2 ERA5 at 1.5° resolution):
- Input variables: [time, level, longitude, latitude] = [T, 5, 240, 121]
- Target variable: [time, longitude, latitude] = [T, 240, 121]
- Transposed to: [time, level, lat, lon] = [T, 5, 121, 240]

Example:
    >>> dataset = WeatherBench2Dataset(config, split='train')
    >>> sample = dataset[0]
    >>> print(sample['input'].shape)   # [4, 30, 121, 240]
    >>> print(sample['target'].shape)  # [4, 121, 240]
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Default ERA5 pressure-level variables available in WeatherBench2
DEFAULT_PRESSURE_LEVEL_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
]

DEFAULT_TARGET_VARIABLE = "total_precipitation_6hr"
DEFAULT_PRESSURE_LEVELS = [1000, 850, 500, 200, 100]


class WeatherBench2Dataset(Dataset):
    """PyTorch Dataset for WeatherBench2 ERA5 NetCDF data.

    Uses lazy loading from xarray to minimize memory usage. Only the
    requested slices are loaded from disk on each __getitem__ call.

    Args:
        config: Configuration dictionary containing data paths and parameters.
        split: 'train' or 'val'.
    """

    def __init__(self, config: Dict, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Extract configuration
        self.data_path = Path(config["data_path"])
        self.input_variables = config.get(
            "input_variables", DEFAULT_PRESSURE_LEVEL_VARIABLES
        )
        self.target_variable = config.get(
            "target_variable", DEFAULT_TARGET_VARIABLE
        )
        self.pressure_levels = config.get(
            "pressure_levels", DEFAULT_PRESSURE_LEVELS
        )
        self.input_timesteps = config.get("input_timesteps", 4)
        self.forecast_horizon = config.get("forecast_horizon", 4)
        self.normalize = config.get("normalize", True)
        self.val_split = config.get("val_split", 0.2)

        # Open dataset with lazy loading (no data loaded into memory yet)
        self._open_dataset()

        logger.info(
            f"WeatherBench2Dataset [{split}]: "
            f"{len(self)} samples, "
            f"input shape per sample: [{self.input_timesteps}, "
            f"{self.n_channels}, {self.lat_size}, {self.lon_size}]"
        )

    def _open_dataset(self) -> None:
        """Open NetCDF dataset and prepare for lazy loading."""
        import xarray as xr

        logger.info(f"Opening WeatherBench2 dataset from {self.data_path}")
        self.ds = xr.open_dataset(self.data_path)

        # Select time range
        time_start = self.config.get("time_start")
        time_end = self.config.get("time_end")
        if time_start and time_end:
            self.ds = self.ds.sel(time=slice(time_start, time_end))
            logger.info(
                f"Selected time range: {time_start} to {time_end}, "
                f"{len(self.ds.time)} timesteps"
            )

        # Select pressure levels
        if "level" in self.ds.dims:
            self.ds = self.ds.sel(level=self.pressure_levels)

        self.n_times = len(self.ds.time)
        self.n_levels = len(self.pressure_levels)
        self.n_vars = len(self.input_variables)
        self.n_channels = self.n_vars * self.n_levels

        # Get spatial dimensions
        # Data is stored as (time, level, longitude, latitude)
        # After transpose: (time, level, lat, lon)
        self.lon_size = len(self.ds.longitude)
        self.lat_size = len(self.ds.latitude)

        # Create sliding window indices with train/val split
        self._create_indices()

        # Compute normalization statistics from a sample
        if self.normalize:
            self._compute_normalization()

    def _create_indices(self) -> None:
        """Create sliding window indices with train/val split."""
        window_size = self.input_timesteps + self.forecast_horizon
        all_indices = list(range(self.n_times - window_size + 1))

        # Temporal split: first part = train, last part = val
        n_val = max(1, int(len(all_indices) * self.val_split))
        n_train = len(all_indices) - n_val

        if self.split == "train":
            self.indices = all_indices[:n_train]
        else:
            self.indices = all_indices[n_train:]

        logger.info(
            f"Split '{self.split}': {len(self.indices)} samples "
            f"(train={n_train}, val={n_val})"
        )

    def _compute_normalization(self) -> None:
        """Compute normalization statistics from a subset of data."""
        logger.info("Computing normalization statistics...")

        # Sample a few timesteps to compute stats (avoid loading all data)
        sample_size = min(20, self.n_times)
        sample_indices = np.linspace(0, self.n_times - 1, sample_size, dtype=int)

        input_sum = np.zeros(self.n_channels, dtype=np.float64)
        input_sq_sum = np.zeros(self.n_channels, dtype=np.float64)
        target_sum = 0.0
        target_sq_sum = 0.0
        n_pixels = 0

        for t_idx in sample_indices:
            # Load input for this timestep: [level, lon, lat] -> [level, lat, lon]
            input_slice = self._load_input_timestep(t_idx)
            # input_slice shape: [n_channels, lat, lon]
            input_sum += input_slice.mean(axis=(1, 2))
            input_sq_sum += (input_slice ** 2).mean(axis=(1, 2))

            # Load target for this timestep: [lon, lat] -> [lat, lon]
            target_slice = self._load_target_timestep(t_idx)
            target_sum += target_slice.mean()
            target_sq_sum += (target_slice ** 2).mean()
            n_pixels += 1

        self.input_mean = (input_sum / n_pixels).astype(np.float32)
        self.input_std = np.sqrt(
            input_sq_sum / n_pixels - self.input_mean ** 2
        ).astype(np.float32)
        self.input_std = np.maximum(self.input_std, 1e-8)

        self.target_mean = float(target_sum / n_pixels)
        self.target_std = float(
            np.sqrt(target_sq_sum / n_pixels - self.target_mean ** 2)
        )
        self.target_std = max(self.target_std, 1e-8)

        logger.info(
            f"Normalization stats: "
            f"input mean range [{self.input_mean.min():.4f}, {self.input_mean.max():.4f}], "
            f"target mean={self.target_mean:.4f}, std={self.target_std:.4f}"
        )

    def _load_input_timestep(self, t_idx: int) -> np.ndarray:
        """Load input data for a single timestep using lazy loading.

        Args:
            t_idx: Time index.

        Returns:
            Input array [n_channels, lat, lon] in float32.
        """
        arrays = []
        for var_name in self.input_variables:
            # Load [level, longitude, latitude] for this timestep
            var_data = self.ds[var_name].isel(time=t_idx).values.astype(np.float32)
            # Transpose: [level, lon, lat] -> [level, lat, lon]
            var_data = np.transpose(var_data, (0, 2, 1))
            arrays.append(var_data)
        # Stack: [n_vars * n_levels, lat, lon]
        return np.concatenate(arrays, axis=0)

    def _load_target_timestep(self, t_idx: int) -> np.ndarray:
        """Load target data for a single timestep using lazy loading.

        Args:
            t_idx: Time index.

        Returns:
            Target array [lat, lon] in float32.
        """
        # Load [longitude, latitude] for this timestep
        target = self.ds[self.target_variable].isel(time=t_idx).values.astype(np.float32)
        # Transpose: [lon, lat] -> [lat, lon]
        return np.transpose(target, (1, 0))

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - 'input': Input tensor [input_timesteps, n_channels, lat, lon]
                - 'target': Target tensor [forecast_horizon, lat, lon]
        """
        start = self.indices[idx]
        input_end = start + self.input_timesteps
        target_end = input_end + self.forecast_horizon

        # Load input timesteps: [T_in, n_channels, lat, lon]
        input_list = []
        for t in range(start, input_end):
            input_list.append(self._load_input_timestep(t))
        x = np.stack(input_list, axis=0)  # [T_in, n_channels, lat, lon]

        # Load target timesteps: [T_out, lat, lon]
        target_list = []
        for t in range(input_end, target_end):
            target_list.append(self._load_target_timestep(t))
        y = np.stack(target_list, axis=0)  # [T_out, lat, lon]

        # Normalize
        if self.normalize and self.input_mean is not None:
            x = (x - self.input_mean[None, :, None, None]) / self.input_std[
                None, :, None, None
            ]
            y = (y - self.target_mean) / self.target_std

        return {
            "input": torch.from_numpy(x.copy()),
            "target": torch.from_numpy(y.copy()),
        }

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Return normalization statistics for denormalization."""
        return {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    def close(self) -> None:
        """Close the underlying xarray dataset."""
        if hasattr(self, "ds") and self.ds is not None:
            self.ds.close()


def create_weatherbench2_dataloaders(
    config: Dict,
    batch_size: int = 2,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation DataLoaders for WeatherBench2 data.

    Args:
        config: Data configuration dictionary.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = WeatherBench2Dataset(config, split="train")
    val_dataset = WeatherBench2Dataset(config, split="val")

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

    return train_loader, val_loader
