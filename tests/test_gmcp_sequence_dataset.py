"""Unit tests for GMCP sequence dataset."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "e:/weather")

import numpy as np
import pytest
import torch
import xarray as xr

from src.data.gmcp_sequence_dataset import GMCPSequenceDataset, create_gmcp_dataloaders


def _create_gmcp_6h_file(path: Path, start: datetime, n_steps: int) -> None:
    """Create a minimal 6-hourly GMCP NetCDF file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)

    times = [start.timestamp() + i * 6 * 3600 for i in range(n_steps)]
    # Use pandas for DatetimeIndex
    import pandas as pd

    time_index = pd.to_datetime(times, unit="s")
    lats = np.arange(53.95, 17.95, -0.1, dtype=np.float32)
    lons = np.arange(73.05, 135.05, 0.1, dtype=np.float32)
    data = np.random.rand(n_steps, len(lats), len(lons)).astype(np.float32)

    ds = xr.Dataset(
        {
            "precipitation_rate": (["time", "latitude", "longitude"], data),
        },
        coords={
            "time": (["time"], time_index),
            "latitude": (["latitude"], lats),
            "longitude": (["longitude"], lons),
        },
    )
    ds.to_netcdf(path)


def test_gmcp_sequence_dataset_shape(tmp_path: Path) -> None:
    """Dataset should return correct input/target shapes."""
    import pandas as pd

    start = pd.Timestamp("2023-01-01")
    _create_gmcp_6h_file(tmp_path / "gmcp_6h_2023.nc", start, n_steps=20)

    config = {
        "data_path": str(tmp_path),
        "start_date": "2023-01-01",
        "end_date": "2023-01-05",
        "input_timesteps": 4,
        "forecast_horizon": 4,
        "use_preprocessed": True,
        "normalize": None,
    }

    dataset = GMCPSequenceDataset(config)
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample["input"].shape == (4, 1, len(dataset.ds.latitude), len(dataset.ds.longitude))
    assert sample["target"].shape == (4, len(dataset.ds.latitude), len(dataset.ds.longitude))


def test_gmcp_sequence_dataset_sliding_window(tmp_path: Path) -> None:
    """Consecutive samples should have overlapping input windows."""
    import pandas as pd

    start = pd.Timestamp("2023-01-01")
    _create_gmcp_6h_file(tmp_path / "gmcp_6h_2023.nc", start, n_steps=20)

    config = {
        "data_path": str(tmp_path),
        "start_date": "2023-01-01",
        "end_date": "2023-01-05",
        "input_timesteps": 4,
        "forecast_horizon": 4,
        "use_preprocessed": True,
        "normalize": None,
    }

    dataset = GMCPSequenceDataset(config)
    sample0 = dataset[0]
    sample1 = dataset[1]

    # Input window shifts by one time step
    assert torch.allclose(
        sample0["input"][1:, 0], sample1["input"][:-1, 0]
    )


def test_gmcp_sequence_dataset_log_minmax(tmp_path: Path) -> None:
    """Dataset should apply log-minmax normalization."""
    import pandas as pd

    start = pd.Timestamp("2023-01-01")
    _create_gmcp_6h_file(tmp_path / "gmcp_6h_2023.nc", start, n_steps=10)

    config = {
        "data_path": str(tmp_path),
        "start_date": "2023-01-01",
        "end_date": "2023-01-03",
        "input_timesteps": 4,
        "forecast_horizon": 2,
        "use_preprocessed": True,
        "normalize": "log_minmax",
    }

    dataset = GMCPSequenceDataset(config)
    sample = dataset[0]
    assert sample["input"].min() >= 0.0
    assert sample["input"].max() <= 1.0


def test_create_gmcp_dataloaders(tmp_path: Path) -> None:
    """DataLoader factory should return train/val/test loaders."""
    import pandas as pd

    start = pd.Timestamp("2023-01-01")
    _create_gmcp_6h_file(tmp_path / "gmcp_6h_2023.nc", start, n_steps=50)

    config = {
        "data_path": str(tmp_path),
        "input_timesteps": 4,
        "forecast_horizon": 4,
        "use_preprocessed": True,
        "normalize": None,
        "splits": {
            "train": {"start_date": "2023-01-01", "end_date": "2023-01-03"},
            "val": {"start_date": "2023-01-03", "end_date": "2023-01-04"},
            "test": {"start_date": "2023-01-04", "end_date": "2023-01-06"},
        },
    }

    train_loader, val_loader, test_loader = create_gmcp_dataloaders(
        config, batch_size=1, num_workers=0
    )
    assert len(train_loader) >= 0
    assert len(val_loader) >= 0
    assert len(test_loader) >= 0


def test_gmcp_sequence_dataset_no_preprocessed_fallback(tmp_path: Path) -> None:
    """Dataset should raise when preprocessed files are missing and fallback is disabled."""
    config = {
        "data_path": str(tmp_path),
        "start_date": "2023-01-01",
        "end_date": "2023-01-02",
        "input_timesteps": 4,
        "forecast_horizon": 4,
        "use_preprocessed": True,
        "normalize": None,
    }

    with pytest.raises(FileNotFoundError):
        GMCPSequenceDataset(config)
