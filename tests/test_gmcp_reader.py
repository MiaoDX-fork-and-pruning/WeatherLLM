"""Unit tests for GMCP data reader."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "e:/weather")

import numpy as np
import pytest
import xarray as xr

from src.data.gmcp_reader import (
    CHINA_REGION,
    GMCPDataset,
    GMCPFileFinder,
    count_files_by_year_month,
    load_gmcp_for_period,
    sample_files,
)


def _create_gmcp_file(
    path: Path,
    year: int,
    month: int,
    day: int,
    hour: int,
    ascending_lat: bool = False,
) -> None:
    """Create a minimal GMCP-like NetCDF file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if ascending_lat:
        lats = np.arange(-89.95, 90.0, 0.1, dtype=np.float32)
    else:
        lats = np.arange(89.95, -90.0, -0.1, dtype=np.float32)
    lons = np.arange(-179.95, 180.0, 0.1, dtype=np.float32)
    rain = np.full((len(lats), len(lons)), 0.5, dtype=np.float32)

    ds = xr.Dataset(
        {
            "rain_rate": (["lat", "lon"], rain),
        },
        coords={
            "lat": (["lat"], lats),
            "lon": (["lon"], lons),
        },
        attrs={"description": "Hourly Rain Rate Map"},
    )
    ds.to_netcdf(path)


def _create_flat_gmcp_file(path: Path, year: int, month: int, day: int, hour: int) -> None:
    """Create a minimal GMCP-like NetCDF file in a flat directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lats = np.arange(89.95, -90.0, -0.1, dtype=np.float32)
    lons = np.arange(-179.95, 180.0, 0.1, dtype=np.float32)
    rain = np.full((len(lats), len(lons)), 0.5, dtype=np.float32)

    ds = xr.Dataset(
        {
            "rain_rate": (["lat", "lon"], rain),
        },
        coords={
            "lat": (["lat"], lats),
            "lon": (["lon"], lons),
        },
    )
    ds.to_netcdf(path)


def test_parse_filename_valid(tmp_path: Path) -> None:
    """GMCPFileFinder should parse valid GMCP filenames."""
    finder = GMCPFileFinder(tmp_path)
    parsed = finder._parse_filename(Path("GMCP_2020_06_15_08.nc"))
    assert parsed is not None
    assert parsed.time == datetime(2020, 6, 15, 8)


def test_parse_filename_invalid(tmp_path: Path) -> None:
    """GMCPFileFinder should reject non-GMCP filenames."""
    finder = GMCPFileFinder(tmp_path)
    assert finder._parse_filename(Path("era5_2020_06.nc")) is None
    assert finder._parse_filename(Path("GMCP_2020_06_15_8.nc")) is None


def test_find_files_standard_structure(tmp_path: Path) -> None:
    """Finder should discover files in YYYY/MM structure."""
    root = tmp_path / "gmcp"
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_00.nc", 2000, 1, 1, 0)
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_01.nc", 2000, 1, 1, 1)
    _create_gmcp_file(root / "2000" / "02" / "GMCP_2000_02_01_00.nc", 2000, 2, 1, 0)

    finder = GMCPFileFinder(root)
    files = finder.find_files()
    assert len(files) == 3
    assert files[0].time == datetime(2000, 1, 1, 0)
    assert files[-1].time == datetime(2000, 2, 1, 0)


def test_find_files_flat_2025(tmp_path: Path) -> None:
    """Finder should discover files in flat 2025 structure."""
    root = tmp_path / "gmcp"
    _create_flat_gmcp_file(root / "2025" / "GMCP_2025_01_01_00.nc", 2025, 1, 1, 0)
    _create_flat_gmcp_file(root / "2025" / "GMCP_2025_01_01_01.nc", 2025, 1, 1, 1)

    finder = GMCPFileFinder(root)
    files = finder.find_files()
    assert len(files) == 2


def test_find_files_time_range_includes_end_date(tmp_path: Path) -> None:
    """String end date should include all hours of that day."""
    root = tmp_path / "gmcp"
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_00.nc", 2000, 1, 1, 0)
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_02_00.nc", 2000, 1, 2, 0)
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_02_23.nc", 2000, 1, 2, 23)
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_03_00.nc", 2000, 1, 3, 0)

    finder = GMCPFileFinder(root)
    files = finder.find_files(start="2000-01-01", end="2000-01-02")
    assert len(files) == 3


def test_find_period_files(tmp_path: Path) -> None:
    """find_period_files should return files for the inclusive date range."""
    root = tmp_path / "gmcp"
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_00.nc", 2000, 1, 1, 0)
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_02_00.nc", 2000, 1, 2, 0)

    finder = GMCPFileFinder(root)
    files = finder.find_period_files("2000-01-01", "2000-01-02")
    assert len(files) == 2


def test_dataset_load_normalizes_names(tmp_path: Path) -> None:
    """GMCPDataset should rename variables and coordinates."""
    root = tmp_path / "gmcp"
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_00.nc", 2000, 1, 1, 0)
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_01.nc", 2000, 1, 1, 1)

    dataset = GMCPDataset(root)
    ds = dataset.load("2000-01-01", "2000-01-01")

    assert "precipitation_rate" in ds.data_vars
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "time" in ds.coords
    assert len(ds.time) == 2


def test_dataset_crops_to_region_descending_lat(tmp_path: Path) -> None:
    """GMCPDataset should crop to the specified region for descending lat."""
    root = tmp_path / "gmcp"
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_00.nc", 2000, 1, 1, 0)

    dataset = GMCPDataset(root, region=CHINA_REGION)
    ds = dataset.load("2000-01-01", "2000-01-01")

    lat_min = float(ds.latitude.min().values)
    lat_max = float(ds.latitude.max().values)
    lon_min = float(ds.longitude.min().values)
    lon_max = float(ds.longitude.max().values)

    assert lat_min >= CHINA_REGION["lat_min"] - 0.11
    assert lat_max <= CHINA_REGION["lat_max"] + 0.11
    assert lon_min >= CHINA_REGION["lon_min"] - 0.11
    assert lon_max <= CHINA_REGION["lon_max"] + 0.11


def test_dataset_crops_to_region_ascending_lat(tmp_path: Path) -> None:
    """GMCPDataset should crop to the specified region for ascending lat."""
    root = tmp_path / "gmcp"
    _create_gmcp_file(
        root / "2000" / "01" / "GMCP_2000_01_01_00.nc", 2000, 1, 1, 0, ascending_lat=True
    )

    dataset = GMCPDataset(root, region=CHINA_REGION)
    ds = dataset.load("2000-01-01", "2000-01-01")

    lat_min = float(ds.latitude.min().values)
    lat_max = float(ds.latitude.max().values)

    assert lat_min >= CHINA_REGION["lat_min"] - 0.11
    assert lat_max <= CHINA_REGION["lat_max"] + 0.11


def test_dataset_missing_variable_raises(tmp_path: Path) -> None:
    """GMCPDataset should raise if the configured variable is missing."""
    root = tmp_path / "gmcp"
    (root / "2000" / "01").mkdir(parents=True, exist_ok=True)

    lats = np.arange(89.95, -90.0, -0.1, dtype=np.float32)
    lons = np.arange(-179.95, 180.0, 0.1, dtype=np.float32)
    ds = xr.Dataset(
        {"other_var": (["lat", "lon"], np.random.rand(len(lats), len(lons)))},
        coords={"lat": lats, "lon": lons},
    )
    ds.to_netcdf(root / "2000" / "01" / "GMCP_2000_01_01_00.nc")

    dataset = GMCPDataset(root)
    with pytest.raises(KeyError):
        dataset.load("2000-01-01", "2000-01-01")


def test_load_gmcp_for_period(tmp_path: Path) -> None:
    """Convenience loader should return normalized data."""
    root = tmp_path / "gmcp"
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_00.nc", 2000, 1, 1, 0)

    ds = load_gmcp_for_period(root, "2000-01-01", "2000-01-01")
    assert "precipitation_rate" in ds.data_vars
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords


def test_count_files_by_year_month(tmp_path: Path) -> None:
    """Coverage audit should count files by year and month."""
    root = tmp_path / "gmcp"
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_00.nc", 2000, 1, 1, 0)
    _create_gmcp_file(root / "2000" / "01" / "GMCP_2000_01_01_01.nc", 2000, 1, 1, 1)
    _create_gmcp_file(root / "2000" / "02" / "GMCP_2000_02_01_00.nc", 2000, 2, 1, 0)

    counts = count_files_by_year_month(root)
    assert counts[(2000, 1)] == 2
    assert counts[(2000, 2)] == 1


def test_sample_files(tmp_path: Path) -> None:
    """Random sampling should return the requested number of files."""
    root = tmp_path / "gmcp"
    for hour in range(24):
        _create_gmcp_file(
            root / "2000" / "01" / f"GMCP_2000_01_01_{hour:02d}.nc", 2000, 1, 1, hour
        )

    samples = sample_files(root, n_samples=10, seed=42)
    assert len(samples) == 10


def test_load_files_batching(tmp_path: Path) -> None:
    """load_files should support small batch sizes."""
    root = tmp_path / "gmcp"
    for hour in range(10):
        _create_gmcp_file(
            root / "2000" / "01" / f"GMCP_2000_01_01_{hour:02d}.nc", 2000, 1, 1, hour
        )

    finder = GMCPFileFinder(root)
    files = finder.find_files()
    dataset = GMCPDataset(root)
    ds = dataset.load_files(files, batch_size=3)
    assert len(ds.time) == 10


def test_empty_directory_raises(tmp_path: Path) -> None:
    """Finding files in an empty directory should return an empty list."""
    root = tmp_path / "gmcp"
    root.mkdir(parents=True, exist_ok=True)
    finder = GMCPFileFinder(root)
    files = finder.find_files()
    assert files == []

    with pytest.raises(FileNotFoundError):
        load_gmcp_for_period(root, "2000-01-01", "2000-01-01")


def test_invalid_region_raises(tmp_path: Path) -> None:
    """GMCPDataset should validate region bounds."""
    with pytest.raises(ValueError):
        GMCPDataset(tmp_path, region={"lat_min": 50.0, "lat_max": 20.0})


def test_china_region_defaults() -> None:
    """Default region should match project China domain."""
    assert CHINA_REGION["lat_min"] == 18.0
    assert CHINA_REGION["lat_max"] == 54.0
    assert CHINA_REGION["lon_min"] == 73.0
    assert CHINA_REGION["lon_max"] == 135.0
