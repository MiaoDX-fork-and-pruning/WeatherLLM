"""GMCP precipitation data reader.

Handles the real on-disk layout of the GMCP dataset:

- Files are hourly NetCDFs named ``GMCP_YYYY_MM_DD_HH.nc``.
- Years 2000-2024 are organized as ``YYYY/MM/GMCP_YYYY_MM_DD_HH.nc``.
- Year 2025 (partial) is stored flat as ``2025/GMCP_YYYY_MM_DD_HH.nc``.
- The precipitation variable is ``rain_rate`` and coordinates are ``lat``/``lon``.
- Files do not contain a ``time`` dimension; time is inferred from the filename.

The reader normalizes all of this so downstream code receives a Dataset with
variable ``precipitation_rate`` and coordinates ``latitude``/``longitude``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

_GMCP_FILENAME_RE = re.compile(
    r"^GMCP_(\d{4})_(\d{2})_(\d{2})_(\d{2})\.nc$"
)

# Default China region used by the project.
CHINA_REGION = {
    "lat_min": 18.0,
    "lat_max": 54.0,
    "lon_min": 73.0,
    "lon_max": 135.0,
}


@dataclass(frozen=True)
class GMCPFile:
    """A single GMCP file with its inferred timestamp."""

    path: Path
    time: datetime

    @property
    def year(self) -> int:
        return self.time.year


class GMCPFileFinder:
    """Discover GMCP files on disk and parse their timestamps."""

    def __init__(self, data_path: str | Path) -> None:
        """Initialize the finder.

        Args:
            data_path: Root directory containing GMCP data (e.g.
                ``F:/GMCP_Precipitation``).
        """
        self.data_path = Path(data_path)

    def find_files(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> list[GMCPFile]:
        """Find all GMCP files within an optional time range.

        Date strings are interpreted as the start of ``start`` and the end of
        ``end``. For example, ``end="2020-01-02"`` includes all hours of
        January 2, 2020. For inclusive date-range queries, prefer
        ``find_period_files``.

        Args:
            start: Optional inclusive start time.
            end: Optional inclusive end time.

        Returns:
            List of ``GMCPFile`` entries sorted by time.
        """
        start = self._normalize_time(start, end_of_day=False)
        end = self._normalize_time(end, end_of_day=True)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"GMCP data path does not exist: {self.data_path}"
            )

        candidates: list[Path] = []

        # 2000-2024 use YYYY/MM structure.
        for year_dir in sorted(self.data_path.glob("[0-9][0-9][0-9][0-9]")):
            year = int(year_dir.name)
            if year < 2000 or year > 2024:
                continue
            for month_dir in sorted(year_dir.glob("[0-9][0-9]")):
                candidates.extend(sorted(month_dir.glob("GMCP_*.nc")))

        # 2025 uses a flat structure (observed partial data).
        flat_2025 = self.data_path / "2025"
        if flat_2025.exists():
            candidates.extend(sorted(flat_2025.glob("GMCP_*.nc")))

        files: list[GMCPFile] = []
        for path in candidates:
            parsed = self._parse_filename(path)
            if parsed is None:
                continue
            if start is not None and parsed.time < start:
                continue
            if end is not None and parsed.time > end:
                continue
            files.append(parsed)

        files.sort(key=lambda f: f.time)
        logger.info("Found %d GMCP files in %s", len(files), self.data_path)
        return files

    def find_period_files(
        self,
        start_date: str,
        end_date: str,
    ) -> list[GMCPFile]:
        """Find GMCP files covering a date range.

        Args:
            start_date: Start date string ``YYYY-MM-DD``.
            end_date: End date string ``YYYY-MM-DD``.

        Returns:
            Files from the start of ``start_date`` through the end of
            ``end_date``.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(hours=23)
        return self.find_files(start, end)

    @staticmethod
    def _parse_filename(path: Path) -> GMCPFile | None:
        match = _GMCP_FILENAME_RE.match(path.name)
        if not match:
            return None
        year, month, day, hour = map(int, match.groups())
        try:
            time = datetime(year, month, day, hour)
        except ValueError:
            logger.warning("Skipping invalid GMCP filename: %s", path)
            return None
        return GMCPFile(path=path, time=time)

    @staticmethod
    def _normalize_time(
        value: datetime | str | None,
        end_of_day: bool = False,
    ) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            parsed = datetime.strptime(value, "%Y-%m-%d")
            if end_of_day:
                return parsed + timedelta(hours=23)
            return parsed
        return value


class GMCPDataset:
    """Lazy loader for GMCP hourly precipitation data."""

    def __init__(
        self,
        data_path: str | Path,
        variable: str = "rain_rate",
        region: dict[str, float] | None = None,
    ) -> None:
        """Initialize the GMCP dataset loader.

        Args:
            data_path: Root directory containing GMCP data.
            variable: Name of the precipitation variable in raw files.
            region: Optional region dict with ``lat_min``, ``lat_max``,
                ``lon_min``, ``lon_max``. Defaults to the China region.
        """
        self.finder = GMCPFileFinder(data_path)
        self.variable = variable
        self.region = self._validate_region(
            region if region is not None else CHINA_REGION.copy()
        )

    @staticmethod
    def _validate_region(region: dict[str, float]) -> dict[str, float]:
        """Validate and return a region dictionary.

        Args:
            region: Region dict with lat_min, lat_max, lon_min, lon_max.

        Raises:
            ValueError: If required keys are missing or bounds are invalid.
        """
        required = {"lat_min", "lat_max", "lon_min", "lon_max"}
        missing = required - set(region.keys())
        if missing:
            raise ValueError(f"Region is missing required keys: {missing}")

        if region["lat_min"] >= region["lat_max"]:
            raise ValueError("lat_min must be less than lat_max")
        if region["lon_min"] >= region["lon_max"]:
            raise ValueError("lon_min must be less than lon_max")

        return dict(region)

    def load(
        self,
        start_date: str,
        end_date: str,
        batch_size: int = 200,
    ) -> xr.Dataset:
        """Load GMCP data for a date range.

        Args:
            start_date: Start date ``YYYY-MM-DD``.
            end_date: End date ``YYYY-MM-DD``.
            batch_size: Maximum number of files to load into memory at once.

        Returns:
            Dataset with variable ``precipitation_rate`` and coordinates
            ``latitude``/``longitude``/``time``.

        Raises:
            FileNotFoundError: If no files are found for the period.
        """
        files = self.finder.find_period_files(start_date, end_date)
        if not files:
            raise FileNotFoundError(
                f"No GMCP files found for {start_date} to {end_date}"
            )

        return self.load_files(files, batch_size=batch_size)

    def load_files(
        self,
        files: Sequence[GMCPFile],
        batch_size: int = 200,
    ) -> xr.Dataset:
        """Load a specific list of files in batches.

        Args:
            files: Sequence of ``GMCPFile`` objects.
            batch_size: Maximum number of files to load into memory at once.

        Returns:
            Concatenated Dataset normalized for downstream use.
        """
        if not files:
            raise FileNotFoundError("No GMCP files provided")

        batches: list[xr.Dataset] = []
        for start in range(0, len(files), batch_size):
            batch = files[start : start + batch_size]
            datasets = [self._load_single(file_info) for file_info in batch]
            batches.append(xr.concat(datasets, dim="time"))

        merged = xr.concat(batches, dim="time") if len(batches) > 1 else batches[0]
        merged = merged.sortby("time")

        logger.info(
            "Loaded GMCP data: %d time steps, variable shape: %s",
            len(merged.time),
            merged["precipitation_rate"].shape,
        )
        return merged

    def _load_single(self, file_info: GMCPFile) -> xr.Dataset:
        with xr.open_dataset(str(file_info.path)) as ds:
            if self.variable not in ds.data_vars:
                raise KeyError(
                    f"Variable '{self.variable}' not found in {file_info.path}; "
                    f"available: {list(ds.data_vars)}"
                )

            da = ds[self.variable]

            # Rename coordinates to project convention.
            rename_map: dict[str, str] = {}
            if "lat" in da.coords:
                rename_map["lat"] = "latitude"
            if "lon" in da.coords:
                rename_map["lon"] = "longitude"
            if rename_map:
                da = da.rename(rename_map)

            # Ensure the data array has a time coordinate.
            da = da.expand_dims("time")
            da["time"] = [pd.Timestamp(file_info.time)]

            # Build a clean dataset with the standardized variable name.
            out = da.to_dataset(name="precipitation_rate")

            # Crop to region if coordinates are present.
            out = self._crop(out)

            # Load cropped data into memory so the file handle can close.
            return out.load()

    def _crop(self, ds: xr.Dataset) -> xr.Dataset:
        has_lat = "latitude" in ds.coords
        has_lon = "longitude" in ds.coords
        if not has_lat or not has_lon:
            return ds

        lat_coord = ds.latitude
        lon_coord = ds.longitude

        # Detect coordinate direction to avoid empty slices.
        if lat_coord[0] > lat_coord[-1]:
            lat_slice = slice(
                self.region["lat_max"], self.region["lat_min"]
            )
        else:
            lat_slice = slice(
                self.region["lat_min"], self.region["lat_max"]
            )

        if lon_coord[0] < lon_coord[-1]:
            lon_slice = slice(
                self.region["lon_min"], self.region["lon_max"]
            )
        else:
            lon_slice = slice(
                self.region["lon_max"], self.region["lon_min"]
            )

        return ds.sel(
            latitude=lat_slice,
            longitude=lon_slice,
        )


def load_gmcp_for_period(
    data_path: str | Path,
    start_date: str,
    end_date: str,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Convenience function to load GMCP data for a period.

    Args:
        data_path: Root directory containing GMCP data.
        start_date: Start date ``YYYY-MM-DD``.
        end_date: End date ``YYYY-MM-DD``.
        region: Optional region dict; defaults to China region.

    Returns:
        Normalized GMCP Dataset.
    """
    dataset = GMCPDataset(data_path, region=region)
    return dataset.load(start_date, end_date)


def count_files_by_year_month(
    data_path: str | Path,
) -> dict[tuple[int, int], int]:
    """Count available GMCP files grouped by (year, month).

    Useful for coverage auditing without loading any data.

    Args:
        data_path: Root directory containing GMCP data.

    Returns:
        Mapping from ``(year, month)`` to file count.
    """
    finder = GMCPFileFinder(data_path)
    files = finder.find_files()
    counts: dict[tuple[int, int], int] = {}
    for file_info in files:
        key = (file_info.year, file_info.time.month)
        counts[key] = counts.get(key, 0) + 1
    return counts


def sample_files(
    data_path: str | Path,
    n_samples: int,
    seed: int = 42,
) -> list[GMCPFile]:
    """Randomly sample GMCP files for exploratory analysis.

    Args:
        data_path: Root directory containing GMCP data.
        n_samples: Number of files to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled ``GMCPFile`` entries.
    """
    finder = GMCPFileFinder(data_path)
    files = finder.find_files()
    if not files:
        return []

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(files), size=min(n_samples, len(files)), replace=False)
    return [files[i] for i in indices]


def load_gmcp_6h_period(
    data_path: str | Path,
    start_date: str,
    end_date: str,
    region: dict[str, float] | None = None,
    batch_size: int = 200,
) -> xr.Dataset:
    """Load GMCP data for a period and aggregate to 6-hourly cumulative.

    The function loads hourly files covering the inclusive date range, crops to
    the requested region, and sums precipitation into consecutive 6-hour windows.
    Windows are aligned to calendar boundaries (00, 06, 12, 18 UTC). Incomplete
    windows at the beginning or end are dropped.

    Args:
        data_path: Root directory containing GMCP data.
        start_date: Start date ``YYYY-MM-DD``.
        end_date: End date ``YYYY-MM-DD``.
        region: Optional region dict; defaults to China region.
        batch_size: Number of hourly files to load per batch.

    Returns:
        Dataset with variable ``precipitation_rate`` [time, latitude, longitude]
        representing 6-hourly cumulative precipitation (mm/6h).
    """
    finder = GMCPFileFinder(data_path)
    files = finder.find_period_files(start_date, end_date)
    if not files:
        raise FileNotFoundError(
            f"No GMCP files found for {start_date} to {end_date}"
        )

    dataset = GMCPDataset(data_path, region=region)
    ds = dataset.load_files(files, batch_size=batch_size)

    # Resample to 6-hourly cumulative precipitation, aligned to 0/6/12/18 UTC.
    precip = ds["precipitation_rate"]
    precip_6h = precip.resample(time="6h", closed="right", label="right").sum()

    logger.info(
        "Aggregated GMCP to 6-hourly: %d windows, shape: %s",
        len(precip_6h.time),
        precip_6h.shape,
    )
    return precip_6h.to_dataset(name="precipitation_rate")


def load_gmcp_6h_windows(
    data_path: str | Path,
    start_date: str,
    end_date: str,
    region: dict[str, float] | None = None,
    batch_size: int = 200,
) -> xr.Dataset:
    """Alias for :func:`load_gmcp_6h_period`."""
    return load_gmcp_6h_period(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        region=region,
        batch_size=batch_size,
    )
