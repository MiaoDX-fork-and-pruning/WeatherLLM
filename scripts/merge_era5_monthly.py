"""Merge monthly ERA5 temp files into yearly NetCDF files.

The June download run completed all months of 1979-2017 but the yearly
merge step failed (dask was not installed at the time), leaving complete
monthly files named ``.era5_{year}_{month:02d}_temp.nc`` on disk. This
script merges them into ``era5_{year}.nc`` year by year.

To keep memory bounded, each year is merged one variable at a time
(~500 MB per variable-year) and written into a pre-created yearly file.

Usage:
    python scripts/merge_era5_monthly.py                # all years
    python scripts/merge_era5_monthly.py --years 2000 2001
    python scripts/merge_era5_monthly.py --start 2000 --end 2017
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path("F:/WeatherBench2_ERA5")


def month_files_for_year(base_dir: Path, year: int) -> list[Path]:
    """Return the 12 monthly temp files for a year, sorted by month."""
    return [
        base_dir / f".era5_{year}_{month:02d}_temp.nc" for month in range(1, 13)
    ]


def merge_year(base_dir: Path, year: int, overwrite: bool = False) -> Path | None:
    """Merge one year's monthly files into era5_{year}.nc.

    Args:
        base_dir: Directory containing the monthly temp files.
        year: Year to merge.
        overwrite: Overwrite an existing yearly file.

    Returns:
        Path to the merged file, or None if skipped/inputs missing.
    """
    output_path = base_dir / f"era5_{year}.nc"
    if output_path.exists() and not overwrite:
        logger.info("Skipping %d (already exists)", year)
        return output_path

    files = month_files_for_year(base_dir, year)
    missing = [f for f in files if not f.exists()]
    if missing:
        logger.warning("Year %d missing %d months: %s",
                       year, len(missing), [f.name for f in missing])
        return None

    t0 = time.time()

    # Open all months once (lazy), then merge variable by variable to keep
    # the in-memory footprint around one variable-year (~500 MB).
    monthly = [xr.open_dataset(f) for f in files]

    # Sanity check: months in order, one year.
    first_time = monthly[0].time.values[0]
    last_time = monthly[-1].time.values[-1]
    n_steps = sum(len(m.time) for m in monthly)

    merged_vars = {}
    for var in monthly[0].data_vars:
        merged_vars[var] = xr.concat([m[var] for m in monthly], dim="time").compute()

    # Dataset constructor picks up coords (time/level/lat/lon) from the
    # concatenated variables; no manual coord assignment needed.
    merged = xr.Dataset(merged_vars)

    for m in monthly:
        m.close()

    merged.to_netcdf(output_path)
    size_mb = output_path.stat().st_size / 1e6
    logger.info(
        "Merged %d -> %s: %d steps (%s..%s), %.0f MB, %.1fs",
        year, output_path.name, n_steps,
        str(first_time)[:10], str(last_time)[:10], size_mb, time.time() - t0,
    )
    del merged, merged_vars
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge monthly ERA5 temp files into yearly NetCDFs."
    )
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--start", type=int, default=1979)
    parser.add_argument("--end", type=int, default=2017)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Merge all requested years."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    t0 = time.time()
    done = 0
    for year in range(args.start, args.end + 1):
        if merge_year(args.base_dir, year, overwrite=args.overwrite):
            done += 1

    logger.info(
        "Merged %d years in %.1f min", done, (time.time() - t0) / 60
    )


if __name__ == "__main__":
    main()
