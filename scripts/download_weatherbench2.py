#!/usr/bin/env python3
"""
WeatherBench2 ERA5数据下载脚本
从Google Cloud Storage下载预处理好的ERA5数据

用法：
    # 下载训练数据（1979-2017年，分批下载，每月下载后合并为年度文件）
    python download_weatherbench2.py --start_year 1979 --end_year 1983 --minimal

    # 下载评测数据（2018-2019年）
    python download_weatherbench2.py --start_year 2018 --end_year 2019

    # 只下载特定年份
    python download_weatherbench2.py --single_year 2019
"""

import os
import sys
import gc
import time
import argparse
from pathlib import Path

try:
    import xarray as xr
    import gcsfs
except ImportError:
    print("Please install: pip install xarray gcsfs zarr")
    sys.exit(1)

# ==================== Configuration ====================

WB2_BASE_URL = "gs://weatherbench2/datasets/era5"
WB2_DATASET = "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

# Pressure level variables
PRESSURE_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
]

# Single level variables
SINGLE_LEVEL_VARIABLES = [
    "total_precipitation_6hr",
    "total_precipitation_24hr",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_column_water_vapour",
]

# Target pressure levels
TARGET_LEVELS = [1000, 850, 500, 200, 100]

# Storage root directory
BASE_DIR = r"F:\WeatherBench2_ERA5"


def open_dataset():
    """Open the remote Zarr dataset with selected variables and levels."""
    print("[1/3] Connecting to Google Cloud Storage...", flush=True)
    fs = gcsfs.GCSFileSystem(token="anon")

    print("[2/3] Opening Zarr dataset...", flush=True)
    url = f"{WB2_BASE_URL}/{WB2_DATASET}"
    print(f"  URL: {url}", flush=True)
    t0 = time.time()
    ds = xr.open_zarr(fs.get_mapper(url), consolidated=True)
    t1 = time.time()
    print(f"  Zarr opened in {t1-t0:.1f}s", flush=True)
    print(f"  Dimensions: {dict(ds.dims)}", flush=True)
    print(f"  Variables: {len(ds.data_vars)}", flush=True)

    # Select needed variables
    print("[3/3] Selecting variables...", flush=True)
    all_needed_vars = PRESSURE_VARIABLES + SINGLE_LEVEL_VARIABLES
    available_vars = [v for v in all_needed_vars if v in ds.data_vars]
    print(f"  Selected variables: {len(available_vars)}", flush=True)

    ds = ds[available_vars]

    # Select pressure levels
    if "level" in ds.dims:
        target_levels_available = [l for l in TARGET_LEVELS if l in ds.level.values]
        print(f"  Selected pressure levels: {target_levels_available}", flush=True)
        ds = ds.sel(level=target_levels_available)

    return ds


def download_year_by_month(ds, year, base_dir):
    """Download a single year by combining monthly downloads.

    Downloads each month separately (low memory), saves as temp files,
    then merges into a single yearly NetCDF file.
    """
    import calendar

    output_file = os.path.join(base_dir, f"era5_{year}.nc")

    # Skip if already downloaded
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        if file_size > 100:  # >100MB means likely complete
            print(f"  [{year}] Already exists ({file_size:.1f} MB), skipping", flush=True)
            return True

    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    year_data = ds.sel(time=slice(year_start, year_end))

    if len(year_data.time) == 0:
        print(f"  [{year}] No data available", flush=True)
        return False

    print(f"  [{year}] Time steps: {len(year_data.time)}", flush=True)

    # Download month by month
    monthly_files = []
    total_months = 12

    for month in range(1, 13):
        month_start = f"{year}-{month:02d}-01"
        last_day = calendar.monthrange(year, month)[1]
        month_end = f"{year}-{month:02d}-{last_day:02d}"

        month_data = year_data.sel(time=slice(month_start, month_end))
        n_time = len(month_data.time)

        if n_time == 0:
            continue

        temp_file = os.path.join(base_dir, f".era5_{year}_{month:02d}_temp.nc")
        print(f"  [{year}-{month:02d}] {n_time} timesteps...", end="", flush=True)

        t0 = time.time()
        try:
            month_data.to_netcdf(temp_file, format="NETCDF4")
            t1 = time.time()
            month_size = os.path.getsize(temp_file) / (1024 * 1024)
            print(f" {month_size:.1f} MB ({t1-t0:.0f}s)", flush=True)
            monthly_files.append(temp_file)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            return False
        finally:
            del month_data
            gc.collect()

    if not monthly_files:
        print(f"  [{year}] No monthly files produced", flush=True)
        return False

    # Merge monthly files into yearly file
    print(f"  [{year}] Merging {len(monthly_files)} monthly files...", flush=True)
    try:
        combined = xr.open_mfdataset(monthly_files, combine="by_coords")
        combined.to_netcdf(output_file, format="NETCDF4")
        combined.close()
        del combined
        gc.collect()

        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  [{year}] DONE! {output_file} ({file_size:.1f} MB)", flush=True)

        # Clean up temp files
        for f in monthly_files:
            try:
                os.remove(f)
            except OSError:
                pass

        return True
    except Exception as e:
        print(f"  [{year}] Merge failed: {e}", flush=True)
        # If merge fails, keep monthly files as backup
        print(f"  [{year}] Keeping monthly files as backup", flush=True)
        return False


def download_year_range(start_year, end_year, base_dir, variables=None):
    """Download data for a range of years (year-by-year, month-by-month internally)."""
    print("=" * 60, flush=True)
    print("WeatherBench2 ERA5 Download", flush=True)
    print("=" * 60, flush=True)
    print(f"Dataset: {WB2_DATASET}", flush=True)
    print(f"Year range: {start_year}-{end_year}", flush=True)
    print(f"Storage: {base_dir}", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(base_dir, exist_ok=True)
    ds = open_dataset()

    print("\nStarting year-by-year download (month-by-month internally)...", flush=True)
    success_years = []
    failed_years = []

    for year in range(start_year, end_year + 1):
        print(f"\n{'='*40}", flush=True)
        print(f"Downloading year {year}...", flush=True)
        print(f"{'='*40}", flush=True)

        t_year_start = time.time()
        ok = download_year_by_month(ds, year, base_dir)
        t_year_end = time.time()

        if ok:
            elapsed_min = (t_year_end - t_year_start) / 60
            print(f"  [{year}] Completed in {elapsed_min:.1f} minutes", flush=True)
            success_years.append(year)
        else:
            failed_years.append(year)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("Download complete!", flush=True)
    print("=" * 60, flush=True)
    print(f"Success: {success_years}", flush=True)
    print(f"Failed: {failed_years}", flush=True)
    print("=" * 60, flush=True)


def main():
    parser = argparse.ArgumentParser(description="WeatherBench2 ERA5 Download")
    parser.add_argument("--start_year", type=int, default=2018, help="Start year")
    parser.add_argument("--end_year", type=int, default=2019, help="End year")
    parser.add_argument("--single_year", type=int, help="Download single year")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="Storage directory")
    parser.add_argument("--minimal", action="store_true",
                        help="Minimal mode (subset of variables)")
    args = parser.parse_args()

    if args.single_year:
        start_year = args.single_year
        end_year = args.single_year
    else:
        start_year = args.start_year
        end_year = args.end_year

    download_year_range(start_year, end_year, args.base_dir)


if __name__ == "__main__":
    main()
