"""Convert GMCP yearly NetCDF files to memory-mappable float16 .npy arrays.

The training bottleneck is dask-chunked reads of the sliding windows (16x
read amplification, ~0.35 s/sample on the mechanical F: drive). Plain .npy
files read via ``np.load(mmap_mode='r')`` let the OS page cache serve
repeated reads, and float16 halves the cache footprint (22 GB -> 11 GB,
fitting mostly in the machine's 17 GB RAM).

Precipitation in float16 loses ~1e-3 relative precision, negligible after
log1p normalization.

Usage:
    python scripts/convert_gmcp_to_npy.py                    # 2000-2024
    python scripts/convert_gmcp_to_npy.py --start 2000 --end 2010
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

DEFAULT_GMCP_DIR = Path("F:/GMCP_Precipitation_6h")


def convert_year(gmcp_dir: Path, year: int, overwrite: bool = False) -> None:
    """Convert one year's NetCDF to a float16 .npy sidecar file."""
    nc_path = gmcp_dir / f"gmcp_6h_{year}.nc"
    npy_path = gmcp_dir / f"gmcp_6h_{year}.npy"

    if npy_path.exists() and not overwrite:
        logger.info("Skipping %d (npy exists)", year)
        return
    if not nc_path.exists():
        logger.warning("Missing %s", nc_path)
        return

    t0 = time.time()
    with xr.open_dataset(nc_path) as ds:
        arr = ds["precipitation_rate"].values  # [T, H, W] float32
    arr16 = arr.astype(np.float16)
    np.save(npy_path, arr16)
    logger.info(
        "%d: %s -> %s (%d steps, %.1f MB, %.1fs)",
        year, nc_path.name, npy_path.name,
        arr16.shape[0], npy_path.stat().st_size / 1e6, time.time() - t0,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert GMCP yearly NetCDF to float16 npy."
    )
    parser.add_argument("--gmcp_dir", type=Path, default=DEFAULT_GMCP_DIR)
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Convert all requested years."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    t0 = time.time()
    for year in range(args.start, args.end + 1):
        convert_year(args.gmcp_dir, year, overwrite=args.overwrite)
    logger.info("All done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
