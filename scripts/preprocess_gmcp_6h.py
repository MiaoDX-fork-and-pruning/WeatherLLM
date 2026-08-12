"""Preprocess GMCP hourly data into 6-hourly cumulative NetCDF files.

This script aggregates GMCP hourly precipitation into 6-hourly cumulative
precipitation (mm/6h) and saves one NetCDF file per year. The preprocessed
files are consumed by ``GMCPSequenceDataset`` for model training.

Usage:
    python scripts/preprocess_gmcp_6h.py
    python scripts/preprocess_gmcp_6h.py --start-year 2000 --end-year 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gmcp_reader import CHINA_REGION, load_gmcp_6h_period

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("F:/GMCP_Precipitation")
DEFAULT_OUTPUT_DIR = Path("F:/GMCP_Precipitation_6h")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess GMCP hourly data to 6-hourly cumulative NetCDF."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Root directory containing hourly GMCP data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save yearly 6-hourly NetCDF files.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="First year to preprocess.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Last year to preprocess.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def preprocess_year(
    data_path: Path,
    output_dir: Path,
    year: int,
    overwrite: bool,
) -> Path | None:
    """Preprocess a single year of GMCP data.

    Args:
        data_path: Root directory containing hourly GMCP data.
        output_dir: Directory to save the output NetCDF.
        year: Year to preprocess.
        overwrite: Whether to overwrite an existing file.

    Returns:
        Path to the output file, or None if skipped.
    """
    output_path = output_dir / f"gmcp_6h_{year}.nc"
    if output_path.exists() and not overwrite:
        logger.info("Skipping %d (already exists)", year)
        return output_path

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    logger.info("Preprocessing GMCP 6-hourly data for %d...", year)
    ds = load_gmcp_6h_period(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        region=CHINA_REGION,
        batch_size=200,
    )

    # Add metadata
    ds.attrs["source"] = "GMCP"
    ds.attrs["temporal_resolution"] = "6h"
    ds.attrs["units"] = "mm/6h"
    ds["precipitation_rate"].attrs["long_name"] = "6-hourly cumulative precipitation"
    ds["precipitation_rate"].attrs["units"] = "mm/6h"

    output_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    logger.info("Saved %s", output_path)
    return output_path


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    for year in range(args.start_year, args.end_year + 1):
        preprocess_year(
            data_path=args.data_path,
            output_dir=args.output_dir,
            year=year,
            overwrite=args.overwrite,
        )

    logger.info("Preprocessing complete. Output directory: %s", args.output_dir)


if __name__ == "__main__":
    main()
