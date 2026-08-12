"""GMCP precipitation exploratory analysis.

Performs a coverage audit and sampled exploratory analysis of the GMCP
dataset stored at ``F:/GMCP_Precipitation``. Results (figures and a JSON
summary) are written to ``outputs/figures/gmcp_analysis/``.

The script avoids loading all 225,000 hourly files by sampling independent
6-hour windows. Each window is formed by six consecutive hourly files, and
the precipitation rates are summed to produce a true 6-hourly cumulative
precipitation field.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gmcp_reader import CHINA_REGION, GMCPDataset, GMCPFile, GMCPFileFinder

logger = logging.getLogger(__name__)

# Default analysis configuration
DEFAULT_DATA_PATH = Path("F:/GMCP_Precipitation")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures" / "gmcp_analysis"
DEFAULT_N_WINDOWS = 300
DEFAULT_SEED = 42

# Extreme precipitation thresholds (mm/6h) used by the project.
EXTREME_THRESHOLDS = [25.0, 50.0, 100.0]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GMCP precipitation exploratory analysis."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Root directory containing GMCP data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save figures and summary JSON.",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=DEFAULT_N_WINDOWS,
        help="Number of 6-hour windows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def audit_coverage(finder: GMCPFileFinder) -> dict[str, Any]:
    """Count available files and report coverage by year/month.

    Returns:
        Dictionary with total file count, per-year counts, and a list of
        missing (year, month) tuples where fewer than expected hours exist.
    """
    logger.info("Starting GMCP coverage audit...")
    files = finder.find_files()

    total = len(files)
    per_year: dict[int, int] = defaultdict(int)
    per_month: dict[tuple[int, int], int] = defaultdict(int)

    for file_info in files:
        per_year[file_info.year] += 1
        per_month[(file_info.year, file_info.time.month)] += 1

    missing: list[dict[str, Any]] = []
    for year in sorted(per_year.keys()):
        for month in range(1, 13):
            expected_hours = _hours_in_month(year, month)
            actual = per_month.get((year, month), 0)
            if actual != expected_hours:
                missing.append(
                    {
                        "year": year,
                        "month": month,
                        "expected": expected_hours,
                        "actual": actual,
                    }
                )

    coverage = {
        "total_files": total,
        "year_range": [min(per_year.keys()), max(per_year.keys())],
        "per_year": dict(sorted(per_year.items())),
        "missing_months": missing,
    }
    logger.info(
        "Coverage audit complete: %d files across %d years",
        total,
        len(per_year),
    )
    return coverage


def _hours_in_month(year: int, month: int) -> int:
    """Return the number of hours in a given month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1)
    return int((end - start).total_seconds() // 3600)


def sample_6h_windows(
    finder: GMCPFileFinder,
    n_windows: int,
    seed: int,
) -> list[list[GMCPFile]]:
    """Sample independent 6-hour windows from the full GMCP record.

    Each window consists of six consecutive hourly files. Windows are
    stratified by year so that inter-annual variability is represented.

    Args:
        finder: File finder for GMCP data.
        n_windows: Number of 6-hour windows to sample.
        seed: Random seed.

    Returns:
        List of windows, where each window is a list of six ``GMCPFile``
        objects ordered in time.
    """
    files = finder.find_files()
    if len(files) < 6:
        return []

    # Build windows: each valid starting index gives one 6-hour window.
    by_year: dict[int, list[int]] = defaultdict(list)
    for idx, file_info in enumerate(files[:-5]):
        by_year[file_info.year].append(idx)

    years = sorted(by_year.keys())
    rng = np.random.default_rng(seed)

    base = n_windows // len(years)
    remainder = n_windows % len(years)
    selected_indices: set[int] = set()

    for idx, year in enumerate(years):
        n = base + (1 if idx < remainder else 0)
        year_indices = by_year[year]
        if n >= len(year_indices):
            selected_indices.update(year_indices)
        else:
            chosen = rng.choice(year_indices, size=n, replace=False)
            selected_indices.update(chosen)

    windows = []
    for start in sorted(selected_indices):
        window = files[start : start + 6]
        # Verify the window is truly consecutive in time.
        expected_times = [
            window[0].time + pd.Timedelta(hours=h) for h in range(6)
        ]
        if [f.time for f in window] == expected_times:
            windows.append(window)

    logger.info(
        "Sampled %d valid consecutive 6-hour windows across %d years",
        len(windows),
        len(years),
    )
    return windows


def load_windows(
    finder: GMCPFileFinder,
    windows: list[list[GMCPFile]],
    batch_size: int = 50,
) -> xr.Dataset:
    """Load sampled 6-hour windows and compute cumulative precipitation.

    Args:
        finder: File finder (used for path context).
        windows: List of 6-hour windows.
        batch_size: Number of windows to process per batch.

    Returns:
        Dataset with variable ``precipitation_rate`` representing 6-hourly
        cumulative precipitation over China.
    """
    logger.info(
        "Loading %d 6-hour windows in batches of %d...",
        len(windows),
        batch_size,
    )
    dataset = GMCPDataset(
        finder.data_path,
        region=CHINA_REGION,
    )

    cumulative_fields: list[xr.DataArray] = []
    for batch_start in range(0, len(windows), batch_size):
        batch = windows[batch_start : batch_start + batch_size]
        logger.info(
            "Loading batch %d/%d",
            batch_start // batch_size + 1,
            (len(windows) - 1) // batch_size + 1,
        )

        for window in batch:
            ds = dataset.load_files(window)
            cumulative = ds["precipitation_rate"].sum(dim="time", keepdims=False)
            cumulative["time"] = pd.Timestamp(window[-1].time)
            cumulative_fields.append(cumulative)

    stacked = xr.concat(
        [da.expand_dims("time") for da in cumulative_fields],
        dim="time",
    )
    out = stacked.to_dataset(name="precipitation_rate")
    out = out.sortby("time")
    logger.info(
        "Loaded 6-hourly cumulative data: %d windows, shape: %s",
        len(out.time),
        out["precipitation_rate"].shape,
    )
    return out


def compute_quality_control_summary(ds: xr.Dataset) -> dict[str, Any]:
    """Compute QC statistics for 6-hourly cumulative GMCP data.

    Returns:
        Dictionary with missing-value count, negative-value count,
        out-of-range counts, and basic descriptive statistics.
    """
    precip = ds["precipitation_rate"]

    missing = int(precip.isnull().sum().values)
    negative = int((precip < 0).sum().values)
    above_200 = int((precip > 200.0).sum().values)

    return {
        "total_grid_points": int(precip.size),
        "missing_values": missing,
        "negative_values": negative,
        "values_above_200mm_6h": above_200,
        "max_mm_6h": float(precip.max().values),
        "mean_mm_6h": float(precip.mean().values),
    }


def compute_extreme_frequencies(ds: xr.Dataset) -> dict[str, float]:
    """Compute frequency of extreme precipitation events at thresholds."""
    precip = ds["precipitation_rate"]
    flat = precip.values.ravel()
    flat = flat[np.isfinite(flat)]

    if len(flat) == 0:
        return {f">={t}mm_6h": 0.0 for t in EXTREME_THRESHOLDS}

    return {
        f">={t}mm_6h": float((flat >= t).sum() / len(flat))
        for t in EXTREME_THRESHOLDS
    }


def plot_mean_precipitation_map(ds: xr.Dataset, save_path: Path) -> None:
    """Plot the mean 6-hourly cumulative precipitation map over China."""
    mean_precip = ds["precipitation_rate"].mean(dim="time")

    fig, ax = plt.subplots(figsize=(12, 8))
    mean_precip.plot(
        ax=ax,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=10.0,
        cbar_kwargs={"label": "Mean 6-hourly cumulative precipitation (mm)"},
    )
    ax.set_title("GMCP Mean 6-Hourly Cumulative Precipitation over China")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mean precipitation map to %s", save_path)


def plot_seasonal_cycle(ds: xr.Dataset, save_path: Path) -> None:
    """Plot the seasonal cycle of 6-hourly cumulative precipitation."""
    monthly = ds["precipitation_rate"].groupby("time.month").mean()
    months = monthly.month.values
    values = monthly.mean(dim=["latitude", "longitude"]).values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(months, values, marker="o", linewidth=2)
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean 6-hourly cumulative precipitation (mm)")
    ax.set_title("GMCP Seasonal Cycle over China (Sampled)")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved seasonal cycle plot to %s", save_path)


def plot_diurnal_cycle(ds: xr.Dataset, save_path: Path) -> None:
    """Plot the diurnal cycle of 6-hourly cumulative precipitation."""
    hourly = ds["precipitation_rate"].groupby("time.hour").mean()
    hours = hourly.hour.values
    values = hourly.mean(dim=["latitude", "longitude"]).values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, values, marker="o", linewidth=2)
    ax.set_xticks(hours[::2])
    ax.set_xlabel("Ending hour (UTC)")
    ax.set_ylabel("Mean 6-hourly cumulative precipitation (mm)")
    ax.set_title("GMCP Diurnal Cycle over China (Sampled)")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diurnal cycle plot to %s", save_path)


def plot_intensity_histogram(ds: xr.Dataset, save_path: Path) -> None:
    """Plot the histogram of 6-hourly cumulative precipitation intensity."""
    flat = ds["precipitation_rate"].values.ravel()
    flat = flat[np.isfinite(flat) & (flat > 0)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(flat, bins=100, range=(0, 50), color="steelblue", edgecolor="white")
    axes[0].set_xlabel("6-hourly cumulative precipitation (mm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Intensity Distribution (0-50 mm/6h)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        flat,
        bins=np.logspace(-2, 2, 100),
        color="steelblue",
        edgecolor="white",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("6-hourly cumulative precipitation (mm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Intensity Distribution (Log-Log)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved intensity histogram to %s", save_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    finder = GMCPFileFinder(args.data_path)
    coverage = audit_coverage(finder)

    windows = sample_6h_windows(finder, args.n_windows, args.seed)
    if not windows:
        logger.error("No valid GMCP windows found. Aborting analysis.")
        return

    ds = load_windows(finder, windows, batch_size=50)

    qc_summary = compute_quality_control_summary(ds)
    extreme_freqs = compute_extreme_frequencies(ds)

    plot_mean_precipitation_map(
        ds, args.output_dir / "gmcp_mean_precipitation_china.png"
    )
    plot_seasonal_cycle(ds, args.output_dir / "gmcp_seasonal_cycle.png")
    plot_diurnal_cycle(ds, args.output_dir / "gmcp_diurnal_cycle.png")
    plot_intensity_histogram(ds, args.output_dir / "gmcp_intensity_histogram.png")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "data_path": str(args.data_path),
        "n_windows": len(windows),
        "coverage": coverage,
        "quality_control": qc_summary,
        "extreme_frequencies": extreme_freqs,
    }

    summary_path = args.output_dir / "gmcp_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Analysis complete. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
