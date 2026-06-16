"""
ERA5 Data Preprocessing Script for PhyDiff-Net
================================================

Preprocesses ERA5 reanalysis data into PyTorch-compatible tensors for
precipitation forecasting model training.

Features:
    - Reads NetCDF format ERA5 data
    - Z-score normalization (mean=0, std=1)
    - Time series organization: history N frames -> future M frames
    - Optional spatial cropping
    - Dataset splitting: train / validation / test
    - Output as PyTorch .pt tensor files

Usage:
    python scripts/preprocess_era5.py --config configs/preprocess_config.yaml
    python scripts/preprocess_era5.py --era5_path F:/ERA5 --output_dir F:/ERA5/processed

Author: weather-model-trainer
Date: 2026-06-16
"""

import sys
sys.path.insert(0, "e:/weather")

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loader
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load preprocessing configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# ERA5 Data Loader
# =============================================================================

class ERA5NetCDFLoader:
    """Loader for ERA5 NetCDF data files."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ERA5 loader.

        Args:
            config: Configuration dictionary with input settings.
        """
        self.era5_path = Path(config["input"]["era5_path"])
        self.variables = config["input"]["variables"]
        self.resolution = config["input"]["era5_resolution"]
        self.pressure_levels = config["input"].get("pressure_levels", [])

    def load_dataset(
        self,
        start_date: str,
        end_date: str,
    ) -> xr.Dataset:
        """Load ERA5 data for specified time range.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            xarray Dataset with ERA5 variables.
        """
        logger.info(
            "Loading ERA5 data from %s to %s", start_date, end_date
        )

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        datasets = []
        current = start

        while current <= end:
            year = current.year
            month = current.month

            # Try multiple file naming conventions
            file_patterns = [
                f"era5_{year}_{month:02d}.nc",
                f"ERA5_{year}_{month:02d}.nc",
                f"era5_{year}{month:02d}.nc",
            ]

            for pattern in file_patterns:
                file_path = self.era5_path / pattern
                if file_path.exists():
                    ds = xr.open_dataset(str(file_path))

                    # Filter time range
                    ds = ds.sel(
                        time=slice(
                            current.strftime("%Y-%m-%d"),
                            end.strftime("%Y-%m-%d"),
                        )
                    )

                    # Filter pressure levels if specified
                    if self.pressure_levels and "level" in ds.dims:
                        ds = ds.sel(
                            level=[
                                l for l in self.pressure_levels
                                if l in ds.level.values
                            ]
                        )

                    datasets.append(ds)
                    logger.info("Loaded: %s", file_path.name)
                    break

            # Move to next month
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        if not datasets:
            raise FileNotFoundError(
                f"No ERA5 data files found for {start_date} to {end_date} "
                f"in {self.era5_path}"
            )

        # Merge all datasets
        merged = xr.concat(datasets, dim="time")

        # Select available variables
        available_vars = [
            v for v in self.variables if v in merged.data_vars
        ]
        if not available_vars:
            available_vars = list(merged.data_vars)[:len(self.variables)]

        merged = merged[available_vars]

        logger.info(
            "Loaded ERA5: %d time steps, variables: %s, shape: %s",
            len(merged.time),
            list(merged.data_vars),
            {k: v.shape for k, v in merged.data_vars.items()},
        )

        return merged


# =============================================================================
# Quality Control
# =============================================================================

class QualityController:
    """Data quality control processor."""

    def __init__(self, thresholds: Dict[str, Dict[str, float]]) -> None:
        """Initialize quality controller.

        Args:
            thresholds: Physical thresholds for each variable.
        """
        self.thresholds = thresholds

    def control(self, data: xr.Dataset) -> xr.Dataset:
        """Apply quality control to dataset.

        Args:
            data: Input xarray Dataset.

        Returns:
            Quality-controlled Dataset.
        """
        logger.info("Applying quality control")
        data = data.copy()

        for var in data.data_vars:
            if var not in self.thresholds:
                continue

            var_data = data[var]
            threshold = self.thresholds[var]

            # Check for NaN values
            nan_count = int(var_data.isnull().sum().values)
            if nan_count > 0:
                logger.warning(
                    "Variable '%s': %d NaN values found", var, nan_count
                )

            # Clip to physical range
            out_of_range = int(
                ((var_data < threshold["min"]) | (var_data > threshold["max"])).sum().values
            )
            if out_of_range > 0:
                logger.warning(
                    "Variable '%s': %d values outside [%.1f, %.1f]",
                    var, out_of_range, threshold["min"], threshold["max"]
                )
                var_data = var_data.clip(
                    min=threshold["min"], max=threshold["max"]
                )

            data[var] = var_data

        return data


# =============================================================================
# Spatial Processor
# =============================================================================

class SpatialProcessor:
    """Spatial cropping and interpolation processor."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize spatial processor.

        Args:
            config: Configuration with spatial settings.
        """
        spatial_config = config["spatial"]
        self.target_resolution = spatial_config["target_resolution"]
        self.crop_config = spatial_config.get("crop", {})

    def process(self, data: xr.Dataset) -> xr.Dataset:
        """Apply spatial processing.

        Args:
            data: Input xarray Dataset.

        Returns:
            Spatially processed Dataset.
        """
        logger.info("Applying spatial processing")

        # Apply spatial cropping if enabled
        if self.crop_config.get("enabled", False):
            data = self._crop_spatial(data)

        # Interpolate to target resolution if needed
        current_res = abs(float(data.latitude[1] - data.latitude[0]))
        if abs(current_res - self.target_resolution) > 0.001:
            data = self._interpolate_resolution(data)

        logger.info(
            "Spatial processing complete: shape=%s",
            {k: v.shape for k, v in data.data_vars.items()},
        )

        return data

    def _crop_spatial(self, data: xr.Dataset) -> xr.Dataset:
        """Crop data to specified spatial bounds.

        Args:
            data: Input xarray Dataset.

        Returns:
            Cropped Dataset.
        """
        lat_min = self.crop_config["lat_min"]
        lat_max = self.crop_config["lat_max"]
        lon_min = self.crop_config["lon_min"]
        lon_max = self.crop_config["lon_max"]

        logger.info(
            "Cropping to lat=[%.1f, %.1f], lon=[%.1f, %.1f]",
            lat_min, lat_max, lon_min, lon_max
        )

        cropped = data.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )

        return cropped

    def _interpolate_resolution(self, data: xr.Dataset) -> xr.Dataset:
        """Interpolate to target spatial resolution.

        Args:
            data: Input xarray Dataset.

        Returns:
            Interpolated Dataset.
        """
        logger.info(
            "Interpolating from ~%.2f to %.2f degrees",
            abs(float(data.latitude[1] - data.latitude[0])),
            self.target_resolution,
        )

        target_lats = np.arange(
            float(data.latitude.max()),
            float(data.latitude.min()),
            -self.target_resolution,
        )
        target_lons = np.arange(
            float(data.longitude.min()),
            float(data.longitude.max()),
            self.target_resolution,
        )

        interpolated = data.interp(
            latitude=target_lats,
            longitude=target_lons,
            method="linear",
        )

        return interpolated


# =============================================================================
# Temporal Processor
# =============================================================================

class TemporalProcessor:
    """Temporal alignment and sequence creation processor."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize temporal processor.

        Args:
            config: Configuration with temporal settings.
        """
        temporal_config = config["temporal"]
        self.original_resolution = temporal_config["original_resolution"]
        self.target_resolution = temporal_config["target_resolution"]
        self.history_length = temporal_config["sequence"]["history_length"]
        self.forecast_length = temporal_config["sequence"]["forecast_length"]
        self.aggregation_method = temporal_config.get("aggregation_method", "mean")

    def align_temporal(self, data: xr.Dataset) -> xr.Dataset:
        """Align data to target temporal resolution.

        Args:
            data: Input xarray Dataset.

        Returns:
            Temporally aligned Dataset.
        """
        logger.info(
            "Aligning to %d-hourly resolution", self.target_resolution
        )

        if len(data.time) < 2:
            return data

        # Calculate current resolution
        time_diff = data.time.diff("time").median().values
        current_hours = int(time_diff / np.timedelta64(1, "h"))

        if current_hours == self.target_resolution:
            logger.info("Data already at target resolution")
            return data

        # Resample based on aggregation method
        if self.aggregation_method == "sum":
            resampled = data.resample(
                time=f"{self.target_resolution}h"
            ).sum()
        elif self.aggregation_method == "max":
            resampled = data.resample(
                time=f"{self.target_resolution}h"
            ).max()
        else:  # mean
            resampled = data.resample(
                time=f"{self.target_resolution}h"
            ).mean()

        logger.info(
            "Temporal alignment complete: %d -> %d time steps",
            len(data.time), len(resampled.time)
        )

        return resampled

    def create_sequences(
        self, data: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-target sequences for seq2seq forecasting.

        Args:
            data: Temporally aligned xarray Dataset.

        Returns:
            Tuple of (inputs, targets) numpy arrays.
            - inputs shape: (N, T_history, C, H, W)
            - targets shape: (N, T_forecast, C, H, W)
            where N=samples, C=channels, H=height, W=width
        """
        total_length = self.history_length + self.forecast_length
        n_timesteps = len(data.time)

        if n_timesteps < total_length:
            raise ValueError(
                f"Not enough time steps ({n_timesteps}) for sequence "
                f"length ({total_length})"
            )

        # Convert to numpy array: (T, C, H, W)
        var_names = list(data.data_vars)
        arrays = []
        for var in var_names:
            arr = data[var].values
            if arr.ndim == 3:  # (T, H, W) -> (T, 1, H, W)
                arr = arr[:, np.newaxis, :, :]
            arrays.append(arr)

        # Stack variables along channel dimension: (T, C, H, W)
        full_array = np.concatenate(arrays, axis=1)

        # Replace NaN with 0
        full_array = np.nan_to_num(full_array, nan=0.0)

        n_samples = n_timesteps - total_length + 1
        inputs = np.zeros(
            (n_samples, self.history_length) + full_array.shape[1:],
            dtype=np.float32,
        )
        targets = np.zeros(
            (n_samples, self.forecast_length) + full_array.shape[1:],
            dtype=np.float32,
        )

        for i in range(n_samples):
            inputs[i] = full_array[i: i + self.history_length]
            targets[i] = full_array[
                i + self.history_length: i + total_length
            ]

        logger.info(
            "Created %d sequences: inputs=%s, targets=%s",
            n_samples, inputs.shape, targets.shape
        )

        return inputs, targets


# =============================================================================
# Normalizer
# =============================================================================

class ZScoreNormalizer:
    """Z-score normalization (mean=0, std=1)."""

    def __init__(
        self, clip_sigma: float = 5.0, compute_from_train: bool = True
    ) -> None:
        """Initialize normalizer.

        Args:
            clip_sigma: Clip values beyond this many standard deviations.
            compute_from_train: If True, compute stats from training data only.
        """
        self.clip_sigma = clip_sigma
        self.compute_from_train = compute_from_train
        self.stats: Dict[str, Dict[str, float]] = {}

    def fit(self, data: np.ndarray, var_names: List[str]) -> None:
        """Compute normalization statistics.

        Args:
            data: Training data array (N, T, C, H, W).
            var_names: List of variable names.
        """
        logger.info("Computing normalization statistics from training data")

        # Reshape to (N*T, C, H, W) for statistics
        n_samples, n_time, n_channels = data.shape[:3]
        reshaped = data.reshape(-1, n_channels, data.shape[3], data.shape[4])

        for c, name in enumerate(var_names):
            channel_data = reshaped[:, c, :, :].flatten()
            # Remove zeros (from NaN fill) for statistics
            valid_mask = channel_data != 0
            if valid_mask.sum() > 0:
                valid_data = channel_data[valid_mask]
                mean_val = float(np.mean(valid_data))
                std_val = float(np.std(valid_data))
                std_val = max(std_val, 1e-8)  # Avoid division by zero
            else:
                mean_val = 0.0
                std_val = 1.0

            self.stats[name] = {
                "mean": mean_val,
                "std": std_val,
            }

            logger.info(
                "Variable '%s': mean=%.4f, std=%.4f",
                name, mean_val, std_val
            )

    def transform(
        self, data: np.ndarray, var_names: List[str]
    ) -> np.ndarray:
        """Apply normalization to data.

        Args:
            data: Input array (N, T, C, H, W).
            var_names: List of variable names matching channels.

        Returns:
            Normalized array.
        """
        normalized = data.copy()

        for c, name in enumerate(var_names):
            if name not in self.stats:
                logger.warning(
                    "No stats for variable '%s', skipping normalization", name
                )
                continue

            mean = self.stats[name]["mean"]
            std = self.stats[name]["std"]

            normalized[:, :, c, :, :] = (
                (data[:, :, c, :, :] - mean) / std
            )

        # Clip outliers
        if self.clip_sigma is not None:
            normalized = np.clip(
                normalized, -self.clip_sigma, self.clip_sigma
            )

        return normalized.astype(np.float32)

    def inverse_transform(
        self, data: np.ndarray, var_names: List[str]
    ) -> np.ndarray:
        """Apply inverse normalization.

        Args:
            data: Normalized array.
            var_names: List of variable names matching channels.

        Returns:
            Denormalized array.
        """
        denormalized = data.copy()

        for c, name in enumerate(var_names):
            if name not in self.stats:
                continue

            mean = self.stats[name]["mean"]
            std = self.stats[name]["std"]

            denormalized[:, :, c, :, :] = (
                data[:, :, c, :, :] * std + mean
            )

        return denormalized

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics.

        Returns:
            Dictionary of statistics per variable.
        """
        return self.stats

    def save_stats(self, path: str) -> None:
        """Save statistics to JSON file.

        Args:
            path: Output file path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        logger.info("Normalization stats saved to %s", path)

    def load_stats(self, path: str) -> None:
        """Load statistics from JSON file.

        Args:
            path: Input file path.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        logger.info("Normalization stats loaded from %s", path)


# =============================================================================
# Dataset Splitter
# =============================================================================

class DatasetSplitter:
    """Split data into train/val/test sets."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize splitter.

        Args:
            config: Configuration with split settings.
        """
        split_config = config["split"]
        self.train_period = split_config["train_period"]
        self.val_period = split_config["val_period"]
        self.test_period = split_config["test_period"]
        self.random_seed = split_config.get("random_seed", 42)

    def split_temporal(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        time_index: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data by time periods.

        Args:
            inputs: Input sequences (N, T, C, H, W).
            targets: Target sequences (N, T, C, H, W).
            time_index: Corresponding time stamps.

        Returns:
            Dictionary with 'train', 'val', 'test' splits.
        """
        logger.info("Splitting dataset by time periods")

        splits = {
            "train": self._filter_period(
                inputs, targets, time_index,
                self.train_period["start"],
                self.train_period["end"],
            ),
            "val": self._filter_period(
                inputs, targets, time_index,
                self.val_period["start"],
                self.val_period["end"],
            ),
            "test": self._filter_period(
                inputs, targets, time_index,
                self.test_period["start"],
                self.test_period["end"],
            ),
        }

        for name, (inp, tgt) in splits.items():
            logger.info(
                "Split '%s': %d samples, inputs=%s, targets=%s",
                name, inp.shape[0], inp.shape, tgt.shape
            )

        return splits

    def _filter_period(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        time_index: np.ndarray,
        start_date: str,
        end_date: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter data to specific time period.

        Args:
            inputs: Input sequences.
            targets: Target sequences.
            time_index: Time stamps.
            start_date: Period start date.
            end_date: Period end date.

        Returns:
            Filtered (inputs, targets) tuple.
        """
        start = np.datetime64(start_date)
        end = np.datetime64(end_date)

        # Find indices where the input window starts within the period
        mask = (time_index >= start) & (time_index <= end)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            logger.warning(
                "No data found for period %s to %s", start_date, end_date
            )
            return np.array([]), np.array([])

        return inputs[indices], targets[indices]


# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================

class ERA5PreprocessingPipeline:
    """Complete ERA5 preprocessing pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize pipeline.

        Args:
            config: Full preprocessing configuration.
        """
        self.config = config

        # Initialize components
        self.loader = ERA5NetCDFLoader(config)
        self.quality_controller = QualityController(
            config["quality_control"]
        )
        self.spatial_processor = SpatialProcessor(config)
        self.temporal_processor = TemporalProcessor(config)
        self.normalizer = ZScoreNormalizer(
            clip_sigma=config["normalization"]["clip_sigma"],
            compute_from_train=config["normalization"]["compute_from_train_only"],
        )
        self.splitter = DatasetSplitter(config)

        # Output settings
        self.output_dir = Path(config["output"]["output_dir"])
        self.output_format = config["output"]["format"]
        self.save_stats_flag = config["output"]["save_stats"]

    def run(self) -> None:
        """Run the complete preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("ERA5 Preprocessing Pipeline")
        logger.info("=" * 60)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each split
        for split_name in ["train", "val", "test"]:
            period = self.config["split"][f"{split_name}_period"]
            logger.info("\n--- Processing %s set ---", split_name.upper())

            self._process_split(
                split_name,
                period["start"],
                period["end"],
            )

        logger.info("\n" + "=" * 60)
        logger.info("Preprocessing complete!")
        logger.info("Output saved to: %s", self.output_dir)
        logger.info("=" * 60)

    def _process_split(
        self, split_name: str, start_date: str, end_date: str
    ) -> None:
        """Process a single data split.

        Args:
            split_name: Name of the split ('train', 'val', 'test').
            start_date: Start date for this split.
            end_date: End date for this split.
        """
        # 1. Load data
        logger.info("Step 1: Loading ERA5 data")
        raw_data = self.loader.load_dataset(start_date, end_date)

        # 2. Quality control
        logger.info("Step 2: Quality control")
        clean_data = self.quality_controller.control(raw_data)

        # 3. Spatial processing
        logger.info("Step 3: Spatial processing")
        spatial_data = self.spatial_processor.process(clean_data)

        # 4. Temporal alignment
        logger.info("Step 4: Temporal alignment")
        temporal_data = self.temporal_processor.align_temporal(spatial_data)

        # 5. Create sequences
        logger.info("Step 5: Creating sequences")
        inputs, targets = self.temporal_processor.create_sequences(temporal_data)

        # 6. Normalization
        logger.info("Step 6: Normalization")
        var_names = list(temporal_data.data_vars)

        if split_name == "train":
            # Fit normalizer on training data only
            self.normalizer.fit(inputs, var_names)

            # Save normalization stats
            if self.save_stats_flag:
                stats_path = self.output_dir / "normalization_stats.json"
                self.normalizer.save_stats(str(stats_path))

        # Apply normalization
        inputs_norm = self.normalizer.transform(inputs, var_names)
        targets_norm = self.normalizer.transform(targets, var_names)

        # 7. Save as PyTorch tensors
        logger.info("Step 7: Saving tensors")
        self._save_tensors(
            split_name, inputs_norm, targets_norm, var_names
        )

        # 8. Validation
        if self.config["processing"].get("validate_output", True):
            logger.info("Step 8: Validating output")
            self._validate_output(inputs_norm, targets_norm, split_name)

    def _save_tensors(
        self,
        split_name: str,
        inputs: np.ndarray,
        targets: np.ndarray,
        var_names: List[str],
    ) -> None:
        """Save processed data as PyTorch tensors.

        Args:
            split_name: Name of the split.
            inputs: Normalized input array.
            targets: Normalized target array.
            var_names: List of variable names.
        """
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Convert to PyTorch tensors
        inputs_tensor = torch.from_numpy(inputs)
        targets_tensor = torch.from_numpy(targets)

        # Create metadata
        metadata = {
            "split": split_name,
            "variables": var_names,
            "history_length": self.temporal_processor.history_length,
            "forecast_length": self.temporal_processor.forecast_length,
            "spatial_shape": list(inputs.shape[-2:]),
            "n_channels": inputs.shape[2],
            "n_samples": inputs.shape[0],
            "normalization": self.config["normalization"]["method"],
            "normalization_stats": self.normalizer.get_stats(),
        }

        # Save tensors
        prefix = self.config["output"]["naming"]["prefix"]
        sep = self.config["output"]["naming"]["separator"]

        inputs_path = split_dir / f"{prefix}{sep}{split_name}{sep}inputs.pt"
        targets_path = split_dir / f"{prefix}{sep}{split_name}{sep}targets.pt"
        metadata_path = split_dir / f"{prefix}{sep}{split_name}{sep}metadata.json"

        torch.save(inputs_tensor, inputs_path)
        torch.save(targets_tensor, targets_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            "Saved %s split: inputs=%s, targets=%s",
            split_name, inputs_path, targets_path
        )

    def _validate_output(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        split_name: str,
    ) -> None:
        """Validate processed output.

        Args:
            inputs: Input array.
            targets: Target array.
            split_name: Name of the split.
        """
        # Check for NaN
        nan_count = int(np.isnan(inputs).sum() + np.isnan(targets).sum())
        if nan_count > 0:
            logger.warning(
                "%s: %d NaN values found in output", split_name, nan_count
            )

        # Check value range
        input_min = float(inputs.min())
        input_max = float(inputs.max())
        target_min = float(targets.min())
        target_max = float(targets.max())

        logger.info(
            "%s validation: inputs=[%.3f, %.3f], targets=[%.3f, %.3f]",
            split_name, input_min, input_max, target_min, target_max
        )

        # Check shapes
        assert inputs.ndim == 5, f"Expected 5D input, got {inputs.ndim}D"
        assert targets.ndim == 5, f"Expected 5D target, got {targets.ndim}D"
        assert inputs.shape[2] == targets.shape[2], (
            f"Channel mismatch: inputs={inputs.shape[2]}, "
            f"targets={targets.shape[2]}"
        )

        logger.info("%s validation passed", split_name)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for ERA5 preprocessing."""
    parser = argparse.ArgumentParser(
        description="ERA5 Data Preprocessing for PhyDiff-Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use config file
    python scripts/preprocess_era5.py --config configs/preprocess_config.yaml

    # Override paths
    python scripts/preprocess_era5.py \\
        --era5_path F:/ERA5 \\
        --output_dir F:/ERA5/processed

    # Process specific time range
    python scripts/preprocess_era5.py \\
        --config configs/preprocess_config.yaml \\
        --start_date 2000-01-01 \\
        --end_date 2010-12-31
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocess_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--era5_path",
        type=str,
        default=None,
        help="ERA5 data directory (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (overrides config)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (overrides config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    if args.era5_path:
        config["input"]["era5_path"] = args.era5_path
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.start_date:
        for split in ["train", "val", "test"]:
            config["split"][f"{split}_period"]["start"] = args.start_date
    if args.end_date:
        for split in ["train", "val", "test"]:
            config["split"][f"{split}_period"]["end"] = args.end_date

    # Run pipeline
    pipeline = ERA5PreprocessingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
