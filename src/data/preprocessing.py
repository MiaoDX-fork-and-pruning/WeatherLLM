"""
Weather Data Preprocessing Pipeline
====================================

ERA5和GMCP数据的预处理管道，包括：
- ERA5数据加载和预处理
- GMCP数据加载和预处理
- 空间配准（将ERA5插值到0.1°）
- 时间对齐（统一到6小时时间步长）
- 归一化处理
- 数据质量控制

Author: weather-model-trainer
Date: 2026-06-15
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration Constants
# ==============================================================================

# 物理阈值常量
PHYSICAL_THRESHOLDS = {
    "temperature": {"min": 150.0, "max": 350.0},  # Kelvin
    "u_wind": {"min": -100.0, "max": 100.0},  # m/s
    "v_wind": {"min": -100.0, "max": 100.0},  # m/s
    "geopotential": {"min": -5000.0, "max": 100000.0},  # m²/s²
    "relative_humidity": {"min": 0.0, "max": 100.0},  # %
    "surface_pressure": {"min": 30000.0, "max": 110000.0},  # Pa
    "total_precipitation": {"min": 0.0, "max": 500.0},  # mm
    "precipitation_rate": {"min": 0.0, "max": 200.0},  # mm/6h
}


class ERA5Preprocessor:
    """ERA5再分析数据预处理器。

    负责加载ERA5数据，执行质量控制，进行空间插值和时间对齐。

    Attributes:
        data_path: ERA5数据根目录路径。
        resolution: ERA5原始分辨率（度）。
        variables: 要加载的ERA5变量列表。
    """

    def __init__(
        self,
        data_path: str,
        resolution: float = 0.25,
        variables: Optional[List[str]] = None,
    ) -> None:
        """初始化ERA5预处理器。

        Args:
            data_path: ERA5数据根目录路径。
            resolution: ERA5原始分辨率（度），默认0.25。
            variables: 要加载的ERA5变量列表，默认为None。
        """
        self.data_path = Path(data_path)
        self.resolution = resolution
        self.variables = variables or [
            "temperature",
            "u_wind",
            "v_wind",
            "geopotential",
            "relative_humidity",
            "surface_pressure",
            "total_precipitation",
        ]

        # 计算ERA5网格
        self.lats = np.arange(90.0, -90.0, -self.resolution)
        self.lons = np.arange(0.0, 360.0, self.resolution)

    def load_data(
        self,
        start_date: str,
        end_date: str,
        pressure_level: Optional[int] = None,
    ) -> xr.Dataset:
        """加载指定时间范围的ERA5数据。

        Args:
            start_date: 开始日期，格式'YYYY-MM-DD'。
            end_date: 结束日期，格式'YYYY-MM-DD'。
            pressure_level: 气压层（hPa），如500, 850等。为None时加载所有层。

        Returns:
            包含ERA5变量的xarray Dataset。

        Raises:
            FileNotFoundError: 当数据文件不存在时。
            ValueError: 当变量名无效时。
        """
        logger.info(
            "Loading ERA5 data from %s to %s", start_date, end_date
        )

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # 收集所有时间步的数据文件
        datasets = []
        current = start
        while current <= end:
            year = current.year
            month = current.month

            # 构造文件路径（假设按年存储）
            file_path = self.data_path / f"era5_{year}_{month:02d}.nc"
            if file_path.exists():
                ds = xr.open_dataset(str(file_path))
                # 筛选时间范围
                ds = ds.sel(
                    time=slice(
                        current.strftime("%Y-%m-%d"),
                        end.strftime("%Y-%m-%d"),
                    )
                )
                if pressure_level is not None and "level" in ds.dims:
                    ds = ds.sel(level=pressure_level)
                datasets.append(ds)

            # 移动到下个月
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        if not datasets:
            raise FileNotFoundError(
                f"No ERA5 data files found for period {start_date} to {end_date}"
            )

        # 合并所有数据
        merged = xr.concat(datasets, dim="time")

        # 选择需要的变量
        available_vars = [v for v in self.variables if v in merged.data_vars]
        merged = merged[available_vars]

        logger.info(
            "Loaded ERA5 data: %d time steps, variables: %s",
            len(merged.time),
            list(merged.data_vars),
        )

        return merged

    def quality_control(self, data: xr.Dataset) -> xr.Dataset:
        """执行ERA5数据质量控制。

        包括缺失值处理和异常值检测。

        Args:
            data: 输入的ERA5 Dataset。

        Returns:
            经过质量控制的Dataset。
        """
        logger.info("Performing ERA5 quality control")

        data = data.copy()

        for var in data.data_vars:
            if var not in PHYSICAL_THRESHOLDS:
                continue

            threshold = PHYSICAL_THRESHOLDS[var]
            var_data = data[var]

            # 1. 检测并标记缺失值
            missing_count = var_data.isnull().sum().values
            if missing_count > 0:
                logger.warning(
                    "Variable '%s' has %d missing values", var, missing_count
                )
                # 使用时空插值填充缺失值
                var_data = self._interpolate_missing(var_data)

            # 2. 异常值检测（基于物理阈值）
            out_of_range = (var_data < threshold["min"]) | (
                var_data > threshold["max"]
            )
            outlier_count = out_of_range.sum().values
            if outlier_count > 0:
                logger.warning(
                    "Variable '%s' has %d values outside physical range "
                    "[%.1f, %.1f]",
                    var,
                    outlier_count,
                    threshold["min"],
                    threshold["max"],
                )
                # 将异常值设置为NaN，然后插值
                var_data = var_data.where(~out_of_range)
                var_data = self._interpolate_missing(var_data)

            # 3. 检测统计异常（3 sigma）
            mean_val = var_data.mean()
            std_val = var_data.std()
            statistical_outliers = (
                abs(var_data - mean_val) > 3 * std_val
            )
            stat_outlier_count = statistical_outliers.sum().values
            if stat_outlier_count > 0:
                logger.info(
                    "Variable '%s': %d statistical outliers detected",
                    var,
                    stat_outlier_count,
                )

            data[var] = var_data

        return data

    def _interpolate_missing(self, data: xr.DataArray) -> xr.DataArray:
        """使用时空插值填充缺失值。

        Args:
            data: 包含缺失值的数据数组。

        Returns:
            填充缺失值后的数据数组。
        """
        # 先进行时间维度的线性插值
        data = data.interpolate_na(dim="time", method="linear")

        # 空间维度的最近邻插值
        if data.isnull().any():
            data = data.interpolate_na(
                dim="latitude", method="nearest"
            )
            data = data.interpolate_na(
                dim="longitude", method="nearest"
            )

        return data

    def spatial_interpolation(
        self, data: xr.Dataset, target_resolution: float = 0.1
    ) -> xr.Dataset:
        """将ERA5数据插值到目标分辨率。

        使用双线性插值将0.25°数据重采样到0.1°。

        Args:
            data: 输入的ERA5 Dataset。
            target_resolution: 目标分辨率（度），默认0.1。

        Returns:
            重采样后的Dataset。
        """
        logger.info(
            "Interpolating ERA5 from %.2f to %.2f degrees",
            self.resolution,
            target_resolution,
        )

        # 创建目标网格
        target_lats = np.arange(
            data.latitude.max().values,
            data.latitude.min().values,
            -target_resolution,
        )
        target_lons = np.arange(
            data.longitude.min().values,
            data.longitude.max().values,
            target_resolution,
        )

        # 使用xarray的重采样方法（双线性插值）
        interpolated = data.interp(
            latitude=target_lats,
            longitude=target_lons,
            method="linear",
        )

        logger.info(
            "Interpolated grid shape: (%d, %d)",
            len(target_lats),
            len(target_lons),
        )

        return interpolated

    def temporal_alignment(
        self, data: xr.Dataset, target_hours: int = 6
    ) -> xr.Dataset:
        """将ERA5数据对齐到目标时间分辨率。

        Args:
            data: 输入的ERA5 Dataset。
            target_hours: 目标时间间隔（小时），默认6。

        Returns:
            时间对齐后的Dataset。
        """
        logger.info(
            "Aligning ERA5 to %d-hourly resolution", target_hours
        )

        # 如果数据已经是目标时间分辨率，直接返回
        if len(data.time) < 2:
            return data

        time_diff = pd.Timedelta(
            data.time.diff("time").median().values
        )
        if time_diff == pd.Timedelta(hours=target_hours):
            return data

        # 使用时间重采样
        resampled = data.resample(time=f"{target_hours}h").mean()

        logger.info(
            "Temporal alignment complete: %d time steps",
            len(resampled.time),
        )

        return resampled


class GMCPPreprocessor:
    """GMCP降水数据预处理器。

    负责加载GMCP数据，执行质量控制，进行时间对齐。

    Attributes:
        data_path: GMCP数据根目录路径。
        resolution: GMCP原始分辨率（度）。
    """

    def __init__(
        self,
        data_path: str,
        resolution: float = 0.1,
    ) -> None:
        """初始化GMCP预处理器。

        Args:
            data_path: GMCP数据根目录路径。
            resolution: GMCP原始分辨率（度），默认0.1。
        """
        self.data_path = Path(data_path)
        self.resolution = resolution

        # 中国区域边界（约73°E-135°E, 18°N-54°N）
        self.lat_min = 18.0
        self.lat_max = 54.0
        self.lon_min = 73.0
        self.lon_max = 135.0

        # 计算GMCP网格
        self.lats = np.arange(
            self.lat_max, self.lat_min, -self.resolution
        )
        self.lons = np.arange(
            self.lon_min, self.lon_max, self.resolution
        )

    def load_data(
        self,
        start_date: str,
        end_date: str,
    ) -> xr.Dataset:
        """加载指定时间范围的GMCP数据。

        Args:
            start_date: 开始日期，格式'YYYY-MM-DD'。
            end_date: 结束日期，格式'YYYY-MM-DD'。

        Returns:
            包含GMCP降水变量的xarray Dataset。

        Raises:
            FileNotFoundError: 当数据文件不存在时。
        """
        logger.info(
            "Loading GMCP data from %s to %s", start_date, end_date
        )

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # 收集所有时间步的数据文件
        datasets = []
        current = start
        while current <= end:
            year = current.year
            month = current.month

            # 构造文件路径
            file_path = self.data_path / f"gmcp_{year}_{month:02d}.nc"
            if file_path.exists():
                ds = xr.open_dataset(str(file_path))
                # 筛选时间范围和空间范围
                ds = ds.sel(
                    time=slice(
                        current.strftime("%Y-%m-%d"),
                        end.strftime("%Y-%m-%d"),
                    ),
                    latitude=slice(self.lat_max, self.lat_min),
                    longitude=slice(self.lon_min, self.lon_max),
                )
                datasets.append(ds)

            # 移动到下个月
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        if not datasets:
            raise FileNotFoundError(
                f"No GMCP data files found for period {start_date} to {end_date}"
            )

        # 合并所有数据
        merged = xr.concat(datasets, dim="time")

        logger.info(
            "Loaded GMCP data: %d time steps, shape: %s",
            len(merged.time),
            merged.precipitation_rate.shape
            if "precipitation_rate" in merged
            else "unknown",
        )

        return merged

    def quality_control(self, data: xr.Dataset) -> xr.Dataset:
        """执行GMCP数据质量控制。

        包括缺失值处理和异常值检测。

        Args:
            data: 输入的GMCP Dataset。

        Returns:
            经过质量控制的Dataset。
        """
        logger.info("Performing GMCP quality control")

        data = data.copy()

        if "precipitation_rate" not in data.data_vars:
            logger.warning(
                "Variable 'precipitation_rate' not found in GMCP data"
            )
            return data

        precip = data["precipitation_rate"]

        # 1. 检测缺失值
        missing_count = precip.isnull().sum().values
        if missing_count > 0:
            logger.warning(
                "GMCP has %d missing values", missing_count
            )
            precip = self._interpolate_missing(precip)

        # 2. 物理阈值检查
        threshold = PHYSICAL_THRESHOLDS["precipitation_rate"]
        out_of_range = (precip < threshold["min"]) | (
            precip > threshold["max"]
        )
        outlier_count = out_of_range.sum().values
        if outlier_count > 0:
            logger.warning(
                "GMCP has %d values outside physical range "
                "[%.1f, %.1f]",
                outlier_count,
                threshold["min"],
                threshold["max"],
            )
            precip = precip.where(~out_of_range)
            precip = self._interpolate_missing(precip)

        # 3. 降水非负约束
        negative_count = (precip < 0).sum().values
        if negative_count > 0:
            logger.warning(
                "GMCP has %d negative values, clipping to 0",
                negative_count,
            )
            precip = precip.clip(min=0)

        data["precipitation_rate"] = precip

        return data

    def _interpolate_missing(self, data: xr.DataArray) -> xr.DataArray:
        """使用时空插值填充缺失值。

        Args:
            data: 包含缺失值的数据数组。

        Returns:
            填充缺失值后的数据数组。
        """
        # 时间维度线性插值
        data = data.interpolate_na(dim="time", method="linear")

        # 空间维度最近邻插值
        if data.isnull().any():
            data = data.interpolate_na(
                dim="latitude", method="nearest"
            )
            data = data.interpolate_na(
                dim="longitude", method="nearest"
            )

        return data

    def temporal_alignment(
        self, data: xr.Dataset, target_hours: int = 6
    ) -> xr.Dataset:
        """将GMCP数据对齐到目标时间分辨率。

        Args:
            data: 输入的GMCP Dataset。
            target_hours: 目标时间间隔（小时），默认6。

        Returns:
            时间对齐后的Dataset。
        """
        logger.info(
            "Aligning GMCP to %d-hourly resolution", target_hours
        )

        if len(data.time) < 2:
            return data

        time_diff = pd.Timedelta(
            data.time.diff("time").median().values
        )
        if time_diff == pd.Timedelta(hours=target_hours):
            return data

        # 时间重采样（累加降水）
        resampled = data.resample(time=f"{target_hours}h").sum()

        logger.info(
            "GMCP temporal alignment complete: %d time steps",
            len(resampled.time),
        )

        return resampled


class SpatialAlignment:
    """空间配准处理器。

    将ERA5和GMCP数据配准到统一的0.1°网格系统。
    """

    def __init__(self, target_resolution: float = 0.1) -> None:
        """初始化空间配准处理器。

        Args:
            target_resolution: 目标分辨率（度），默认0.1。
        """
        self.target_resolution = target_resolution

    def align_grids(
        self,
        era5_data: xr.Dataset,
        gmcp_data: xr.Dataset,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """将ERA5和GMCP数据对齐到统一网格。

        Args:
            era5_data: ERA5数据Dataset。
            gmcp_data: GMCP数据Dataset。

        Returns:
            对齐后的(ERA5, GMCP)元组。
        """
        logger.info("Aligning ERA5 and GMCP grids")

        # 获取GMCP的空间范围（中国区域）
        gmcp_lat_min = float(gmcp_data.latitude.min())
        gmcp_lat_max = float(gmcp_data.latitude.max())
        gmcp_lon_min = float(gmcp_data.longitude.min())
        gmcp_lon_max = float(gmcp_data.longitude.max())

        # 裁剪ERA5到中国区域
        era5_china = era5_data.sel(
            latitude=slice(gmcp_lat_max, gmcp_lat_min),
            longitude=slice(gmcp_lon_min, gmcp_lon_max),
        )

        # 创建统一的目标网格
        target_lats = np.arange(
            gmcp_lat_max, gmcp_lat_min, -self.target_resolution
        )
        target_lons = np.arange(
            gmcp_lon_min, gmcp_lon_max, self.target_resolution
        )

        # ERA5双线性插值到0.1°
        era5_aligned = era5_china.interp(
            latitude=target_lats,
            longitude=target_lons,
            method="linear",
        )

        # GMCP重采样到统一网格
        gmcp_aligned = gmcp_data.interp(
            latitude=target_lats,
            longitude=target_lons,
            method="nearest",
        )

        logger.info(
            "Grid alignment complete: lat=(%.1f, %.1f), "
            "lon=(%.1f, %.1f), shape=(%d, %d)",
            gmcp_lat_min,
            gmcp_lat_max,
            gmcp_lon_min,
            gmcp_lon_max,
            len(target_lats),
            len(target_lons),
        )

        return era5_aligned, gmcp_aligned


class Normalizer:
    """数据归一化处理器。

    支持多种归一化方法：Z-score标准化、Log-MinMax归一化等。
    """

    def __init__(self) -> None:
        """初始化归一化处理器。"""
        self.stats: Dict[str, Dict[str, float]] = {}

    def compute_statistics(
        self, data: xr.Dataset, method: str = "zscore"
    ) -> Dict[str, Dict[str, float]]:
        """计算归一化统计参数。

        Args:
            data: 输入数据Dataset。
            method: 归一化方法，'zscore'或'log_minmax'。

        Returns:
            包含统计参数的字典。
        """
        stats = {}

        for var in data.data_vars:
            var_data = data[var]

            if method == "zscore":
                stats[var] = {
                    "mean": float(var_data.mean()),
                    "std": float(var_data.std()),
                    "min": float(var_data.min()),
                    "max": float(var_data.max()),
                }
            elif method == "log_minmax":
                # 对数变换后的统计
                log_data = np.log1p(var_data.clip(min=0))
                stats[var] = {
                    "log_mean": float(log_data.mean()),
                    "log_std": float(log_data.std()),
                    "log_min": float(log_data.min()),
                    "log_max": float(log_data.max()),
                }

        self.stats.update(stats)

        logger.info(
            "Computed statistics for %d variables using %s method",
            len(stats),
            method,
        )

        return stats

    def normalize(
        self,
        data: xr.Dataset,
        method: str = "zscore",
        stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> xr.Dataset:
        """执行数据归一化。

        Args:
            data: 输入数据Dataset。
            method: 归一化方法。
            stats: 预计算的统计参数，如为None则从数据计算。

        Returns:
            归一化后的Dataset。
        """
        if stats is None:
            stats = self.compute_statistics(data, method)

        normalized = data.copy()

        for var in data.data_vars:
            if var not in stats:
                continue

            var_stats = stats[var]

            if method == "zscore":
                mean = var_stats["mean"]
                std = var_stats["std"]
                # 避免除零
                std = max(std, 1e-8)
                normalized[var] = (data[var] - mean) / std

            elif method == "log_minmax":
                log_data = np.log1p(data[var].clip(min=0))
                log_min = var_stats["log_min"]
                log_max = var_stats["log_max"]
                log_range = log_max - log_min
                # 避免除零
                log_range = max(log_range, 1e-8)
                normalized[var] = (log_data - log_min) / log_range

        logger.info(
            "Normalization complete using %s method", method
        )

        return normalized

    def denormalize(
        self,
        data: xr.Dataset,
        method: str = "zscore",
        stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> xr.Dataset:
        """执行数据反归一化。

        Args:
            data: 归一化后的Dataset。
            method: 归一化方法。
            stats: 归一化时使用的统计参数。

        Returns:
            反归一化后的Dataset。
        """
        if stats is None:
            stats = self.stats

        denormalized = data.copy()

        for var in data.data_vars:
            if var not in stats:
                continue

            var_stats = stats[var]

            if method == "zscore":
                mean = var_stats["mean"]
                std = var_stats["std"]
                denormalized[var] = data[var] * std + mean

            elif method == "log_minmax":
                log_min = var_stats["log_min"]
                log_max = var_stats["log_max"]
                log_range = log_max - log_min
                log_data = data[var] * log_range + log_min
                denormalized[var] = np.expm1(log_data)

        logger.info(
            "Denormalization complete using %s method", method
        )

        return denormalized


class WeatherPreprocessingPipeline:
    """天气数据预处理管道。

    整合所有预处理步骤，提供统一的预处理接口。

    Attributes:
        era5_preprocessor: ERA5预处理器实例。
        gmcp_preprocessor: GMCP预处理器实例。
        spatial_alignment: 空间配准处理器实例。
        normalizer: 归一化处理器实例。
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """初始化预处理管道。

        Args:
            config: 配置字典，包含数据路径和预处理参数。
        """
        if config is None:
            config = self._get_default_config()

        # 初始化各处理器
        era5_config = config.get("era5", {})
        gmcp_config = config.get("gmcp", {})
        prep_config = config.get("preprocessing", {})

        self.era5_preprocessor = ERA5Preprocessor(
            data_path=era5_config.get("path", ""),
            resolution=era5_config.get("resolution", 0.25),
            variables=era5_config.get("variables", []),
        )

        self.gmcp_preprocessor = GMCPPreprocessor(
            data_path=gmcp_config.get("path", ""),
            resolution=gmcp_config.get("resolution", 0.1),
        )

        self.spatial_alignment = SpatialAlignment(
            target_resolution=prep_config.get("target_resolution", 0.1),
        )

        self.normalizer = Normalizer()

        # 归一化统计参数
        self.normalization_stats: Dict[str, Dict] = {}

    def _get_default_config(self) -> Dict:
        """获取默认配置。

        Returns:
            默认配置字典。
        """
        return {
            "era5": {
                "path": "F:/ERA5再分析数据下载",
                "resolution": 0.25,
                "variables": [
                    "temperature",
                    "u_wind",
                    "v_wind",
                    "geopotential",
                    "relative_humidity",
                    "surface_pressure",
                    "total_precipitation",
                ],
            },
            "gmcp": {
                "path": "F:/GMCP_Precipitation",
                "resolution": 0.1,
                "variables": ["precipitation_rate"],
            },
            "preprocessing": {
                "target_resolution": 0.1,
                "temporal_resolution": 6,
                "train_period": ["2000-01-01", "2019-12-31"],
                "val_period": ["2020-01-01", "2022-12-31"],
                "test_period": ["2023-01-01", "2024-12-31"],
            },
        }

    def process(
        self,
        start_date: str,
        end_date: str,
        compute_norm_stats: bool = False,
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """执行完整的预处理管道。

        Args:
            start_date: 开始日期，格式'YYYY-MM-DD'。
            end_date: 结束日期，格式'YYYY-MM-DD'。
            compute_norm_stats: 是否计算归一化统计参数。

        Returns:
            包含(era5_normalized, gmcp_normalized, norm_stats)的元组。
        """
        logger.info(
            "Starting preprocessing pipeline for %s to %s",
            start_date,
            end_date,
        )

        # 1. 加载ERA5数据
        logger.info("Step 1: Loading ERA5 data")
        era5_raw = self.era5_preprocessor.load_data(
            start_date, end_date
        )

        # 2. ERA5质量控制
        logger.info("Step 2: ERA5 quality control")
        era5_clean = self.era5_preprocessor.quality_control(era5_raw)

        # 3. ERA5时间对齐
        logger.info("Step 3: ERA5 temporal alignment")
        era5_aligned = self.era5_preprocessor.temporal_alignment(
            era5_clean, target_hours=6
        )

        # 4. 加载GMCP数据
        logger.info("Step 4: Loading GMCP data")
        gmcp_raw = self.gmcp_preprocessor.load_data(
            start_date, end_date
        )

        # 5. GMCP质量控制
        logger.info("Step 5: GMCP quality control")
        gmcp_clean = self.gmcp_preprocessor.quality_control(gmcp_raw)

        # 6. GMCP时间对齐
        logger.info("Step 6: GMCP temporal alignment")
        gmcp_aligned = self.gmcp_preprocessor.temporal_alignment(
            gmcp_clean, target_hours=6
        )

        # 7. 空间配准
        logger.info("Step 7: Spatial alignment")
        era5_gridded, gmcp_gridded = self.spatial_alignment.align_grids(
            era5_aligned, gmcp_aligned
        )

        # 8. 归一化
        logger.info("Step 8: Normalization")
        if compute_norm_stats:
            era5_stats = self.normalizer.compute_statistics(
                era5_gridded, method="zscore"
            )
            gmcp_stats = self.normalizer.compute_statistics(
                gmcp_gridded, method="log_minmax"
            )
            self.normalization_stats = {
                "era5": era5_stats,
                "gmcp": gmcp_stats,
            }
        else:
            self.normalization_stats = {}

        era5_normalized = self.normalizer.normalize(
            era5_gridded,
            method="zscore",
            stats=self.normalization_stats.get("era5"),
        )

        gmcp_normalized = self.normalizer.normalize(
            gmcp_gridded,
            method="log_minmax",
            stats=self.normalization_stats.get("gmcp"),
        )

        logger.info("Preprocessing pipeline complete")

        return (
            era5_normalized,
            gmcp_normalized,
            self.normalization_stats,
        )

    def save_statistics(self, output_path: str) -> None:
        """保存归一化统计参数。

        Args:
            output_path: 输出文件路径。
        """
        import json

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 将numpy值转换为Python原生类型
        serializable_stats = {}
        for source, stats in self.normalization_stats.items():
            serializable_stats[source] = {}
            for var, var_stats in stats.items():
                serializable_stats[source][var] = {
                    k: float(v) for k, v in var_stats.items()
                }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

        logger.info("Normalization statistics saved to %s", output_path)

    def load_statistics(self, stats_path: str) -> None:
        """加载归一化统计参数。

        Args:
            stats_path: 统计参数文件路径。
        """
        import json

        with open(stats_path, "r", encoding="utf-8") as f:
            self.normalization_stats = json.load(f)

        logger.info(
            "Normalization statistics loaded from %s", stats_path
        )


def create_preprocessing_pipeline(
    config_path: Optional[str] = None,
) -> WeatherPreprocessingPipeline:
    """创建预处理管道的工厂函数。

    Args:
        config_path: YAML配置文件路径，如为None则使用默认配置。

    Returns:
        初始化后的预处理管道实例。
    """
    if config_path is not None:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = None

    return WeatherPreprocessingPipeline(config)


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 创建预处理管道
    pipeline = create_preprocessing_pipeline(
        config_path="e:/weather/src/configs/data_config.yaml"
    )

    # 处理训练数据
    era5_train, gmcp_train, stats = pipeline.process(
        start_date="2000-01-01",
        end_date="2005-12-31",
        compute_norm_stats=True,
    )

    # 保存统计参数
    pipeline.save_statistics("e:/weather/data/normalization_stats.json")

    print(f"ERA5 shape: {era5_train.dims}")
    print(f"GMCP shape: {gmcp_train.dims}")
