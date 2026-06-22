#!/usr/bin/env python3
"""
WeatherBench2 ERA5 Data Preprocessor
=====================================

将WeatherBench2 ERA5数据转换为PyTorch训练格式。

数据规格：
- 时间范围：2018-01-01 至 2019-12-31
- 时间步数：2920（每天4次，6小时间隔）
- 气压层：5层（1000/850/500/200/100 hPa）
- 空间分辨率：121 x 240（1.5°）
- 变量：13个（包括降水）

处理流程：
1. 计算Z-score统计参数（逐变量加载，避免内存溢出）
2. 按年份划分（2018=验证集, 2019=测试集）
3. 逐变量标准化并保存原始数据为numpy .npy文件
4. 配套WeatherBench2Dataset类在训练时按需创建序列

Usage:
    python preprocess_weatherbench2.py

Author: weather-model-trainer
Date: 2026-06-15
"""

import gc
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocess_weatherbench2.log"),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 配置常量
# ==============================================================================

DATA_CONFIG = {
    "input_path": "F:/WeatherBench2_ERA5/era5_2018-2019.nc",
    "output_dir": "F:/WeatherBench2_ERA5/processed",
}

SEQUENCE_CONFIG = {
    "history_steps": 4,   # 历史4步 = 24小时
    "forecast_steps": 4,  # 预测4步 = 24小时
    "time_step_hours": 6,
}

SPLIT_CONFIG = {
    "val_year": 2018,
    "test_year": 2019,
}

VARIABLE_GROUPS = {
    "pressure_level_vars": [
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "temperature",
        "specific_humidity",
        "vertical_velocity",
    ],
    "surface_vars": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "total_column_water_vapour",
    ],
    "precipitation_vars": [
        "total_precipitation_6hr",
        "total_precipitation_24hr",
    ],
}

ALL_VARIABLE_NAMES: List[str] = (
    list(VARIABLE_GROUPS["pressure_level_vars"])
    + list(VARIABLE_GROUPS["surface_vars"])
    + list(VARIABLE_GROUPS["precipitation_vars"])
)


# ==============================================================================
# 统计参数计算（逐变量加载）
# ==============================================================================


def compute_statistics_per_variable(
    file_path: str, variable_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """逐变量计算Z-score统计参数。

    Args:
        file_path: NetCDF文件路径。
        variable_names: 变量名列表。

    Returns:
        包含每个变量mean/std/min/max的字典。
    """
    logger.info("Computing normalization statistics (variable-by-variable)")
    stats: Dict[str, Dict[str, float]] = {}

    for var_name in variable_names:
        logger.info("  Computing stats for: %s", var_name)

        ds_var = xr.open_dataset(file_path)[[var_name]]
        var_da = ds_var[var_name]

        mean_val = float(var_da.mean().values)
        std_val = float(var_da.std().values)
        min_val = float(var_da.min().values)
        max_val = float(var_da.max().values)

        if std_val < 1e-8:
            logger.warning("  '%s' std too small (%.2e), clamping to 1e-8", var_name, std_val)
            std_val = 1e-8

        stats[var_name] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
        }
        logger.info("    mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
                     mean_val, std_val, min_val, max_val)

        ds_var.close()
        del ds_var, var_da
        gc.collect()

    return stats


# ==============================================================================
# 逐变量标准化并保存原始数据
# ==============================================================================


def normalize_and_save_variable(
    file_path: str,
    var_name: str,
    stats: Dict[str, float],
    year: int,
    split_name: str,
    output_dir: str,
) -> Tuple[int, Tuple[int, ...]]:
    """处理单个变量的单年数据：加载、标准化、保存为.npy。

    Args:
        file_path: NetCDF文件路径。
        var_name: 变量名。
        stats: 该变量的统计参数。
        year: 年份。
        split_name: "val"或"test"。
        output_dir: 输出目录。

    Returns:
        (时间步数, 数据shape) 元组。
    """
    # 加载该变量的该年数据
    ds_full = xr.open_dataset(file_path)[[var_name]]
    ds_year = ds_full.sel(time=str(year))
    ds_full.close()
    del ds_full

    var_data = ds_year[var_name].values
    del ds_year
    gc.collect()

    # 标准化
    mean = stats["mean"]
    std = stats["std"]
    var_data = (var_data - mean) / std

    # 确保float32
    var_data = var_data.astype(np.float32)

    # 保存为 .npy 文件（内存映射友好）
    out_path = os.path.join(output_dir, f"{split_name}_{var_name}.npy")
    np.save(out_path, var_data)

    shape = var_data.shape
    n_times = shape[0]
    file_mb = Path(out_path).stat().st_size / 1024 / 1024

    logger.info("    %s: shape=%s, %d time steps, %.1f MB",
                var_name, str(shape), n_times, file_mb)

    del var_data
    gc.collect()

    return n_times, shape


# ==============================================================================
# 保存工具函数
# ==============================================================================


def save_statistics(stats: Dict[str, Dict[str, float]], output_path: str) -> None:
    """保存统计参数到JSON文件。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Statistics saved to: %s", output_path)


def save_metadata(
    output_dir: str,
    variable_order: List[str],
    stats: Dict[str, Dict[str, float]],
    year_shapes: Dict[str, Dict[str, Tuple[int, ...]]],
) -> None:
    """保存数据集元数据。"""
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")

    metadata = {
        "description": "WeatherBench2 ERA5 preprocessed for weather prediction models",
        "source_file": DATA_CONFIG["input_path"],
        "spatial_resolution": {"lat": 121, "lon": 240, "deg": 1.5},
        "pressure_levels_hpa": [1000, 850, 500, 200, 100],
        "sequence_config": SEQUENCE_CONFIG,
        "split_config": SPLIT_CONFIG,
        "variable_order": variable_order,
        "variable_groups": VARIABLE_GROUPS,
        "normalization_stats": stats,
        "splits": {},
    }

    for split_name in ["val", "test"]:
        year = SPLIT_CONFIG[f"{split_name}_year"]
        split_info: Dict = {"year": year, "variables": {}}
        for var_name in variable_order:
            shape = year_shapes[split_name][var_name]
            n_seq = shape[0] - SEQUENCE_CONFIG["history_steps"] - SEQUENCE_CONFIG["forecast_steps"] + 1
            split_info["variables"][var_name] = {
                "file": f"{split_name}_{var_name}.npy",
                "shape": list(shape),
                "n_sequences": n_seq,
            }
        metadata["splits"][split_name] = split_info

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved to: %s", metadata_path)


# ==============================================================================
# PyTorch Dataset（用于训练时加载，按需创建序列）
# ==============================================================================


class WeatherBench2Dataset(Dataset):
    """WeatherBench2预处理数据的PyTorch Dataset。

    从逐变量保存的.npy文件中加载数据（使用内存映射），
    按variable_order拼接通道，在__getitem__中按需创建序列。

    这种设计：
    - 内存占用极低（只有当前访问的序列在内存中）
    - 磁盘空间高效（无序列冗余存储）
    - 灵活可配置（可修改序列长度无需重新预处理）

    Usage:
        dataset = WeatherBench2Dataset("F:/WeatherBench2_ERA5/processed", split="val")
        history, forecast = dataset[0]
        # history shape: (4, 43, 121, 240)
        # forecast shape: (4, 43, 121, 240)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        history_steps: int = 4,
        forecast_steps: int = 4,
    ) -> None:
        """初始化数据集。

        Args:
            data_dir: 处理后数据目录。
            split: "val" 或 "test"。
            history_steps: 历史时间步数。
            forecast_steps: 预测时间步数。
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.total_steps = history_steps + forecast_steps

        # 加载元数据
        with open(self.data_dir / "dataset_metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.variable_order: List[str] = self.metadata["variable_order"]

        # 使用内存映射加载所有变量
        self._arrays: Dict[str, np.ndarray] = {}
        self._n_times = 0

        for var_name in self.variable_order:
            npy_path = self.data_dir / f"{split}_{var_name}.npy"
            arr = np.load(str(npy_path), mmap_mode="r")
            self._arrays[var_name] = arr
            if self._n_times == 0:
                self._n_times = arr.shape[0]

        # 可用序列数
        self.n_sequences = self._n_times - self.total_steps + 1

        # 计算通道数
        self.n_channels = 0
        for var_name in self.variable_order:
            arr = self._arrays[var_name]
            if arr.ndim == 4:  # (time, level, lat, lon)
                self.n_channels += arr.shape[1]
            else:  # (time, lat, lon)
                self.n_channels += 1

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一对历史-预测序列。

        Args:
            idx: 序列索引。

        Returns:
            (history, forecast) 元组。
            history shape: (history_steps, n_channels, lat, lon)
            forecast shape: (forecast_steps, n_channels, lat, lon)
        """
        start = idx
        mid = idx + self.history_steps
        end = idx + self.total_steps

        history_parts = []
        forecast_parts = []

        for var_name in self.variable_order:
            arr = self._arrays[var_name]

            h = arr[start:mid]
            f = arr[mid:end]

            # 确保有通道维度: (time, lat, lon) -> (time, 1, lat, lon)
            if h.ndim == 3:
                h = h[:, np.newaxis, :, :]
                f = f[:, np.newaxis, :, :]

            history_parts.append(h)
            forecast_parts.append(f)

        history = np.concatenate(history_parts, axis=1)
        forecast = np.concatenate(forecast_parts, axis=1)

        return torch.from_numpy(history.copy()), torch.from_numpy(forecast.copy())

    def get_variable_info(self) -> Dict:
        """获取变量信息。"""
        return {
            "variable_order": self.variable_order,
            "n_channels": self.n_channels,
            "n_sequences": self.n_sequences,
            "spatial_shape": (
                self._arrays[self.variable_order[0]].shape[-2],
                self._arrays[self.variable_order[0]].shape[-1],
            ),
        }

    def close(self) -> None:
        """释放内存映射。"""
        self._arrays.clear()
        gc.collect()


# ==============================================================================
# 主处理管道
# ==============================================================================


def preprocess_weatherbench2() -> None:
    """主预处理管道。"""
    logger.info("=" * 80)
    logger.info("WeatherBench2 ERA5 Data Preprocessing Pipeline")
    logger.info("=" * 80)

    input_path = DATA_CONFIG["input_path"]
    output_dir = DATA_CONFIG["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    variable_order = ALL_VARIABLE_NAMES

    # ------------------------------------------------------------------
    # Step 1: 探查数据
    # ------------------------------------------------------------------
    logger.info("\n[Step 1] Inspecting WeatherBench2 data")
    ds_probe = xr.open_dataset(input_path)
    logger.info("  Time steps: %d", len(ds_probe.time))
    logger.info("  Pressure levels: %s", ds_probe.level.values.tolist())
    logger.info("  Spatial grid: %d x %d", len(ds_probe.latitude), len(ds_probe.longitude))
    logger.info("  Variables: %s", list(ds_probe.data_vars))
    ds_probe.close()
    del ds_probe

    # ------------------------------------------------------------------
    # Step 2: 计算归一化统计参数
    # ------------------------------------------------------------------
    logger.info("\n[Step 2] Computing normalization statistics")
    stats = compute_statistics_per_variable(input_path, variable_order)

    stats_path = os.path.join(output_dir, "normalization_stats.json")
    save_statistics(stats, stats_path)

    # ------------------------------------------------------------------
    # Step 3: 逐变量、逐年标准化并保存
    # ------------------------------------------------------------------
    logger.info("\n[Step 3] Normalizing and saving variable data")

    year_shapes: Dict[str, Dict[str, Tuple[int, ...]]] = {"val": {}, "test": {}}

    for year, split_name in [
        (SPLIT_CONFIG["val_year"], "val"),
        (SPLIT_CONFIG["test_year"], "test"),
    ]:
        logger.info("\n  --- Year %d (%s set) ---", year, split_name)

        for var_name in variable_order:
            n_times, shape = normalize_and_save_variable(
                file_path=input_path,
                var_name=var_name,
                stats=stats[var_name],
                year=year,
                split_name=split_name,
                output_dir=output_dir,
            )
            year_shapes[split_name][var_name] = shape

    # ------------------------------------------------------------------
    # Step 4: 保存元数据
    # ------------------------------------------------------------------
    logger.info("\n[Step 4] Saving dataset metadata")
    save_metadata(output_dir, variable_order, stats, year_shapes)

    # ------------------------------------------------------------------
    # Step 5: 验证 -- 用Dataset类加载并检查
    # ------------------------------------------------------------------
    logger.info("\n[Step 5] Verifying with WeatherBench2Dataset")

    for split_name in ["val", "test"]:
        ds = WeatherBench2Dataset(output_dir, split=split_name)
        info = ds.get_variable_info()
        h, f = ds[0]
        logger.info("  %s set:", split_name)
        logger.info("    Sequences: %d", info["n_sequences"])
        logger.info("    Channels: %d", info["n_channels"])
        logger.info("    Spatial: %s", str(info["spatial_shape"]))
        logger.info("    Sample history shape: %s", str(h.shape))
        logger.info("    Sample forecast shape: %s", str(f.shape))
        ds.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_val = year_shapes["val"][variable_order[0]][0] - SEQUENCE_CONFIG["history_steps"] - SEQUENCE_CONFIG["forecast_steps"] + 1
    n_test = year_shapes["test"][variable_order[0]][0] - SEQUENCE_CONFIG["history_steps"] - SEQUENCE_CONFIG["forecast_steps"] + 1
    n_pl = len(VARIABLE_GROUPS["pressure_level_vars"])
    n_sfc = len(VARIABLE_GROUPS["surface_vars"]) + len(VARIABLE_GROUPS["precipitation_vars"])
    total_ch = n_pl * 5 + n_sfc

    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 80)
    logger.info("  Variables: %d (%d pressure-level x 5 levels + %d surface = %d channels)",
                len(variable_order), n_pl, n_sfc, total_ch)
    logger.info("  Validation (2018): %d sequences", n_val)
    logger.info("  Test (2019):       %d sequences", n_test)
    logger.info("  Sequence: history=%dh -> forecast=%dh",
                SEQUENCE_CONFIG["history_steps"] * 6, SEQUENCE_CONFIG["forecast_steps"] * 6)
    logger.info("  Output: %s", output_dir)
    logger.info("  Files: normalization_stats.json, dataset_metadata.json,")
    logger.info("         {val,test}_{variable}.npy (x%d per split)", len(variable_order))


if __name__ == "__main__":
    preprocess_weatherbench2()
