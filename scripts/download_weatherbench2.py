#!/usr/bin/env python3
"""
WeatherBench2 ERA5数据下载脚本
从Google Cloud Storage下载预处理好的ERA5数据

用法：
    # 下载评测数据（2018-2019年）
    python download_weatherbench2.py --start_year 2018 --end_year 2019

    # 下载训练数据（1979-2017年，分批下载）
    python download_weatherbench2.py --start_year 1979 --end_year 1985

    # 只下载特定年份
    python download_weatherbench2.py --single_year 2019
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import xarray as xr
    import gcsfs
except ImportError:
    print("请先安装依赖: pip install xarray gcsfs zarr")
    sys.exit(1)

# ==================== 配置区域 ====================

# WeatherBench2 ERA5数据地址
WB2_BASE_URL = "gs://weatherbench2/datasets/era5"

# 推荐的数据集（13层，6小时，含派生变量）
WB2_DATASET = "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# 我们需要的变量
# 气压层变量
PRESSURE_VARIABLES = [
    "u_component_of_wind",      # U风速
    "v_component_of_wind",      # V风速
    "geopotential",             # 位势高度
    "temperature",              # 温度
    "specific_humidity",        # 比湿
    "vertical_velocity",        # 垂直速度
]

# 地面变量
SINGLE_LEVEL_VARIABLES = [
    "total_precipitation_6hr",  # 6小时累计降水
    "total_precipitation_24hr", # 24小时累计降水
    "2m_temperature",           # 2米温度
    "10m_u_component_of_wind",  # 10米U风速
    "10m_v_component_of_wind",  # 10米V风速
    "surface_pressure",         # 地面气压
    "total_column_water_vapour", # 整层水汽
]

# 气压层（我们只需要5层）
TARGET_LEVELS = [1000, 850, 500, 200, 100]

# 数据存储根目录（必须在F盘）
BASE_DIR = r"F:\WeatherBench2_ERA5"

# ==================== 下载函数 ====================

def download_year_range(start_year, end_year, base_dir, variables=None):
    """下载指定年份范围的数据"""
    print("=" * 60)
    print("WeatherBench2 ERA5数据下载")
    print("=" * 60)
    print(f"数据集: {WB2_DATASET}")
    print(f"年份范围: {start_year}-{end_year}")
    print(f"存储路径: {base_dir}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(base_dir, exist_ok=True)

    # 连接GCS
    print("\n[1/4] 连接Google Cloud Storage...")
    fs = gcsfs.GCSFileSystem(token="anon")

    # 打开Zarr数据集
    print("[2/4] 打开Zarr数据集...")
    url = f"{WB2_BASE_URL}/{WB2_DATASET}"
    print(f"  URL: {url}")

    try:
        # 使用xr.open_zarr打开远程数据
        ds = xr.open_zarr(fs.get_mapper(url), consolidated=True)
        print(f"  数据集维度: {dict(ds.dims)}")
        print(f"  变量数: {len(ds.data_vars)}")
    except Exception as e:
        print(f"  打开失败: {e}")
        print("  尝试使用默认参数...")
        ds = xr.open_zarr(fs.get_mapper(url))

    # 选择需要的变量
    print("[3/4] 选择变量...")
    if variables is None:
        # 选择所有需要的变量
        available_vars = list(ds.data_vars)
        needed_vars = [v for v in PRESSURE_VARIABLES + SINGLE_LEVEL_VARIABLES
                      if v in available_vars]
        print(f"  可用变量: {len(available_vars)}")
        print(f"  选择变量: {len(needed_vars)}")
        ds = ds[needed_vars]
    else:
        ds = ds[variables]

    # 选择气压层
    print("  选择气压层...")
    if "level" in ds.dims:
        available_levels = ds.level.values
        print(f"  可用气压层: {available_levels}")
        target_levels_available = [l for l in TARGET_LEVELS if l in available_levels]
        print(f"  选择气压层: {target_levels_available}")
        ds = ds.sel(level=target_levels_available)

    # 选择时间范围
    print("[4/4] 选择时间范围...")
    time_range = slice(f"{start_year}-01-01", f"{end_year}-12-31")
    ds_subset = ds.sel(time=time_range)
    print(f"  时间范围: {start_year}-01-01 至 {end_year}-12-31")
    print(f"  时间步数: {len(ds_subset.time)}")

    # 分年保存
    print("\n开始分年保存...")
    for year in range(start_year, end_year + 1):
        # 选择该年的数据
        year_data = ds_subset.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

        if len(year_data.time) == 0:
            print(f"  [{year}] 无数据，跳过")
            continue

        # 保存为NetCDF格式
        output_file = os.path.join(base_dir, f"era5_{year}.nc")
        print(f"  [{year}] 保存到 {output_file}...")

        try:
            # 保存为NetCDF4格式
            year_data.to_netcdf(output_file, format="NETCDF4")
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"  [{year}] 完成! 文件大小: {file_size:.2f} MB")
        except Exception as e:
            print(f"  [{year}] 保存失败: {e}")
            # 尝试保存为zarr格式
            zarr_file = os.path.join(base_dir, f"era5_{year}.zarr")
            try:
                year_data.to_zarr(zarr_file, mode="w")
                print(f"  [{year}] 保存为zarr格式: {zarr_file}")
            except Exception as e2:
                print(f"  [{year}] zarr保存也失败: {e2}")

    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)


def download_specific_variables(start_year, end_year, base_dir):
    """只下载特定变量（更小的文件）"""
    print("=" * 60)
    print("WeatherBench2 ERA5数据下载（精简版）")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(base_dir, exist_ok=True)

    # 连接GCS
    print("\n[1/3] 连接Google Cloud Storage...")
    fs = gcsfs.GCSFileSystem(token="anon")

    # 打开Zarr数据集
    print("[2/3] 打开Zarr数据集...")
    url = f"{WB2_BASE_URL}/{WB2_DATASET}"
    ds = xr.open_zarr(fs.get_mapper(url), consolidated=True)

    # 选择需要的变量
    print("[3/3] 选择变量并下载...")
    all_needed_vars = PRESSURE_VARIABLES + SINGLE_LEVEL_VARIABLES
    available_vars = [v for v in all_needed_vars if v in ds.data_vars]
    print(f"  选择变量: {available_vars}")

    ds_subset = ds[available_vars]

    # 选择气压层
    if "level" in ds_subset.dims:
        target_levels_available = [l for l in TARGET_LEVELS if l in ds_subset.level.values]
        ds_subset = ds_subset.sel(level=target_levels_available)
        print(f"  选择气压层: {target_levels_available}")

    # 选择时间范围
    time_range = slice(f"{start_year}-01-01", f"{end_year}-12-31")
    ds_subset = ds_subset.sel(time=time_range)
    print(f"  时间范围: {start_year} - {end_year}")
    print(f"  时间步数: {len(ds_subset.time)}")

    # 保存为单个NetCDF文件
    output_file = os.path.join(base_dir, f"era5_{start_year}-{end_year}.nc")
    print(f"\n保存到 {output_file}...")

    try:
        ds_subset.to_netcdf(output_file, format="NETCDF4")
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"完成! 文件大小: {file_size:.2f} MB")
    except Exception as e:
        print(f"保存失败: {e}")
        # 保存为zarr
        zarr_file = os.path.join(base_dir, f"era5_{start_year}-{end_year}.zarr")
        try:
            ds_subset.to_zarr(zarr_file, mode="w")
            print(f"保存为zarr格式: {zarr_file}")
        except Exception as e2:
            print(f"zarr保存也失败: {e2}")


def main():
    parser = argparse.ArgumentParser(description="WeatherBench2 ERA5数据下载")
    parser.add_argument("--start_year", type=int, default=2018, help="起始年份")
    parser.add_argument("--end_year", type=int, default=2019, help="结束年份")
    parser.add_argument("--single_year", type=int, help="只下载某一年")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="存储根目录")
    parser.add_argument("--minimal", action="store_true", help="精简模式（只下载核心变量）")
    args = parser.parse_args()

    # 确定年份范围
    if args.single_year:
        start_year = args.single_year
        end_year = args.single_year
    else:
        start_year = args.start_year
        end_year = args.end_year

    # 执行下载
    if args.minimal:
        download_specific_variables(start_year, end_year, args.base_dir)
    else:
        download_year_range(start_year, end_year, args.base_dir)


if __name__ == "__main__":
    main()
