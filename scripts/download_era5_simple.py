#!/usr/bin/env python3
"""
ERA5数据下载脚本（简化版）
基于CDS API全量格式，只下载PhyDiff-Net需要的变量

用法：
    python download_era5_simple.py --start_year 1979 --end_year 2005
    python download_era5_simple.py --start_year 2018 --end_year 2019  # 评测数据
    python download_era5_simple.py --single_year 2019  # 只下载2019年
"""

import os
import sys
import time
import argparse
from pathlib import Path

try:
    import cdsapi
except ImportError:
    print("请先安装cdsapi: pip install cdsapi")
    sys.exit(1)

# ==================== 配置区域 ====================

# 数据存储根目录（必须在F盘）
BASE_DIR = r"F:\ERA5再分析数据下载"

# 我们需要的气压层（5层，不是全部37层）
PRESSURE_LEVELS = ["1000", "850", "500", "200", "100"]

# 我们需要的变量（气压层）
PRESSURE_VARIABLES = [
    "u_component_of_wind",      # U风速
    "v_component_of_wind",      # V风速
    "geopotential",             # 位势高度
    "temperature",              # 温度
    "relative_humidity",        # 相对湿度
    "specific_humidity",        # 比湿
]

# 我们需要的变量（地面）
SINGLE_LEVEL_VARIABLES = [
    "total_precipitation",      # 总降水量
    "2m_temperature",           # 2米温度
    "10m_u_component_of_wind",  # 10米U风速
    "10m_v_component_of_wind",  # 10米V风速
    "surface_pressure",         # 地面气压
    "total_column_water_vapour", # 整层水汽
    "surface_solar_radiation_downwards",  # 向下短波辐射
    "surface_thermal_radiation_upwards",  # 向上长波辐射
]

# 每天只需要4个时间点（00/06/12/18 UTC）
TIMES = ["00:00", "06:00", "12:00", "18:00"]

# ==================== 下载函数 ====================

def download_pressure_level(year, month, day, output_dir):
    """下载气压层数据"""
    client = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "variable": PRESSURE_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": f"{day:02d}",
        "time": TIMES,
        "pressure_level": PRESSURE_LEVELS,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    # 输出文件名
    filename = f"era5_pressure_{year}{month:02d}{day:02d}.nc"
    filepath = os.path.join(output_dir, filename)

    # 检查是否已存在
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"  [跳过] {filename} 已存在")
        return True

    try:
        print(f"  [下载] {filename}...")
        client.retrieve("reanalysis-era5-pressure-levels", request).download(filepath)
        print(f"  [完成] {filename}")
        return True
    except Exception as e:
        print(f"  [失败] {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def download_single_level(year, month, day, output_dir):
    """下载地面数据"""
    client = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "variable": SINGLE_LEVEL_VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": f"{day:02d}",
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    # 输出文件名
    filename = f"era5_single_{year}{month:02d}{day:02d}.nc"
    filepath = os.path.join(output_dir, filename)

    # 检查是否已存在
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"  [跳过] {filename} 已存在")
        return True

    try:
        print(f"  [下载] {filename}...")
        client.retrieve("reanalysis-era5-single-levels", request).download(filepath)
        print(f"  [完成] {filename}")
        return True
    except Exception as e:
        print(f"  [失败] {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def download_day(year, month, day, base_dir):
    """下载某一天的数据"""
    # 创建输出目录
    pressure_dir = os.path.join(base_dir, str(year), "pressure_level")
    single_dir = os.path.join(base_dir, str(year), "single_level")
    os.makedirs(pressure_dir, exist_ok=True)
    os.makedirs(single_dir, exist_ok=True)

    # 下载气压层数据
    pressure_ok = download_pressure_level(year, month, day, pressure_dir)
    time.sleep(1)  # 避免API过载

    # 下载地面数据
    single_ok = download_single_level(year, month, day, single_dir)
    time.sleep(1)

    return pressure_ok and single_ok


def main():
    parser = argparse.ArgumentParser(description="ERA5数据下载（简化版）")
    parser.add_argument("--start_year", type=int, default=1979, help="起始年份")
    parser.add_argument("--end_year", type=int, default=2005, help="结束年份")
    parser.add_argument("--single_year", type=int, help="只下载某一年")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="存储根目录")
    args = parser.parse_args()

    # 确定年份范围
    if args.single_year:
        start_year = args.single_year
        end_year = args.single_year
    else:
        start_year = args.start_year
        end_year = args.end_year

    print("=" * 60)
    print("ERA5数据下载（简化版）")
    print("=" * 60)
    print(f"年份范围: {start_year}-{end_year}")
    print(f"气压层: {PRESSURE_LEVELS}")
    print(f"变量数: {len(PRESSURE_VARIABLES)} (气压层) + {len(SINGLE_LEVEL_VARIABLES)} (地面)")
    print(f"时间分辨率: 每天4次 (00/06/12/18 UTC)")
    print(f"存储路径: {args.base_dir}")
    print("=" * 60)

    # 统计
    total_days = 0
    success_days = 0
    failed_days = 0

    # 按年-月-日遍历
    for year in range(start_year, end_year + 1):
        # 确定每月天数
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for month in range(1, 13):
            for day in range(1, days_in_month[month - 1] + 1):
                total_days += 1
                print(f"\n[{year}-{month:02d}-{day:02d}]")

                if download_day(year, month, day, args.base_dir):
                    success_days += 1
                else:
                    failed_days += 1

                # 进度
                progress = (success_days / total_days) * 100 if total_days > 0 else 0
                print(f"  进度: {success_days}/{total_days} ({progress:.1f}%)")

    # 总结
    print("\n" + "=" * 60)
    print("下载完成")
    print("=" * 60)
    print(f"总天数: {total_days}")
    print(f"成功: {success_days}")
    print(f"失败: {failed_days}")
    print("=" * 60)


if __name__ == "__main__":
    main()
