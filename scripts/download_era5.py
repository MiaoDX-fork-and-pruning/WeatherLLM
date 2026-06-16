#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Reanalysis Data Download Script

功能：
- 下载ERA5压力层数据 (Pressure Level)
- 下载ERA5单层数据 (Single Level)
- 支持多线程并行下载
- 自动跳过已下载文件
- 支持断点续传

使用前请配置CDS API：
1. 注册账号：https://cds.climate.copernicus.eu
2. 获取API Key：登录后点击用户名，找到API Key
3. 创建配置文件 ~/.cdsapi，内容如下：
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY

Author: Weather Project
Date: 2024
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import cdsapi
except ImportError:
    print("请先安装cdsapi: pip install cdsapi")
    sys.exit(1)

# ==================== 配置区域 ====================

# 默认配置
DEFAULT_CONFIG = {
    # 数据存储根目录（必须在F盘）
    "base_dir": r"F:\ERA5再分析数据下载",

    # 时间范围（按需下载，分阶段）
    "start_year": 2000,
    "end_year": 2005,  # 阶段1：先下载2000-2005年

    # 空间范围 (全球)
    "area": "90/-180/-90/180",  # North/West/South/East

    # 下载线程数 (建议不超过10，避免API限制)
    "num_threads": 5,

    # 下载间隔 (秒，避免请求过快)
    "download_interval": 1,

    # 重试次数
    "max_retries": 3,

    # 重试间隔 (秒)
    "retry_delay": 30,
}

# 压力层变量
PRESSURE_LEVEL_VARIABLES = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]

# 压力层列表
PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10",
    "20", "30", "50", "70", "100", "125",
    "150", "175", "200", "225", "250", "300",
    "350", "400", "450", "500", "550", "600",
    "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975", "1000"
]

# 单层变量
SINGLE_LEVEL_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation",
]

# 24小时时间列表
HOURLY_TIMES = [
    "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00", "21:00", "22:00", "23:00",
]

# ==================== 日志配置 ====================

def setup_logging(log_file: Optional[str] = None):
    """配置日志"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
    )

logger = logging.getLogger(__name__)

# ==================== 工具函数 ====================

def get_days_in_month(year: int, month: int) -> int:
    """获取指定年月的天数"""
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        else:
            return 28
    return 0

def generate_date_list(start_year: int, end_year: int) -> List[str]:
    """生成日期列表 (格式: YYYY-MM-DD)"""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            days = get_days_in_month(year, month)
            for day in range(1, days + 1):
                dates.append(f"{year}-{month:02d}-{day:02d}")
    return dates

def ensure_directory(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

# ==================== 下载器类 ====================

class ERA5Downloader:
    """ERA5数据下载器"""

    def __init__(self, config: dict):
        self.config = config
        self.base_dir = config["base_dir"]
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化CDS API客户端"""
        try:
            self.client = cdsapi.Client()
            logger.info("CDS API客户端初始化成功")
        except Exception as e:
            logger.error(f"CDS API客户端初始化失败: {e}")
            logger.error("请检查 ~/.cdsapi 配置文件是否正确")
            raise

    def _retry_download(self, func, *args, **kwargs):
        """带重试的下载函数"""
        max_retries = self.config["max_retries"]
        retry_delay = self.config["retry_delay"]

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"下载失败 (已达最大重试次数): {e}")
                    raise

    def download_pressure_level(self, date_str: str, output_dir: str) -> bool:
        """
        下载单日压力层数据

        Args:
            date_str: 日期字符串，格式 YYYY-MM-DD
            output_dir: 输出目录

        Returns:
            是否下载成功
        """
        year, month, day = date_str.split("-")
        filename = f"ERA5_0.25_PL_{date_str}.nc"
        filepath = os.path.join(output_dir, filename)

        # 检查文件是否已存在
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.debug(f"文件已存在，跳过: {filename}")
            return True

        logger.info(f"开始下载压力层数据: {date_str}")

        try:
            self._retry_download(
                self.client.retrieve,
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": PRESSURE_LEVEL_VARIABLES,
                    "year": year,
                    "month": month,
                    "day": day,
                    "time": HOURLY_TIMES,
                    "pressure_level": PRESSURE_LEVELS,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                filepath,
            )
            logger.info(f"下载完成: {filename}")
            return True

        except Exception as e:
            logger.error(f"下载失败 {date_str}: {e}")
            # 删除可能的不完整文件
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def download_single_level(self, date_str: str, output_dir: str) -> bool:
        """
        下载单日单层数据

        Args:
            date_str: 日期字符串，格式 YYYY-MM-DD
            output_dir: 输出目录

        Returns:
            是否下载成功
        """
        year, month, day = date_str.split("-")
        filename = f"ERA5_0.25_SL_{date_str}.nc"
        filepath = os.path.join(output_dir, filename)

        # 检查文件是否已存在
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.debug(f"文件已存在，跳过: {filename}")
            return True

        logger.info(f"开始下载单层数据: {date_str}")

        try:
            self._retry_download(
                self.client.retrieve,
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    "variable": SINGLE_LEVEL_VARIABLES,
                    "time": HOURLY_TIMES,
                    "day": [day],
                    "month": [month],
                    "year": [year],
                },
                filepath,
            )
            logger.info(f"下载完成: {filename}")
            return True

        except Exception as e:
            logger.error(f"下载失败 {date_str}: {e}")
            # 删除可能的不完整文件
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def download_pressure_level_batch(self, dates: List[str], num_threads: int = 5):
        """
        批量下载压力层数据 (多线程)

        Args:
            dates: 日期列表
            num_threads: 线程数
        """
        output_dir = os.path.join(self.base_dir, "pressure_level")
        ensure_directory(output_dir)

        # 统计信息
        total = len(dates)
        success_count = 0
        skip_count = 0
        fail_count = 0

        logger.info(f"开始批量下载压力层数据: 共 {total} 天, {num_threads} 线程")

        def worker(date_str):
            year = date_str.split("-")[0]
            year_dir = os.path.join(output_dir, year)
            ensure_directory(year_dir)
            return self.download_pressure_level(date_str, year_dir)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(worker, date): date for date in dates}

            for future in as_completed(futures):
                date = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"处理异常 {date}: {e}")
                    fail_count += 1

                # 下载间隔
                time.sleep(self.config["download_interval"])

        logger.info(f"压力层数据下载完成: 成功 {success_count}, 失败 {fail_count}")

    def download_single_level_batch(self, dates: List[str], num_threads: int = 5):
        """
        批量下载单层数据 (多线程)

        Args:
            dates: 日期列表
            num_threads: 线程数
        """
        output_dir = os.path.join(self.base_dir, "single_level")
        ensure_directory(output_dir)

        # 统计信息
        total = len(dates)
        success_count = 0
        fail_count = 0

        logger.info(f"开始批量下载单层数据: 共 {total} 天, {num_threads} 线程")

        def worker(date_str):
            year = date_str.split("-")[0]
            year_dir = os.path.join(output_dir, year)
            ensure_directory(year_dir)
            return self.download_single_level(date_str, year_dir)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(worker, date): date for date in dates}

            for future in as_completed(futures):
                date = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"处理异常 {date}: {e}")
                    fail_count += 1

                # 下载间隔
                time.sleep(self.config["download_interval"])

        logger.info(f"单层数据下载完成: 成功 {success_count}, 失败 {fail_count}")

    def get_download_progress(self, data_type: str = "pressure_level"):
        """获取下载进度"""
        output_dir = os.path.join(self.base_dir, data_type)
        if not os.path.exists(output_dir):
            return {"total": 0, "downloaded": 0, "progress": 0}

        dates = generate_date_list(self.config["start_year"], self.config["end_year"])
        total = len(dates)
        downloaded = 0

        for date in dates:
            year = date.split("-")[0]
            if data_type == "pressure_level":
                filename = f"ERA5_0.25_PL_{date}.nc"
            else:
                filename = f"ERA5_0.25_SL_{date}.nc"

            filepath = os.path.join(output_dir, year, filename)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                downloaded += 1

        progress = (downloaded / total * 100) if total > 0 else 0

        return {
            "total": total,
            "downloaded": downloaded,
            "remaining": total - downloaded,
            "progress": round(progress, 2),
        }

# ==================== 命令行接口 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ERA5 Reanalysis Data Download Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载所有数据 (2000-2024)
  python download_era5.py --all

  # 下载压力层数据
  python download_era5.py --pressure-level

  # 下载单层数据
  python download_era5.py --single-level

  # 下载指定年份范围
  python download_era5.py --all --start-year 2010 --end-year 2020

  # 查看下载进度
  python download_era5.py --progress

  # 使用自定义输出目录
  python download_era5.py --all --output-dir /path/to/data
        """,
    )

    # 下载选项
    download_group = parser.add_argument_group("下载选项")
    download_group.add_argument(
        "--all", action="store_true", help="下载所有数据类型"
    )
    download_group.add_argument(
        "--pressure-level", action="store_true", help="仅下载压力层数据"
    )
    download_group.add_argument(
        "--single-level", action="store_true", help="仅下载单层数据"
    )

    # 时间范围
    time_group = parser.add_argument_group("时间范围")
    time_group.add_argument(
        "--start-year", type=int, default=DEFAULT_CONFIG["start_year"],
        help=f"起始年份 (默认: {DEFAULT_CONFIG['start_year']})"
    )
    time_group.add_argument(
        "--end-year", type=int, default=DEFAULT_CONFIG["end_year"],
        help=f"结束年份 (默认: {DEFAULT_CONFIG['end_year']})"
    )

    # 下载设置
    setting_group = parser.add_argument_group("下载设置")
    setting_group.add_argument(
        "--threads", type=int, default=DEFAULT_CONFIG["num_threads"],
        help=f"下载线程数 (默认: {DEFAULT_CONFIG['num_threads']})"
    )
    setting_group.add_argument(
        "--output-dir", type=str, default=DEFAULT_CONFIG["base_dir"],
        help=f"输出目录 (默认: {DEFAULT_CONFIG['base_dir']})"
    )

    # 其他选项
    other_group = parser.add_argument_group("其他选项")
    other_group.add_argument(
        "--progress", action="store_true", help="查看下载进度"
    )
    other_group.add_argument(
        "--log-file", type=str, help="日志文件路径"
    )

    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(args.log_file)

    # 构建配置
    config = DEFAULT_CONFIG.copy()
    config["base_dir"] = args.output_dir
    config["start_year"] = args.start_year
    config["end_year"] = args.end_year
    config["num_threads"] = args.threads

    # 创建下载器
    downloader = ERA5Downloader(config)

    # 查看进度
    if args.progress:
        for data_type in ["pressure_level", "single_level"]:
            progress = downloader.get_download_progress(data_type)
            logger.info(f"{data_type}: {progress['downloaded']}/{progress['total']} "
                       f"({progress['progress']}%)")
        return

    # 生成日期列表
    dates = generate_date_list(args.start_year, args.end_year)
    logger.info(f"日期范围: {args.start_year}-01-01 至 {args.end_year}-12-31")
    logger.info(f"总天数: {len(dates)}")

    # 开始下载
    start_time = time.time()

    try:
        if args.all or args.pressure_level:
            downloader.download_pressure_level_batch(dates, args.threads)

        if args.all or args.single_level:
            downloader.download_single_level_batch(dates, args.threads)

        if not (args.all or args.pressure_level or args.single_level):
            logger.warning("请指定下载类型: --all, --pressure-level, 或 --single-level")
            logger.info("使用 --help 查看帮助")
            return

    except KeyboardInterrupt:
        logger.info("用户中断下载")
    except Exception as e:
        logger.error(f"下载过程中发生错误: {e}")
        raise

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    logger.info(f"下载完成! 总耗时: {hours}小时 {minutes}分钟 {seconds}秒")

if __name__ == "__main__":
    main()
