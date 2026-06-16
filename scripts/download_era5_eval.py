#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA5 Evaluation Data Download Script (2018-2019)

评测数据专用下载脚本：
- 2018年数据：与 GraphCast / Pangu-Wind 对比评测（均使用2018年）
- 2019年数据：与 GenCast 对比评测（GenCast使用2019年）
- 与训练数据相同的29个变量（5个压力层变量 + 5个单层变量）
- 数据存储路径：F:\\ERA5再分析数据下载\\eval\\{2018,2019}

功能：
- 支持断点续传：自动跳过已下载文件
- 多线程并行下载：提高下载效率
- 下载进度跟踪：实时显示和持久化记录
- 失败重试机制：应对网络波动
- License检查：自动检测CDS API许可证是否已接受

使用前请配置CDS API：
1. 注册账号：https://cds.climate.copernicus.eu
2. 获取API Key：登录后点击用户名，找到API Key
3. 创建配置文件 ~/.cdsapi，内容如下：
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY

还需要接受数据集许可证：
- 压力层: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download#manage-licences
- 单层: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences

Author: Weather Project
Date: 2026-06
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import cdsapi
except ImportError:
    print("请先安装cdsapi: pip install cdsapi")
    sys.exit(1)


# ==================== 配置区域 ====================

# 评测数据下载默认配置
DEFAULT_CONFIG = {
    # 评测数据根目录（存储在F盘）
    "base_dir": r"F:\ERA5再分析数据下载\eval",

    # 评测年份（GenCast用2019，GraphCast/Pangu用2018）
    "years": [2018, 2019],

    # 空间范围（全球）
    "area": "90/-180/-90/180",  # North/West/South/East

    # 下载线程数（建议不超过10，避免API限制）
    "num_threads": 5,

    # 下载间隔（秒，避免请求过快）
    "download_interval": 1,

    # 重试次数
    "max_retries": 3,

    # 重试间隔（秒）
    "retry_delay": 30,

    # 进度跟踪文件
    "progress_file": r"F:\ERA5再分析数据下载\eval\download_progress.json",
}

# ==================== 变量定义（与训练数据相同的29个变量） ====================

# 压力层变量（5个变量）
PRESSURE_LEVEL_VARIABLES = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]

# 压力层列表（37层，GenCast/Pangu/GraphCast均使用此标准层）
PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10",
    "20", "30", "50", "70", "100", "125",
    "150", "175", "200", "225", "250", "300",
    "350", "400", "450", "500", "550", "600",
    "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975", "1000",
]

# 单层变量（5个变量）
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


def generate_date_list(year: int) -> List[str]:
    """生成指定年份的日期列表（格式: YYYY-MM-DD）"""
    dates = []
    for month in range(1, 13):
        days = get_days_in_month(year, month)
        for day in range(1, days + 1):
            dates.append(f"{year}-{month:02d}-{day:02d}")
    return dates


def ensure_directory(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(filepath: str) -> float:
    """获取文件大小（MB）"""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


# ==================== 进度跟踪器 ====================

class DownloadProgressTracker:
    """
    下载进度跟踪器

    功能：
    - 实时统计下载进度
    - 持久化进度到JSON文件（支持断点续传）
    - 记录失败文件以便重试
    """

    def __init__(self, progress_file: str, years: List[int]):
        self.progress_file = progress_file
        self.years = years
        self.lock = threading.Lock()

        # 进度数据
        self.progress_data = {
            "start_time": None,
            "last_update": None,
            "years": {},
        }

        # 为每个年份初始化
        for year in years:
            self.progress_data["years"][str(year)] = {
                "pressure_level": {
                    "total": 0,
                    "completed": [],
                    "failed": [],
                    "skipped": [],
                },
                "single_level": {
                    "total": 0,
                    "completed": [],
                    "failed": [],
                    "skipped": [],
                },
            }

        # 尝试加载已有进度
        self._load_progress()

    def _load_progress(self):
        """从文件加载已有进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)

                # 合并已有进度
                for year_str in self.years:
                    year_key = str(year_str)
                    if year_key in saved.get("years", {}):
                        self.progress_data["years"][year_key] = saved["years"][year_key]

                self.progress_data["start_time"] = saved.get("start_time")
                logger.info(f"已加载下载进度: {self.progress_file}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"加载进度文件失败，使用默认进度: {e}")

    def _save_progress(self):
        """保存进度到文件"""
        self.progress_data["last_update"] = datetime.now().isoformat()

        try:
            ensure_directory(os.path.dirname(self.progress_file))
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存进度文件失败: {e}")

    def start(self):
        """记录下载开始时间"""
        if self.progress_data["start_time"] is None:
            self.progress_data["start_time"] = datetime.now().isoformat()
        self._save_progress()

    def init_year_data(self, year: int, data_type: str, dates: List[str]):
        """初始化某年份某类型数据的日期列表"""
        year_key = str(year)
        with self.lock:
            self.progress_data["years"][year_key][data_type]["total"] = len(dates)
            # 初始化未完成列表（排除已完成的）
            completed = set(self.progress_data["years"][year_key][data_type]["completed"])
            self.progress_data["years"][year_key][data_type]["remaining"] = [
                d for d in dates if d not in completed
            ]
        self._save_progress()

    def mark_completed(self, year: int, data_type: str, date_str: str):
        """标记某天数据下载完成"""
        year_key = str(year)
        with self.lock:
            completed = self.progress_data["years"][year_key][data_type]["completed"]
            if date_str not in completed:
                completed.append(date_str)
            # 从失败列表移除
            failed = self.progress_data["years"][year_key][data_type]["failed"]
            if date_str in failed:
                failed.remove(date_str)
        self._save_progress()

    def mark_failed(self, year: int, data_type: str, date_str: str):
        """标记某天数据下载失败"""
        year_key = str(year)
        with self.lock:
            failed = self.progress_data["years"][year_key][data_type]["failed"]
            if date_str not in failed:
                failed.append(date_str)
        self._save_progress()

    def mark_skipped(self, year: int, data_type: str, date_str: str):
        """标记某天数据已跳过（已存在）"""
        year_key = str(year)
        with self.lock:
            skipped = self.progress_data["years"][year_key][data_type]["skipped"]
            if date_str not in skipped:
                skipped.append(date_str)
        self._save_progress()

    def get_summary(self) -> Dict:
        """获取下载进度摘要"""
        summary = {
            "total_completed": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "total_expected": 0,
            "by_year": {},
        }

        for year_str in self.years:
            year_key = str(year_str)
            year_data = self.progress_data["years"][year_key]
            year_summary = {"pressure_level": {}, "single_level": {}}

            for data_type in ["pressure_level", "single_level"]:
                type_data = year_data[data_type]
                completed = len(type_data["completed"])
                failed = len(type_data["failed"])
                skipped = len(type_data["skipped"])
                total = type_data["total"]

                year_summary[data_type] = {
                    "completed": completed,
                    "failed": failed,
                    "skipped": skipped,
                    "total": total,
                    "progress": round(completed / total * 100, 1) if total > 0 else 0,
                }

                summary["total_completed"] += completed
                summary["total_failed"] += failed
                summary["total_skipped"] += skipped
                summary["total_expected"] += total

            summary["by_year"][year_key] = year_summary

        total_all = summary["total_expected"]
        completed_all = summary["total_completed"] + summary["total_skipped"]
        summary["overall_progress"] = round(
            completed_all / total_all * 100, 1
        ) if total_all > 0 else 0

        return summary

    def print_summary(self):
        """打印下载进度摘要"""
        summary = self.get_summary()

        logger.info("=" * 70)
        logger.info("ERA5 评测数据下载进度")
        logger.info("=" * 70)

        for year_str in self.years:
            year_key = str(year_str)
            logger.info(f"\n--- {year_key}年 ---")
            year_summary = summary["by_year"][year_key]

            for data_type, label in [
                ("pressure_level", "压力层"),
                ("single_level", "单层"),
            ]:
                type_summary = year_summary[data_type]
                logger.info(
                    f"  {label}: {type_summary['completed']}/{type_summary['total']} "
                    f"({type_summary['progress']}%) "
                    f"失败: {type_summary['failed']} 跳过: {type_summary['skipped']}"
                )

        logger.info(f"\n总计: {summary['total_completed']}/{summary['total_expected']} "
                     f"({summary['overall_progress']}%)")
        logger.info("=" * 70)


# ==================== 下载器类 ====================

class ERA5EvalDownloader:
    """
    ERA5 评测数据下载器

    专门用于下载2018-2019年评测数据，与训练数据使用相同变量。
    支持断点续传、多线程下载和进度跟踪。
    """

    def __init__(self, config: dict):
        self.config = config
        self.base_dir = config["base_dir"]
        self.years = config["years"]
        self.client = None

        # 进度跟踪器
        self.tracker = DownloadProgressTracker(
            config["progress_file"], self.years
        )

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
                error_msg = str(e)

                # 检查是否是许可证问题
                if "required licences not accepted" in error_msg or "403" in error_msg:
                    logger.error("=" * 60)
                    logger.error("错误：CDS API许可证未接受！")
                    logger.error("=" * 60)
                    logger.error("请按以下步骤接受许可证：")
                    logger.error("1. 访问 https://cds.climate.copernicus.eu")
                    logger.error("2. 登录您的CDS账号")
                    logger.error("3. 访问以下链接接受许可证：")
                    logger.error(
                        "   - 压力层数据: https://cds.climate.copernicus.eu/datasets/"
                        "reanalysis-era5-pressure-levels?tab=download#manage-licences"
                    )
                    logger.error(
                        "   - 单层数据: https://cds.climate.copernicus.eu/datasets/"
                        "reanalysis-era5-single-levels?tab=download#manage-licences"
                    )
                    logger.error("4. 接受许可证后重新运行脚本")
                    logger.error("=" * 60)
                    raise

                if attempt < max_retries - 1:
                    logger.warning(
                        f"下载失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                    )
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"下载失败 (已达最大重试次数): {e}")
                    raise

    def download_pressure_level(self, year: int, date_str: str) -> bool:
        """
        下载单日压力层数据

        Args:
            year: 年份
            date_str: 日期字符串（格式: YYYY-MM-DD）

        Returns:
            是否下载成功
        """
        _, month, day = date_str.split("-")
        filename = f"ERA5_0.25_PL_{date_str}.nc"
        year_dir = os.path.join(self.base_dir, str(year))
        filepath = os.path.join(year_dir, filename)

        # 断点续传：检查文件是否已存在且完整
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.debug(f"文件已存在，跳过: {filename}")
            self.tracker.mark_skipped(year, "pressure_level", date_str)
            return True

        logger.info(f"[{year}] 下载压力层数据: {date_str}")

        try:
            ensure_directory(year_dir)
            self._retry_download(
                self.client.retrieve,
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": PRESSURE_LEVEL_VARIABLES,
                    "year": str(year),
                    "month": month,
                    "day": day,
                    "time": HOURLY_TIMES,
                    "pressure_level": PRESSURE_LEVELS,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                filepath,
            )
            file_size = get_file_size_mb(filepath)
            logger.info(
                f"[{year}] 下载完成: {filename} ({file_size:.1f} MB)"
            )
            self.tracker.mark_completed(year, "pressure_level", date_str)
            return True

        except Exception as e:
            logger.error(f"[{year}] 下载失败 {date_str}: {e}")
            # 删除可能的不完整文件
            if os.path.exists(filepath):
                os.remove(filepath)
            self.tracker.mark_failed(year, "pressure_level", date_str)
            return False

    def download_single_level(self, year: int, date_str: str) -> bool:
        """
        下载单日单层数据

        Args:
            year: 年份
            date_str: 日期字符串（格式: YYYY-MM-DD）

        Returns:
            是否下载成功
        """
        _, month, day = date_str.split("-")
        filename = f"ERA5_0.25_SL_{date_str}.nc"
        year_dir = os.path.join(self.base_dir, str(year))
        filepath = os.path.join(year_dir, filename)

        # 断点续传：检查文件是否已存在且完整
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.debug(f"文件已存在，跳过: {filename}")
            self.tracker.mark_skipped(year, "single_level", date_str)
            return True

        logger.info(f"[{year}] 下载单层数据: {date_str}")

        try:
            ensure_directory(year_dir)
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
                    "year": [str(year)],
                },
                filepath,
            )
            file_size = get_file_size_mb(filepath)
            logger.info(
                f"[{year}] 下载完成: {filename} ({file_size:.1f} MB)"
            )
            self.tracker.mark_completed(year, "single_level", date_str)
            return True

        except Exception as e:
            logger.error(f"[{year}] 下载失败 {date_str}: {e}")
            # 删除可能的不完整文件
            if os.path.exists(filepath):
                os.remove(filepath)
            self.tracker.mark_failed(year, "single_level", date_str)
            return False

    def download_year(self, year: int, data_type: str, num_threads: int):
        """
        下载指定年份和类型的全部数据

        Args:
            year: 年份
            data_type: 数据类型（"pressure_level" 或 "single_level"）
            num_threads: 线程数
        """
        dates = generate_date_list(year)

        # 初始化进度
        self.tracker.init_year_data(year, data_type, dates)

        # 过滤已完成的日期
        completed = set(
            self.tracker.progress_data["years"][str(year)][data_type]["completed"]
        )
        skipped = set(
            self.tracker.progress_data["years"][str(year)][data_type]["skipped"]
        )
        remaining = [d for d in dates if d not in completed and d not in skipped]

        total = len(dates)
        done_count = len(completed) + len(skipped)
        logger.info(
            f"\n[{year}] {data_type} 数据: "
            f"总计 {total} 天, 已完成 {done_count}, 剩余 {len(remaining)} 天"
        )

        if not remaining:
            logger.info(f"[{year}] {data_type} 数据已全部下载完成，跳过")
            return

        # 选择下载函数
        if data_type == "pressure_level":
            download_func = self.download_pressure_level
        else:
            download_func = self.download_single_level

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(download_func, year, date): date
                for date in remaining
            }

            completed_count = 0
            failed_count = 0

            for future in as_completed(futures):
                date = futures[future]
                try:
                    result = future.result()
                    if result:
                        completed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"处理异常 {date}: {e}")
                    failed_count += 1

                # 下载间隔
                time.sleep(self.config["download_interval"])

        elapsed = time.time() - start_time
        logger.info(
            f"[{year}] {data_type} 下载完成: "
            f"成功 {completed_count}, 失败 {failed_count}, "
            f"耗时 {elapsed:.0f} 秒"
        )

    def download_all(self, data_types: Optional[List[str]] = None, num_threads: int = 5):
        """
        下载全部评测数据

        Args:
            data_types: 数据类型列表，None表示全部
            num_threads: 线程数
        """
        if data_types is None:
            data_types = ["pressure_level", "single_level"]

        self.tracker.start()

        logger.info("=" * 70)
        logger.info("ERA5 评测数据下载 - 开始")
        logger.info(f"年份: {self.years}")
        logger.info(f"数据类型: {data_types}")
        logger.info(f"线程数: {num_threads}")
        logger.info(f"存储路径: {self.base_dir}")
        logger.info("=" * 70)

        start_time = time.time()

        for year in self.years:
            for data_type in data_types:
                self.download_year(year, data_type, num_threads)

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        logger.info("\n" + "=" * 70)
        logger.info(f"全部下载完成! 总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
        logger.info("=" * 70)

        # 打印最终进度
        self.tracker.print_summary()

    def retry_failed(self, data_types: Optional[List[str]] = None, num_threads: int = 5):
        """
        重试失败的下载

        Args:
            data_types: 数据类型列表，None表示全部
            num_threads: 线程数
        """
        if data_types is None:
            data_types = ["pressure_level", "single_level"]

        logger.info("开始重试失败的下载...")

        for year in self.years:
            year_key = str(year)
            for data_type in data_types:
                failed = self.tracker.progress_data["years"][year_key][data_type][
                    "failed"
                ].copy()

                if not failed:
                    continue

                logger.info(f"[{year}] 重试 {data_type}: {len(failed)} 天")

                # 清空失败列表，让下载函数重新记录
                self.tracker.progress_data["years"][year_key][data_type][
                    "failed"
                ] = []

                for date in failed:
                    if data_type == "pressure_level":
                        self.download_pressure_level(year, date)
                    else:
                        self.download_single_level(year, date)

                    time.sleep(self.config["download_interval"])


# ==================== 命令行接口 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ERA5 Evaluation Data Download Script (2018-2019)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
评测背景说明：
  - 2018年数据：用于与 GraphCast / Pangu-Wind 对比评测
  - 2019年数据：用于与 GenCast 对比评测
  - 所有模型使用相同的ERA5再分析数据作为ground truth

使用示例:
  # 下载全部评测数据（2018-2019年）
  python download_era5_eval.py --all

  # 仅下载压力层数据
  python download_era5_eval.py --pressure-level

  # 仅下载单层数据
  python download_era5_eval.py --single-level

  # 仅下载2019年数据（GenCast评测）
  python download_era5_eval.py --all --years 2019

  # 仅下载2018年数据（GraphCast/Pangu评测）
  python download_era5_eval.py --all --years 2018

  # 重试失败的下载
  python download_era5_eval.py --retry

  # 查看下载进度
  python download_era5_eval.py --progress

  # 使用自定义输出目录
  python download_era5_eval.py --all --output-dir /path/to/eval/data
        """,
    )

    # 下载选项
    download_group = parser.add_argument_group("下载选项")
    download_group.add_argument(
        "--all", action="store_true", help="下载所有数据类型（压力层+单层）"
    )
    download_group.add_argument(
        "--pressure-level", action="store_true", help="仅下载压力层数据"
    )
    download_group.add_argument(
        "--single-level", action="store_true", help="仅下载单层数据"
    )
    download_group.add_argument(
        "--retry", action="store_true", help="重试之前失败的下载"
    )

    # 年份选择
    year_group = parser.add_argument_group("年份选择")
    year_group.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_CONFIG["years"],
        help=f"指定下载的年份 (默认: {DEFAULT_CONFIG['years']})",
    )

    # 下载设置
    setting_group = parser.add_argument_group("下载设置")
    setting_group.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_CONFIG["num_threads"],
        help=f"下载线程数 (默认: {DEFAULT_CONFIG['num_threads']})",
    )
    setting_group.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_CONFIG["base_dir"],
        help=f"输出目录 (默认: {DEFAULT_CONFIG['base_dir']})",
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
    config["years"] = args.years
    config["num_threads"] = args.threads

    # 更新进度文件路径
    config["progress_file"] = os.path.join(
        args.output_dir, "download_progress.json"
    )

    # 创建下载器
    downloader = ERA5EvalDownloader(config)

    # 查看进度
    if args.progress:
        downloader.tracker.print_summary()
        return

    # 确定下载类型
    data_types = []
    if args.all or args.pressure_level:
        data_types.append("pressure_level")
    if args.all or args.single_level:
        data_types.append("single_level")

    if not data_types and not args.retry:
        logger.warning("请指定下载类型: --all, --pressure-level, 或 --single-level")
        logger.info("使用 --help 查看帮助")
        return

    # 执行下载
    start_time = time.time()

    try:
        if args.retry:
            downloader.retry_failed(data_types or None, args.threads)
        else:
            downloader.download_all(data_types, args.threads)

    except KeyboardInterrupt:
        logger.info("用户中断下载")
        logger.info("进度已保存，下次运行将自动续传")
        downloader.tracker.print_summary()
    except Exception as e:
        logger.error(f"下载过程中发生错误: {e}")
        logger.info("进度已保存，可使用 --retry 重试失败的下载")
        raise

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    logger.info(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")


if __name__ == "__main__":
    main()
