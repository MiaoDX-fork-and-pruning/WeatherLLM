#!/usr/bin/env python3
"""
ERA5数据下载脚本 - 改进版
支持压力层和单层数据，带进度追踪和错误重试

用法:
    python download_era5_improved.py --start_year 2000 --end_year 2017 --data_type pressure_level
    python download_era5_improved.py --start_year 2018 --end_year 2018 --data_type single_level
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import cdsapi
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('era5_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据目录配置
BASE_DIR = Path(r"F:\ERA5再分析数据下载")
PROGRESS_FILE = BASE_DIR / "download_progress_v2.json"

# ERA5变量配置
PRESSURE_LEVEL_VARIABLES = [
    "geopotential", "temperature", "specific_humidity",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity"
]

SINGLE_LEVEL_VARIABLES = [
    "2m_temperature", "surface_pressure", "total_precipitation",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "cloud_cover"
]

# 压力层配置（37层）
PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10",
    "20", "30", "50", "70", "100", "125",
    "150", "175", "200", "225", "250", "300",
    "350", "400", "450", "500", "550", "600",
    "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975", "1000"
]


class ERA5Downloader:
    def __init__(self, start_year, end_year, data_type='pressure_level'):
        self.start_year = start_year
        self.end_year = end_year
        self.data_type = data_type
        self.client = cdsapi.Client()
        self.progress = self.load_progress()

    def load_progress(self):
        """加载下载进度"""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'years': {}
        }

    def save_progress(self):
        """保存下载进度"""
        self.progress['last_update'] = datetime.now().isoformat()
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def get_days_in_month(self, year, month):
        """获取某月天数"""
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        current_month = datetime(year, month, 1)
        return (next_month - current_month).days

    def get_download_dir(self, year):
        """获取下载目录"""
        if self.data_type == 'pressure_level':
            return BASE_DIR / "pressure_level" / str(year)
        else:
            return BASE_DIR / "single_level" / str(year)

    def get_filename(self, year, month, day, hour):
        """生成文件名"""
        month_str = f"{month:02d}"
        day_str = f"{day:02d}"
        hour_str = f"{hour:02d}"

        if self.data_type == 'pressure_level':
            return f"ERA5_0.25_PL_{year}-{month_str}-{day_str}-{hour_str}.nc"
        else:
            return f"ERA5_0.25_SL_{year}-{month_str}-{day_str}-{hour_str}.nc"

    def download_pressure_level(self, year, month, day, hour, output_path):
        """下载压力层数据"""
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        time_str = f"{hour:02d}:00"

        request = {
            "product_type": ["reanalysis"],
            "variable": PRESSURE_LEVEL_VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{day:02d}"],
            "time": [time_str],
            "pressure_level": PRESSURE_LEVELS,
            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        self.client.retrieve('reanalysis-era5-pressure-levels', request, str(output_path))

    def download_single_level(self, year, month, day, hour, output_path):
        """下载单层数据"""
        request = {
            "product_type": ["reanalysis"],
            "variable": SINGLE_LEVEL_VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{day:02d}"],
            "time": [f"{hour:02d}:00"],
            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        self.client.retrieve('reanalysis-era5-single-levels', request, str(output_path))

    def download_one_file(self, year, month, day, hour, max_retries=3):
        """下载单个文件，带重试机制"""
        download_dir = self.get_download_dir(year)
        download_dir.mkdir(parents=True, exist_ok=True)

        filename = self.get_filename(year, month, day, hour)
        output_path = download_dir / filename

        # 检查文件是否已存在
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"文件已存在，跳过: {filename}")
            return True

        for attempt in range(max_retries):
            try:
                logger.info(f"下载: {filename} (尝试 {attempt + 1}/{max_retries})")

                if self.data_type == 'pressure_level':
                    self.download_pressure_level(year, month, day, hour, output_path)
                else:
                    self.download_single_level(year, month, day, hour, output_path)

                # 验证文件
                if output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"下载成功: {filename}")
                    return True
                else:
                    logger.warning(f"文件为空或不存在: {filename}")

            except Exception as e:
                logger.error(f"下载失败 {filename}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 30  # 指数退避
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        return False

    def download_year(self, year):
        """下载一年的数据"""
        logger.info(f"开始下载 {year} 年 {self.data_type} 数据")

        # 初始化进度
        if str(year) not in self.progress['years']:
            self.progress['years'][str(year)] = {
                'total': 0,
                'completed': [],
                'failed': [],
                'skipped': []
            }

        year_progress = self.progress['years'][str(year)][self.data_type] if self.data_type in self.progress['years'][str(year)] else {
            'total': 0,
            'completed': [],
            'failed': [],
            'skipped': []
        }
        self.progress['years'][str(year)][self.data_type] = year_progress

        # 计算总任务数
        total_tasks = 0
        for month in range(1, 13):
            days = self.get_days_in_month(year, month)
            total_tasks += days * 24  # 每天24小时

        year_progress['total'] = total_tasks
        completed_count = len(year_progress['completed'])
        failed_count = len(year_progress['failed'])

        logger.info(f"总任务数: {total_tasks}, 已完成: {completed_count}, 失败: {failed_count}")

        # 下载数据
        for month in range(1, 13):
            days = self.get_days_in_month(year, month)

            for day in range(1, days + 1):
                for hour in range(24):
                    date_key = f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}"

                    # 跳过已完成的
                    if date_key in year_progress['completed']:
                        continue

                    # 尝试下载
                    success = self.download_one_file(year, month, day, hour)

                    if success:
                        if date_key not in year_progress['completed']:
                            year_progress['completed'].append(date_key)
                        if date_key in year_progress['failed']:
                            year_progress['failed'].remove(date_key)
                    else:
                        if date_key not in year_progress['failed']:
                            year_progress['failed'].append(date_key)

                    # 定期保存进度
                    if len(year_progress['completed']) % 100 == 0:
                        self.save_progress()
                        logger.info(f"进度: {len(year_progress['completed'])}/{total_tasks} "
                                  f"({len(year_progress['completed'])/total_tasks*100:.1f}%)")

        self.save_progress()
        logger.info(f"{year} 年下载完成")

    def run(self):
        """主运行函数"""
        logger.info(f"开始下载 ERA5 数据")
        logger.info(f"年份范围: {self.start_year} - {self.end_year}")
        logger.info(f"数据类型: {self.data_type}")

        start_time = time.time()

        for year in range(self.start_year, self.end_year + 1):
            self.download_year(year)

        elapsed_time = time.time() - start_time
        logger.info(f"下载完成! 总耗时: {elapsed_time/3600:.2f} 小时")

        # 打印统计
        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        logger.info("\n" + "="*60)
        logger.info("下载统计")
        logger.info("="*60)

        total_completed = 0
        total_failed = 0

        for year_str, year_data in self.progress['years'].items():
            if self.data_type in year_data:
                year_stats = year_data[self.data_type]
                completed = len(year_stats.get('completed', []))
                failed = len(year_stats.get('failed', []))
                total_completed += completed
                total_failed += failed
                logger.info(f"{year_str}: 完成 {completed}, 失败 {failed}")

        logger.info("-"*60)
        logger.info(f"总计: 完成 {total_completed}, 失败 {total_failed}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='ERA5数据下载脚本')
    parser.add_argument('--start_year', type=int, required=True, help='起始年份')
    parser.add_argument('--end_year', type=int, required=True, help='结束年份')
    parser.add_argument('--data_type', choices=['pressure_level', 'single_level'],
                       default='pressure_level', help='数据类型')

    args = parser.parse_args()

    downloader = ERA5Downloader(args.start_year, args.end_year, args.data_type)
    downloader.run()


if __name__ == '__main__':
    main()
