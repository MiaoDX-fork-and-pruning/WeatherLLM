#!/usr/bin/env python3
"""
ERA5优化下载脚本
减少并发数，添加请求间隔，避免被CDS限流

用法:
    python download_era5_optimized.py --start_year 2000 --end_year 2000 --workers 2
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cdsapi
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('era5_download_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据目录配置
BASE_DIR = Path(r"F:\ERA5再分析数据下载")
PROGRESS_FILE = BASE_DIR / "download_progress_optimized.json"

# ERA5变量配置
PRESSURE_LEVEL_VARIABLES = [
    "geopotential", "temperature", "specific_humidity",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity"
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


class ERA5OptimizedDownloader:
    def __init__(self, start_year, end_year, workers=2, delay=5):
        self.start_year = start_year
        self.end_year = end_year
        self.workers = workers
        self.delay = delay  # 请求间隔（秒）
        self.progress = self.load_progress()
        self.stats = {'completed': 0, 'failed': 0, 'skipped': 0}
        self.last_request_time = 0

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

    def generate_download_tasks(self, year):
        """生成下载任务列表"""
        tasks = []
        for month in range(1, 13):
            days = self.get_days_in_month(year, month)
            for day in range(1, days + 1):
                for hour in range(24):
                    tasks.append({
                        'year': year,
                        'month': month,
                        'day': day,
                        'hour': hour,
                        'date_key': f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}"
                    })
        return tasks

    def wait_for_rate_limit(self):
        """等待以避免速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay:
            sleep_time = self.delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def download_one_file(self, task, max_retries=3):
        """下载单个文件"""
        year = task['year']
        month = task['month']
        day = task['day']
        hour = task['hour']
        date_key = task['date_key']

        # 检查是否已完成
        year_str = str(year)
        if year_str in self.progress['years']:
            year_data = self.progress['years'][year_str]
            if 'pressure_level' in year_data:
                if date_key in year_data['pressure_level'].get('completed', []):
                    self.stats['skipped'] += 1
                    return True

        # 创建输出目录
        download_dir = BASE_DIR / "pressure_level" / str(year)
        download_dir.mkdir(parents=True, exist_ok=True)

        filename = f"ERA5_0.25_PL_{year:04d}-{month:02d}-{day:02d}-{hour:02d}.nc"
        output_path = download_dir / filename

        # 检查文件是否已存在
        if output_path.exists() and output_path.stat().st_size > 0:
            self.stats['skipped'] += 1
            return True

        # 等待速率限制
        self.wait_for_rate_limit()

        # 尝试下载
        for attempt in range(max_retries):
            try:
                c = cdsapi.Client()

                c.retrieve('reanalysis-era5-pressure-levels', {
                    "product_type": ["reanalysis"],
                    "variable": PRESSURE_LEVEL_VARIABLES,
                    "year": [str(year)],
                    "month": [f"{month:02d}"],
                    "day": [f"{day:02d}"],
                    "time": [f"{hour:02d}:00"],
                    "pressure_level": PRESSURE_LEVELS,
                    "data_format": "netcdf",
                    "download_format": "unarchived"
                }, str(output_path))

                # 验证文件
                if output_path.exists() and output_path.stat().st_size > 0:
                    self.stats['completed'] += 1
                    return True

            except Exception as e:
                logger.warning(f"下载失败 {filename} (尝试 {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    # 指数退避，但最少等待10秒
                    wait_time = max(10, 2 ** attempt * 10)
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        self.stats['failed'] += 1
        return False

    def update_progress(self, year, date_key, success):
        """更新进度"""
        year_str = str(year)
        if year_str not in self.progress['years']:
            self.progress['years'][year_str] = {}

        if 'pressure_level' not in self.progress['years'][year_str]:
            self.progress['years'][year_str]['pressure_level'] = {
                'completed': [],
                'failed': []
            }

        pl_data = self.progress['years'][year_str]['pressure_level']

        if success:
            if date_key not in pl_data['completed']:
                pl_data['completed'].append(date_key)
            if date_key in pl_data['failed']:
                pl_data['failed'].remove(date_key)
        else:
            if date_key not in pl_data['failed']:
                pl_data['failed'].append(date_key)

    def download_year(self, year):
        """下载一年的数据"""
        logger.info(f"开始下载 {year} 年数据 (workers={self.workers}, delay={self.delay}s)")

        tasks = self.generate_download_tasks(year)
        total_tasks = len(tasks)
        logger.info(f"总任务数: {total_tasks}")

        # 过滤已完成的任务
        pending_tasks = []
        for task in tasks:
            year_str = str(year)
            if year_str in self.progress['years']:
                year_data = self.progress['years'][year_str]
                if 'pressure_level' in year_data:
                    if task['date_key'] in year_data['pressure_level'].get('completed', []):
                        self.stats['skipped'] += 1
                        continue
            pending_tasks.append(task)

        logger.info(f"待下载: {len(pending_tasks)}, 已跳过: {self.stats['skipped']}")

        # 并行下载（减少并发）
        completed_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_task = {
                executor.submit(self.download_one_file, task): task
                for task in pending_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    success = future.result()
                    self.update_progress(year, task['date_key'], success)

                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1

                    # 定期保存进度
                    total_done = completed_count + failed_count
                    if total_done % 20 == 0:
                        self.save_progress()
                        progress_pct = (self.stats['skipped'] + completed_count) / total_tasks * 100
                        logger.info(f"进度: {self.stats['skipped'] + completed_count}/{total_tasks} "
                                  f"({progress_pct:.1f}%) - 成功: {completed_count}, 失败: {failed_count}")

                except Exception as e:
                    logger.error(f"任务异常 {task['date_key']}: {str(e)}")
                    failed_count += 1

        self.save_progress()
        logger.info(f"{year} 年下载完成: 成功 {completed_count}, 失败 {failed_count}")

    def run(self):
        """主运行函数"""
        logger.info(f"开始优化下载 ERA5 数据")
        logger.info(f"年份范围: {self.start_year} - {self.end_year}")
        logger.info(f"并行数: {self.workers}, 请求间隔: {self.delay}秒")

        start_time = time.time()

        for year in range(self.start_year, self.end_year + 1):
            self.download_year(year)

        elapsed_time = time.time() - start_time
        logger.info(f"下载完成! 总耗时: {elapsed_time/3600:.2f} 小时")

        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        logger.info("\n" + "="*60)
        logger.info("下载统计")
        logger.info("="*60)
        logger.info(f"跳过: {self.stats['skipped']}")
        logger.info(f"成功: {self.stats['completed']}")
        logger.info(f"失败: {self.stats['failed']}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='ERA5优化下载脚本')
    parser.add_argument('--start_year', type=int, required=True, help='起始年份')
    parser.add_argument('--end_year', type=int, required=True, help='结束年份')
    parser.add_argument('--workers', type=int, default=2, help='并行下载数 (默认: 2，CDS限制)')
    parser.add_argument('--delay', type=float, default=5, help='请求间隔秒数 (默认: 5)')

    args = parser.parse_args()

    downloader = ERA5OptimizedDownloader(args.start_year, args.end_year, args.workers, args.delay)
    downloader.run()


if __name__ == '__main__':
    main()
