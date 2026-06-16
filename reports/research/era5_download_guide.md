# ERA5数据下载指南

## 1. 概述

本指南用于指导如何从Copernicus Climate Data Store (CDS) 下载ERA5再分析数据。ERA5是ECMWF提供的全球大气再分析数据集，提供从1979年至今的全球大气、陆地和海洋气候变量数据。

### 1.1 数据说明

| 数据类型 | 变量 | 时间分辨率 | 空间分辨率 | 时间范围 |
|---------|------|-----------|-----------|---------|
| 压力层数据 | 位势、温度、比湿、U/V风速 | 每小时 | 0.25° | 2000-2024 |
| 单层数据 | 10m风速、2m温度、海平面气压、总降水 | 每小时 | 0.25° | 2000-2024 |

### 1.2 压力层数据变量

- `geopotential` - 位势 (m²/s²)
- `temperature` - 温度 (K)
- `specific_humidity` - 比湿 (kg/kg)
- `u_component_of_wind` - U分量风速 (m/s)
- `v_component_of_wind` - V分量风速 (m/s)

### 1.3 压力层列表

共37层：1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000 (hPa)

### 1.4 单层数据变量

- `10m_u_component_of_wind` - 10米U分量风速 (m/s)
- `10m_v_component_of_wind` - 10米V分量风速 (m/s)
- `2m_temperature` - 2米温度 (K)
- `mean_sea_level_pressure` - 平均海平面气压 (Pa)
- `total_precipitation` - 总降水量 (m)

---

## 2. CDS API配置

### 2.1 注册账号

1. 访问 https://cds.climate.copernicus.eu
2. 点击右上角 "Register" 注册新账号
3. 填写注册信息并验证邮箱
4. 登录后，点击用户名 -> "API key"

### 2.2 获取API Key

登录CDS后，在用户主页可以看到：
- **UID**: 用户唯一标识 (例如: 88032)
- **API Key**: API密钥 (例如: a94ec4ba-2991-4342-a0db-1ef8dc42d78d)

### 2.3 创建配置文件

在用户主目录下创建 `.cdsapi` 配置文件：

**Windows系统:**
```
C:\Users\<你的用户名>\.cdsapi
```

**Linux/Mac系统:**
```
~/.cdsapi
```

**文件内容:**
```yaml
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

**示例:**
```yaml
url: https://cds.climate.copernicus.eu/api/v2
key: 88032:a94ec4ba-2991-4342-a0db-1ef8dc42d78d
```

> **注意**: 请将上面的UID和API Key替换为你自己的密钥！

### 2.4 验证配置

运行以下命令验证配置是否正确：

```python
import cdsapi
c = cdsapi.Client()
print("CDS API配置成功!")
```

---

## 3. 安装依赖

### 3.1 安装cdsapi

```bash
pip install cdsapi
```

### 3.2 安装其他依赖 (可选)

```bash
pip install xarray netCDF4 dask
```

---

## 4. 使用下载脚本

### 4.1 脚本位置

```
E:\weather\scripts\download_era5.py
```

### 4.2 命令行参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--all` | 下载所有数据类型 | - |
| `--pressure-level` | 仅下载压力层数据 | - |
| `--single-level` | 仅下载单层数据 | - |
| `--start-year` | 起始年份 | 2000 |
| `--end-year` | 结束年份 | 2024 |
| `--threads` | 下载线程数 | 5 |
| `--output-dir` | 输出目录 | E:\ERA5 |
| `--progress` | 查看下载进度 | - |
| `--log-file` | 日志文件路径 | - |

### 4.3 使用示例

#### 下载所有数据

```bash
python download_era5.py --all
```

#### 仅下载压力层数据

```bash
python download_era5.py --pressure-level
```

#### 仅下载单层数据

```bash
python download_era5.py --single-level
```

#### 下载指定年份范围

```bash
python download_era5.py --all --start-year 2010 --end-year 2020
```

#### 自定义输出目录

```bash
python download_era5.py --all --output-dir D:\ERA5_Data
```

#### 使用更多线程加速下载

```bash
python download_era5.py --all --threads 8
```

#### 查看下载进度

```bash
python download_era5.py --progress
```

#### 保存日志

```bash
python download_era5.py --all --log-file download.log
```

---

## 5. 数据存储结构

下载完成后，数据将按以下结构存储：

```
E:\ERA5\
├── pressure_level\
│   ├── 2000\
│   │   ├── ERA5_0.25_PL_2000-01-01.nc
│   │   ├── ERA5_0.25_PL_2000-01-02.nc
│   │   └── ...
│   ├── 2001\
│   │   └── ...
│   └── 2024\
│       └── ...
└── single_level\
    ├── 2000\
    │   ├── ERA5_0.25_SL_2000-01-01.nc
    │   ├── ERA5_0.25_SL_2000-01-02.nc
    │   └── ...
    ├── 2001\
    │   └── ...
    └── 2024\
        └── ...
```

---

## 6. 下载注意事项

### 6.1 API限制

- 每个账号有并发请求限制（通常为10个）
- 线程数建议设置为5-10
- 过多线程可能导致请求被拒绝

### 6.2 服务器维护

- CDS服务器通常在欧洲时间凌晨维护
- 建议在下午3点（北京时间）后开始下载
- 如果请求无响应，可能是服务器维护中

### 6.3 断点续传

脚本支持断点续传功能：
- 已下载的文件会自动跳过
- 如果下载中断，重新运行脚本即可继续
- 建议删除下载不完整的文件（文件大小异常）

### 6.4 存储空间

- 压力层数据: 约50-100 MB/天
- 单层数据: 约10-20 MB/天
- 25年数据总计: 约500GB-1TB

### 6.5 下载时间

- 单线程: 约2-3天/年
- 5线程: 约1天/年
- 10线程: 约12小时/年

---

## 7. 常见问题

### 7.1 安装cdsapi失败

```bash
# 使用镜像源安装
pip install cdsapi -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 7.2 API Key无效

1. 确认已登录CDS账号
2. 确认UID和API Key正确（注意大小写和冒号）
3. 确认配置文件路径正确

### 7.3 下载速度慢

1. 增加线程数（但不要超过10）
2. 检查网络连接
3. 选择服务器空闲时段下载

### 7.4 文件下载不完整

1. 删除不完整的文件
2. 重新运行脚本
3. 如果频繁失败，减少线程数或增加重试间隔

### 7.5 内存不足

1. 减少线程数
2. 分批下载（按年份分批）
3. 使用64位Python

---

## 8. Python API使用示例

### 8.1 基本下载示例

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'geopotential',
            'temperature',
            'specific_humidity',
            'u_component_of_wind',
            'v_component_of_wind',
        ],
        'pressure_level': [
            '1', '2', '3', '5', '7', '10',
            '20', '30', '50', '70', '100', '125',
            '150', '175', '200', '225', '250', '300',
            '350', '400', '450', '500', '550', '600',
            '650', '700', '750', '775', '800', '825',
            '850', '875', '900', '925', '950', '975',
            '1000',
        ],
        'year': '2020',
        'month': '01',
        'day': '01',
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    },
    'era5_pressure_level_20200101.nc')
```

### 8.2 批量下载示例

```python
import cdsapi
from pathlib import Path

c = cdsapi.Client()

# 设置输出目录
output_dir = Path('E:/ERA5/pressure_level/2020')
output_dir.mkdir(parents=True, exist_ok=True)

# 下载2020年1月数据
for day in range(1, 32):
    date_str = f'2020-01-{day:02d}'
    filename = f'ERA5_PL_{date_str}.nc'
    filepath = output_dir / filename

    if filepath.exists():
        print(f'跳过已存在文件: {filename}')
        continue

    print(f'下载: {date_str}')
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['geopotential', 'temperature'],
            'pressure_level': ['500', '850', '1000'],
            'year': '2020',
            'month': '01',
            'day': f'{day:02d}',
            'time': ['00:00', '12:00'],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        str(filepath))
```

---

## 9. 数据验证

### 9.1 检查文件完整性

```python
import xarray as xr

# 打开NetCDF文件
ds = xr.open_dataset('ERA5_0.25_PL_2020-01-01.nc')

# 查看变量
print(ds.data_vars)

# 查看维度
print(ds.dims)

# 关闭数据集
ds.close()
```

### 9.2 检查文件大小

```python
import os

# 检查文件大小是否合理
filepath = 'ERA5_0.25_PL_2020-01-01.nc'
size_mb = os.path.getsize(filepath) / (1024 * 1024)
print(f'文件大小: {size_mb:.2f} MB')

# 压力层数据通常50-100MB
if size_mb < 10:
    print('警告: 文件可能不完整')
```

---

## 10. 联系支持

如果遇到问题，可以：

1. 查看CDS官方文档: https://cds.climate.copernicus.eu
2. 提交帮助请求: https://cds.climate.copernicus.eu/support
3. 查看CDS论坛: https://cds.climate.copernicus.eu/forum

---

## 附录: 变量对照表

| 英文变量名 | 中文名称 | 单位 |
|-----------|---------|------|
| geopotential | 位势 | m²/s² |
| temperature | 温度 | K |
| specific_humidity | 比湿 | kg/kg |
| u_component_of_wind | U分量风速 | m/s |
| v_component_of_wind | V分量风速 | m/s |
| 10m_u_component_of_wind | 10米U分量风速 | m/s |
| 10m_v_component_of_wind | 10米V分量风速 | m/s |
| 2m_temperature | 2米温度 | K |
| mean_sea_level_pressure | 平均海平面气压 | Pa |
| total_precipitation | 总降水量 | m |
