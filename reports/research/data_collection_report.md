# 数据收集与预处理报告

## 一、数据需求分析

### 1.1 项目核心数据需求

基于算法设计方案，本项目需要以下两类核心数据：

| 数据类型 | 分辨率 | 时间范围 | 变量 | 用途 |
|----------|--------|----------|------|------|
| **ERA5再分析数据** | 0.25° | 1979-present | 温度、风场、位势高度、湿度、气压、降水 | 大尺度环流背景场 |
| **GMCP降水数据** | 0.1° | 2000-2024 | 降水率 | 高分辨率降水训练标签 |

### 1.2 ERA5数据详细需求

#### 气压层数据 (Pressure Level)
- **变量**: 位势高度(geopotential)、温度(Temperature)、比湿(Specific humidity)、U/V风分量、垂直速度
- **气压层**: 1-1000 hPa共37层
- **时间分辨率**: 逐小时
- **空间范围**: 全球 (90°N-90°S, 180°W-180°E)
- **时间范围**: 1979-2025年

#### 单层数据 (Single Level)
- **变量**: 10m U/V风、2m温度、海平面气压、总降水量
- **时间分辨率**: 逐小时
- **空间范围**: 全球
- **时间范围**: 1979-2025年

#### 数据量估算
```
ERA5气压层数据:
- 每天24小时 × 37层 = 876个时次/年
- 每年约876个NetCDF文件
- 1979-2025年 = 47年
- 总计: 47 × 876 ≈ 41,172个文件

ERA5单层数据:
- 每天24小时 = 24个时次/年
- 每年约365个NetCDF文件
- 1979-2025年 = 47年
- 总计: 47 × 365 ≈ 17,155个文件

预估存储需求: 5-10 TB
```

### 1.3 GMCP数据详细需求

- **变量**: 降水率 (precipitation_rate)
- **分辨率**: 0.1° (约10km)
- **时间分辨率**: 逐小时
- **空间范围**: 中国区域
- **时间范围**: 2000-2024年

#### 数据量估算
```
GMCP数据:
- 每天24小时 × 365天 = 8,760个时次/年
- 2000-2024年 = 25年
- 总计: 25 × 8,760 = 219,000个文件

实际文件数: 225,000个NetCDF文件
预估存储需求: 3-5 TB
```

---

## 二、现有数据状态

### 2.1 ERA5数据状态

**目录**: `F:\ERA5再分析数据下载`

**状态**: ❌ **数据未下载**

现有内容仅包含下载脚本和说明文档:
- `Data_Download_ERA5_SingleLevel_Hourly.ipynb` - 单层数据下载脚本
- `DownloadERA5_PressureLevel_Hourly.ipynb` - 气压层数据下载脚本
- `下载代码使用说明.docx` - 使用说明文档

**关键问题**:
1. 下载脚本中的存储路径需要修改
2. 脚本默认只下载单年数据，需要修改为多年下载
3. 需要配置CDS API密钥

### 2.2 GMCP数据状态

**目录**: `F:\GMCP_Precipitation`

**状态**: ✅ **数据完整**

```
目录结构:
├── 2000/ (含12个月份子目录)
├── 2001/
├── ...
├── 2024/
├── 2025/
├── GMCP Datainfo.docx (数据说明)
├── How To Read GMCP.zip (读取示例)
└── Ma et al. - 2025 - GMCP Paper.pdf (论文)

文件格式:
- 命名: GMCP_YYYY_MM_DD_HH.nc
- 例: GMCP_2024_01_01_00.nc (2024年1月1日00时)
- 总文件数: 225,000个
```

### 2.3 数据状态总结

| 数据集 | 状态 | 文件数 | 预估大小 | 完成度 |
|--------|------|--------|----------|--------|
| ERA5气压层 | 未下载 | 0 | - | 0% |
| ERA5单层 | 未下载 | 0 | - | 0% |
| GMCP | ✅ 已完成 | 225,000 | 3-5 TB | 100% |

---

## 三、数据下载计划

### 3.1 ERA5数据下载方案

#### 前置条件
1. **CDS API配置**
   ```
   文件路径: C:\Users\<username>\.cdsapirc
   内容格式:
   url: https://cds.climate.copernicus.eu/api/v2
   key: <your_uid>:<your_api_key>
   ```

2. **申请CDS账号**
   - 访问 https://cds.climate.copernicus.eu
   - 注册账号并获取API Key

#### 下载策略

**策略1: 分年下载 (推荐)**
- 每次下载1年的数据
- 避免单次请求过大导致超时
- 支持断点续传

**策略2: 分变量下载**
- 先下载单层数据（存储较小）
- 再下载气压层数据（存储较大）

#### 下载脚本修改要点

```python
# 需要修改的参数
year = 2025  # 修改为目标年份
dirs = r'F:\ERA5_data\' + str(year)  # 修改为实际存储路径

# 添加的变量
variables_single_level = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind", 
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation"
]

variables_pressure_level = [
    "geopotential",
    "Temperature",
    "Specific humidity",
    "U-component of wind",
    "V-component of wind",
    "Vertical velocity"
]
```

#### 下载时间估算
```
单层数据:
- 每天24小时 = 24个时次
- 每年约365天 = 8,760个时次
- 每个时次约5-10秒请求时间
- 预计: 12-24小时/年

气压层数据:
- 每天24小时 × 37层 = 876个时次
- 每年约365天 = 319,560个时次
- 每个时次约10-20秒请求时间
- 预计: 3-7天/年

总计: 47年数据预计需要6-12个月
```

#### 优先级建议
1. **第一优先**: 2000-2024年（与GMCP重叠期）
2. **第二优先**: 1979-1999年（预训练扩展）
3. **第三优先**: 2025年（最新数据）

### 3.2 下载执行计划

| 阶段 | 时间范围 | 数据类型 | 预计耗时 | 存储需求 |
|------|----------|----------|----------|----------|
| 阶段1 | 2000-2024 | ERA5单层 | 1-2周 | 500GB |
| 阶段2 | 2000-2024 | ERA5气压层 | 1-2月 | 3-4TB |
| 阶段3 | 1979-1999 | ERA5单层 | 1周 | 200GB |
| 阶段4 | 1979-1999 | ERA5气压层 | 2-3周 | 1-2TB |
| 阶段5 | 2025 | 全部 | 1周 | 200GB |

---

## 四、数据预处理流程

### 4.1 预处理架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    数据预处理Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   数据加载   │───▶│   质量控制   │───▶│   空间配准   │     │
│  │   Module    │    │   Module    │    │   Module    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   时间对齐   │───▶│   特征工程   │───▶│   归一化    │     │
│  │   Module    │    │   Module    │    │   Module    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │   数据存储      │                       │
│                   │ (Zarr/NetCDF)   │                       │
│                   └─────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 模块详细设计

#### Module 1: 数据加载 (Data Loader)

```python
class DataLoaderModule:
    """统一加载ERA5和GMCP数据"""
    
    def load_era5_single_level(self, year, month, day):
        """加载ERA5单层数据"""
        # 文件名格式: YYYY-MM-DD.nc
        # 返回: xarray.Dataset
        
    def load_era5_pressure_level(self, year, month, day):
        """加载ERA5气压层数据"""
        # 文件名格式: ERA5_0.25_PL_YYYY-MM-DD-HH.nc
        # 返回: xarray.Dataset
        
    def load_gmcp(self, year, month, day, hour):
        """加载GMCP数据"""
        # 文件名格式: GMCP_YYYY_MM_DD_HH.nc
        # 返回: xarray.Dataset
```

#### Module 2: 质量控制 (Quality Control)

```python
class QualityControlModule:
    """数据质量控制"""
    
    def check_missing_values(self, data):
        """缺失值检测与处理"""
        # 策略: 时空插值 + 物理约束修复
        
    def detect_outliers(self, data):
        """异常值检测"""
        # 策略: 基于物理阈值和统计分布双重检验
        # 阈值:
        #   - 温度: -90°C ~ 60°C
        #   - 风速: 0 ~ 100 m/s
        #   - 降水量: 0 ~ 200 mm/h
        
    def validate_physical_consistency(self, data):
        """物理一致性验证"""
        # 检查变量间的物理关系
```

#### Module 3: 空间配准 (Spatial Alignment)

```python
class SpatialAlignmentModule:
    """空间分辨率统一到0.1°"""
    
    def regrid_era5_to_01(self, era5_data):
        """ERA5从0.25°重采样到0.1°"""
        # 方法: 双线性插值
        # 输出: 统一到中国区域0.1°网格
        
    def align_to_common_grid(self, era5_data, gmcp_data):
        """对齐到共同网格"""
        # 确保ERA5和GMCP使用相同的空间范围和网格
```

#### Module 4: 时间对齐 (Temporal Alignment)

```python
class TemporalAlignmentModule:
    """时间分辨率统一"""
    
    def align_to_6h(self, hourly_data):
        """从逐小时聚合到6小时"""
        # 方法: 累计/平均
        # 降水: 累计
        # 温度/风场: 平均
        
    def create_time_windows(self, data, window_size=6):
        """创建时间窗口"""
        # 输入: 12个时间步 (72小时历史)
        # 输出: 4个预测目标 (24小时预测)
```

#### Module 5: 特征工程 (Feature Engineering)

```python
class FeatureEngineeringModule:
    """特征构建"""
    
    def compute_derived_features(self, data):
        """计算衍生特征"""
        # - 相对湿度 (从比湿和温度计算)
        # - 风速 (从U/V分量计算)
        # - 降水强度等级
        
    def add_static_features(self, data):
        """添加静态特征"""
        # - 地形高度 (DEM数据)
        # - 海陆掩码
        # - 纬度/经度编码
        
    def add_temporal_features(self, data):
        """添加时间特征"""
        # - 小时编码 (sin/cos变换)
        # - 日编码
        # - 季节编码
```

#### Module 6: 归一化 (Normalization)

```python
class NormalizationModule:
    """数据标准化"""
    
    def normalize_era5(self, data, stats):
        """ERA5变量Z-score标准化"""
        # 基于1979-2019气候态计算均值和标准差
        # 公式: (x - mean) / std
        
    def normalize_gmcp(self, data):
        """GMCP降水归一化"""
        # 对数变换 + Min-Max归一化
        # 公式: log(x + 1) -> min-max scaling
        
    def normalize_topography(self, data, max_elev):
        """地形归一化"""
        # 公式: elevation / max_elevation
```

### 4.3 预处理脚本框架

```python
# scripts/preprocess_data.py

import xarray as xr
import numpy as np
from pathlib import Path

class WeatherDataPreprocessor:
    """气象数据预处理主类"""
    
    def __init__(self, config):
        self.config = config
        self.era5_path = Path(config['era5_path'])
        self.gmcp_path = Path(config['gmcp_path'])
        self.output_path = Path(config['output_path'])
        
    def process_year(self, year):
        """处理单年数据"""
        for month in range(1, 13):
            for day in range(1, 32):
                self.process_day(year, month, day)
                
    def process_day(self, year, month, day):
        """处理单日数据"""
        # 1. 加载数据
        era5_sl = self.load_era5_single_level(year, month, day)
        era5_pl = self.load_era5_pressure_level(year, month, day)
        gmcp = self.load_gmcp_day(year, month, day)
        
        # 2. 质量控制
        era5_sl = self.quality_control(era5_sl)
        era5_pl = self.quality_control(era5_pl)
        gmcp = self.quality_control(gmcp)
        
        # 3. 空间配准
        era5_aligned = self.regrid_to_01(era5_sl, era5_pl)
        
        # 4. 时间对齐
        era5_6h = self.align_temporal(era5_aligned, freq='6h')
        gmcp_6h = self.align_temporal(gmcp, freq='6h')
        
        # 5. 特征工程
        features = self.build_features(era5_6h, gmcp_6h)
        
        # 6. 归一化
        features_norm = self.normalize(features)
        
        # 7. 保存
        self.save_processed(features_norm, year, month, day)
        
    def build_dataset(self, start_year, end_year):
        """构建完整数据集"""
        for year in range(start_year, end_year + 1):
            self.process_year(year)
            
        # 生成数据索引
        self.create_index_file()

if __name__ == '__main__':
    config = {
        'era5_path': 'F:/ERA5_data',
        'gmcp_path': 'F:/GMCP_Precipitation',
        'output_path': 'F:/processed_data',
        'start_year': 2000,
        'end_year': 2024
    }
    
    preprocessor = WeatherDataPreprocessor(config)
    preprocessor.build_dataset(2000, 2024)
```

### 4.4 预处理输出格式

```
输出目录结构:
F:/processed_data/
├── train/
│   ├── 2000/
│   │   ├── 2000_01_01_00.zarr
│   │   ├── 2000_01_01_06.zarr
│   │   └── ...
│   └── ...
├── val/
│   └── 2022/
├── test/
│   └── 2023-2024/
├── stats/
│   ├── era5_normalization_stats.nc
│   └── gmcp_normalization_stats.nc
└── metadata.json
```

### 4.5 存储优化

#### 推荐存储格式: Zarr
- **优势**: 高效的压缩和分块读取
- **压缩**: LZ4或ZSTD
- **分块**: 时间维1、空间维(64, 64)

```python
# Zarr存储配置
zarr_config = {
    'compressor': 'zstd',
    'compression_level': 5,
    'chunks': {
        'time': 1,
        'latitude': 64,
        'longitude': 64
    }
}
```

---

## 五、问题和解决方案

### 5.1 数据获取问题

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| ERA5数据未下载 | 无法进行训练 | 优先下载2000-2024年数据 |
| CDS API限制 | 下载速度慢 | 使用多线程，控制并发数(5-10) |
| 存储空间不足 | 无法存储所有数据 | 使用外部硬盘或云存储 |
| 网络不稳定 | 下载中断 | 实现断点续传机制 |

### 5.2 数据质量问题

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| 缺失值 | 模型训练不稳定 | 时空插值 + 物理约束修复 |
| 异常值 | 模型过拟合 | 物理阈值检验 + 统计检验 |
| 时间不一致 | 特征对齐困难 | 统一到6小时时间步 |
| 空间分辨率差异 | 特征融合困难 | ERA5重采样到0.1° |

### 5.3 存储和计算问题

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| 数据量大(8-15TB) | 存储成本高 | 使用Zarr格式压缩 |
| IO瓶颈 | 训练速度慢 | 使用HDF5/Zarr分块读取 |
| 内存不足 | 无法加载全量数据 | 流式加载 + 分块处理 |

### 5.4 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| GPU内存不足 | 中 | 高 | 减小batch size，使用混合精度 |
| 数据加载慢 | 高 | 中 | 预加载 + 多进程 + 缓存 |
| 数值溢出 | 低 | 高 | 数据类型检查 + 数值裁剪 |

---

## 六、执行计划时间线

### 6.1 数据准备阶段 (4周)

```
第1周:
├── 配置CDS API
├── 修改下载脚本
└── 测试下载1年数据

第2-3周:
├── 批量下载ERA5数据 (2000-2024)
├── 并行下载多个年份
└── 监控下载进度

第4周:
├── 数据完整性验证
├── 开始预处理脚本开发
└── 搭建预处理Pipeline
```

### 6.2 预处理阶段 (2周)

```
第5周:
├── 实现质量控制模块
├── 实现空间配准模块
└── 测试单日数据

第6周:
├── 实现时间对齐模块
├── 实现特征工程模块
└── 批量处理2000-2024年数据
```

### 6.3 验证阶段 (1周)

```
第7周:
├── 数据质量检查
├── 生成数据统计报告
└── 准备训练数据
```

---

## 七、资源需求

### 7.1 存储需求

| 数据类型 | 预估大小 | 存储位置 |
|----------|----------|----------|
| ERA5原始数据 | 5-10 TB | 外部硬盘 |
| GMCP原始数据 | 3-5 TB | F:\GMCP_Precipitation |
| 预处理后数据 | 2-3 TB | F:\processed_data |
| **总计** | **10-18 TB** | - |

### 7.2 计算需求

| 任务 | CPU核心 | 内存 | 预计耗时 |
|------|---------|------|----------|
| 数据下载 | 5-10线程 | 8GB | 6-12个月 |
| 数据预处理 | 8-16核 | 32-64GB | 2-4周 |
| 数据验证 | 4-8核 | 16GB | 1周 |

### 7.3 软件依赖

```python
# requirements_data.txt
xarray>=2023.1.0
netcdf4>=1.6.0
h5py>=3.8.0
zarr>=2.14.0
cdsapi>=0.7.0
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
dask[complete]>=2023.1.0
```

---

## 八、总结

### 8.1 当前状态

- **GMCP数据**: ✅ 完整可用 (225,000个文件，3-5TB)
- **ERA5数据**: ❌ 需要下载 (预计41,000+文件，5-10TB)
- **预处理脚本**: ⏳ 待开发

### 8.2 下一步行动

1. **立即执行**:
   - 配置CDS API密钥
   - 修改ERA5下载脚本
   - 开始下载2000-2024年ERA5数据

2. **并行执行**:
   - 开发数据预处理Pipeline
   - 设计数据加载器
   - 搭建训练环境

3. **后续执行**:
   - 下载1979-1999年ERA5数据
   - 完成所有数据预处理
   - 开始模型训练

### 8.3 成功标准

- [ ] ERA5数据下载完成 (2000-2024年)
- [ ] 预处理脚本开发完成
- [ ] 训练数据集构建完成
- [ ] 数据质量验证通过
- [ ] 数据加载器可高效运行

---

**报告版本**: v1.0  
**创建日期**: 2026-06-15  
**作者**: data-collector  
**状态**: 数据收集计划完成，待执行
