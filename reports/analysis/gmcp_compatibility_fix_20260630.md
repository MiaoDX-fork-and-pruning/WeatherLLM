# GMCP 数据兼容性修复报告

**日期**: 2026-07-01  
**涉及文件**:

- 新增: `src/data/gmcp_reader.py`
- 新增: `tests/test_gmcp_reader.py`
- 新增: `scripts/analyze_gmcp.py`
- 修改: `src/data/preprocessing.py`

---

## 1. 问题概述

项目原有代码基于对 GMCP 数据格式的假设编写，但实际下载到的 GMCP 文件在路径结构、变量名、坐标名和时间维度等方面均与假设不符。这导致 `GMCPPreprocessor.load_data()` 无法找到文件，整个预处理管道无法运行。

## 2. 不兼容点清单

| 序号 | 不兼容项 | 项目代码假设 | 真实 GMCP 数据 | 修复方式 |
|------|----------|--------------|----------------|----------|
| 1 | 文件路径结构 | `gmcp_YYYY_MM.nc` 月度合并文件，位于数据根目录 | `YYYY/MM/GMCP_YYYY_MM_DD_HH.nc` 小时文件；2025 年为扁平结构 | 新增 `GMCPFileFinder`，支持两种目录结构 |
| 2 | 降水变量名 | `precipitation_rate` | `rain_rate` | 读取时重命名为 `precipitation_rate` |
| 3 | 坐标名 | `latitude` / `longitude` | `lat` / `lon` | 读取时重命名为 `latitude` / `longitude` |
| 4 | 时间维度 | 文件内含 `time` 坐标 | 无 `time` 维度，需从文件名解析 | `GMCPDataset` 根据文件名构造 `time` 坐标 |
| 5 | 文件查找逻辑 | 按月遍历查找 `gmcp_YYYY_MM.nc` | 按小时文件遍历 | `GMCPPreprocessor.load_data()` 改为使用 `GMCPDataset` |

## 3. 修复详情

### 3.1 新增 `src/data/gmcp_reader.py`

提供三个核心组件：

- `GMCPFileFinder`: 扫描磁盘，解析 `GMCP_YYYY_MM_DD_HH.nc` 文件名，支持 2000–2024 的 `YYYY/MM/` 结构和 2025 的扁平结构。
- `GMCPDataset`: 惰性加载文件，构造 `time` 坐标，统一变量名和坐标名，并支持中国区域裁剪。
- 便捷函数 `load_gmcp_for_period` 和 `count_files_by_year_month`。

### 3.2 修改 `src/data/preprocessing.py`

- 导入 `GMCPDataset`。
- 重写 `GMCPPreprocessor.load_data()`：
  - 不再查找月度合并文件；
  - 使用 `GMCPDataset` 加载真实小时文件；
  - 保留原有返回 `xr.Dataset` 的接口，下游代码无需改动。
- `GMCPPreprocessor.quality_control()` 和 `temporal_alignment()` 无需修改，因为 `GMCPDataset` 已经输出标准化的 `precipitation_rate`、`latitude`、`longitude`、`time`。

### 3.3 新增 `tests/test_gmcp_reader.py`

覆盖以下场景：

- 标准目录结构（`YYYY/MM/`）文件查找
- 2025 扁平结构文件查找
- 时间范围过滤
- 变量/坐标重命名
- 中国区域裁剪
- 缺失变量异常
- 按年月计数

## 4. 验证结果

```bash
python -m pytest tests/test_gmcp_reader.py -v
```

结果：11 项测试全部通过。

```
============================= 11 passed in 7.36s ==============================
```

## 5. 后续注意事项

1. **数据仍存储在 `F:\`**：读取模块仅引用 `F:/GMCP_Precipitation`，未复制数据。
2. **6 小时累计处理**：`GMCPPreprocessor.temporal_alignment()` 中的 `resample(time="6h").sum()` 在整年数据上内存消耗大，后续处理大规模数据时建议按块（chunk）处理。
3. **2025 年数据不完整**：读取模块已自动支持扁平结构，但仅覆盖到 2025-08。
4. **静态数据缺失**：地形、土地利用等静态数据尚未准备，需后续补充。

## 6. 影响范围

- `src/data/preprocessing.py` 中的 `GMCPPreprocessor` 现在可以正确读取真实 GMCP 数据。
- `scripts/preprocess_data.py` 和 `scripts/evaluate_benchmark.py` 等下游脚本无需修改即可受益（只要它们调用 `GMCPPreprocessor`）。
- 模型训练流程中的数据加载障碍已清除。
