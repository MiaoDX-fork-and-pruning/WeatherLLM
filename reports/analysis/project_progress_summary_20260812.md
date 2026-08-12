# 项目进展总结

**项目名称**: PhyDiff-Net 降水预报 AI 模型
**总结日期**: 2026-08-12
**Git 状态**: 已同步至 origin/main（commit `a318c14`）
**维护者**: weather-planner

---

## 一、项目目标与定位

基于 ECMWF 模式数据与团队自研的 **GMCP 高分辨率降水资料（0.1°、1 小时、2000–2024）**，借助物理约束扩散模型等前沿 AI 技术，显著提升降水预报精度，**目标发表 Nature 级别论文**。

核心破局点：现有气象大模型（Pangu、GraphCast、GenCast、Aurora）在降水预报上表现不佳，根因是降水时空异质性强、粗分辨率模式数据难以刻画。团队持有的 GMCP 0.1° 数据是关键差异化资产。

### 性能目标

| 指标 | 目标值 | SOTA 基准 | 提升幅度 |
|------|--------|----------|----------|
| 极端降水 CSI | >0.5 | 0.4–0.45 | 10–25% |
| 强降水 POD | >0.8 | 0.7–0.75 | 7–14% |
| 强降水 FAR | <0.3 | 0.35–0.4 | 15–25% |
| 降水 RMSE | 较 GenCast 降 10–15% | 基准线 | 10–15% |

---

## 二、整体进度概览

| 阶段 | 状态 | 完成度 |
|------|------|--------|
| 项目启动与规范建立 | ✅ 完成 | 100% |
| 文献研究（Pangu/GraphCast/GenCast/Aurora） | ✅ 完成 | 100% |
| 技术调研与 SOTA 分析 | ✅ 完成 | 100% |
| 算法设计（PhyDiff-Net 五模块） | ✅ 完成 | 100% |
| 模型代码实现 | ✅ 完成 | 100% |
| GMCP 真实数据管道打通 | ✅ 完成 | 100% |
| GMCP-only baseline 训练通路 | ✅ 完成 | 100% |
| 6h 标签数据批量生成 | 🔄 进行中（2000 完成，2001-2024 后台生成） | ~5% |
| GMCP-only 端到端训练验证 | ✅ 完成（冒烟通过） | 100% |
| ERA5 数据下载 | 🔄 进行中（瓶颈） | ~55% |
| 模型正式训练与调优 | ⏳ 待启动（待全量标签） | 0% |
| 评估对比 SOTA | ⏳ 待启动 | 0% |

---

## 三、本轮（2026-06 至 2026-08）完成的关键工作

### 3.1 GMCP 真实数据兼容性修复

项目原代码基于对 GMCP 格式的假设编写，与真实数据存在 5 处不兼容（文件路径结构、变量名 `rain_rate`、坐标名 `lat/lon`、无 `time` 维度、文件查找逻辑）。新增 `src/data/gmcp_reader.py`（`GMCPFileFinder` + `GMCPDataset`）彻底解决，11 项测试通过。

详见 [gmcp_compatibility_fix_20260630.md](gmcp_compatibility_fix_20260630.md)。

### 3.2 GMCP 数据探索性分析

对 300 个真实连续 6 小时窗口完成分析（[gmcp_analysis_20260630.md](gmcp_analysis_20260630.md)）：

- **数据质量好**：无缺失、无负值；少量 >200mm/6h 极端格点需裁剪。
- **分布高度偏斜**：6h 均值仅 0.58mm，绝大多数格点接近 0。
- **极端事件稀少**：≥25mm/6h 占 0.19%，≥100mm/6h 占 0.004% → 需 focal loss / 加权采样。
- **2025 年仅到 8 月**：训练截断至 2025-08-31。

### 3.3 PhyDiff-Net 新增 GMCP-only 训练模式

在 ERA5 下载受阻期间，为避免研发停滞，新增仅用 GMCP 降水自回归训练的 baseline 通路：

- **`use_era5` 开关**：`CrossResolutionFusion` / `MultiScaleEncoder` / `PhyDiffNet` 支持 ERA5+GMCP 双源与 GMCP 单源两种模式。
- **高分辨率适配**：新增 `encoder_spatial_size`，将 360×620 的 GMCP 下采样到 64×64 进入编码器，输出端再上采样还原，避免显存爆炸。
- **UNet skip connection 修复**：修正 decoder 首个 ResidualBlock 通道拼接错配，移除原先错误的 "remaining skip" 兜底分支。
- **`ConditionEncoder` 参数化**：`in_channels` 不再硬编码 19；`EnergyConstraint` 梯度 padding 对齐。

### 3.4 GMCP-only 训练管道

| 组件 | 文件 | 作用 |
|------|------|------|
| 数据集 | `src/data/gmcp_sequence_dataset.py` | 滑动窗口 6h 样本，支持预处理 NetCDF / 原始小时文件 |
| 预处理 | `scripts/preprocess_gmcp_6h.py` | 小时降水聚合为 6h 累计，按年存 NetCDF |
| 训练脚本 | `scripts/train_gmcp_only.py` | 完整训练/验证/checkpoint，含 `--verify_only` 冒烟测试 |
| 配置 | `configs/training_gmcp_only.yaml` | 完整训练配置 |
| 验证配置 | `configs/training_gmcp_only_verify.yaml` | 小规模快速验证 |
| 测试 | `tests/test_gmcp_sequence_dataset.py`、`tests/test_phydiff_net_gmcp_only.py` | 数据集与模型前向/反向测试 |

**测试现状**：`tests/test_gmcp_reader.py` + `tests/test_gmcp_sequence_dataset.py` + `tests/test_phydiff_net_gmcp_only.py` 共 **28 项测试全部通过**（53.78s）。

### 3.5 ERA5 下载脚本迭代

针对 CDS 下载慢与限流，新增三个脚本变体：
- `download_era5_parallel.py`（多线程，8 workers）
- `download_era5_optimized.py`（降并发 + 请求间隔，规避限流）
- `download_era5_improved.py`（压力层/单层 + 重试）

同时以 WeatherBench2 1.5° 数据作为快速原型通路。

### 3.6 预处理性能突破与端到端训练验证（2026-08-12）

**瓶颈诊断**：原 `preprocess_gmcp_6h.py` 单月耗时 787 秒，25 年需 ~65 小时。根因是 xarray 逐文件 `open_dataset` + `load()` 的元数据开销（~1 秒/文件）。

**优化方案**（`scripts/preprocess_gmcp_6h_fast.py`）：改用 netCDF4 直接读取中国区域切片，预计算固定索引，绕过 xarray 元数据开销。

| 指标 | 原方案 | 优化后 | 提升 |
|------|--------|--------|------|
| 单月耗时 | 787 秒 | 38 秒 | 20× |
| 全年耗时 | ~2.6 小时 | 7 分钟 | 22× |
| 单文件读取 | ~1 秒 | 0.015 秒 | 66× |

2000 年输出验证：1464 个 6h 窗口（闰年正确），mean 0.59mm/6h（与分析报告一致）。

**模型 5D 输入修复**：
- `_maybe_resize_gmcp`：修复 5D 输入 `[B, T_in, 1, H, W]` 下采样时 `F.interpolate` 维度错配
- `forward/sample`：GMCP-only 模式下 squeeze 单例 channel 维，使 T_in 作为 channel 轴进入 conv2d

**端到端训练冒烟验证通过**（`configs/training_gmcp_smoke.yaml`）：
- 2000 年 6-7 月数据，hidden_dim=64，1 epoch，GPU RTX 2060
- train_loss 0.0654 → 0.0445，val_loss 0.0400
- checkpoint 落盘 `outputs/gmcp_smoke/best_model.pt`
- **数据加载 → 前向 → 反向 → 验证 → 存盘全链路打通**

### 3.7 6h 标签全量生成（进行中）

2001-2024 年数据后台生成中（每月 ~30 秒，预计 2.5 小时完成）。完成后即可用 `training_gmcp_only.yaml` 启动正式小规模训练。

---

## 四、当前瓶颈与风险

### 4.1 ERA5 下载是最大瓶颈
- WeatherBench2 训练数据下载到 1999 年（卡在 2000 年），曾出现 `dask` 合并报错。
- 评测数据（2018–2019）已完成，约 12GB。
- **影响**：ERA5+GMCP 双源融合训练无法启动，目前只能跑 GMCP-only baseline。

### 4.2 GMCP-only 模式的局限
- 缺少大气环流输入（ERA5），模型只能做降水自回归，预报能力天花板受限，**无法直接对标 GenCast 等基于大气初始场的模型**。
- 仅适合作为消融对照与管道验证，非最终方案。

### 4.3 待人工确认事项
- [ ] GPU 资源确认（高分辨率训练显存需求大）
- [ ] ERA5 实际数据格式人工核验
- [ ] 基线模型（GenCast 等）下载与运行

---

## 五、下一步计划

> **主线判断**：数据管道与模型代码已就绪，当前最高优先级是**让训练真正跑起来产出第一个可用 checkpoint**，再谈调优与对比。ERA5 瓶颈短期内若无法突破，则先用 GMCP-only 跑通端到端流程。

### 5.1 P0 — 立即执行（本周）

1. **生成 6h 累计标签数据** ✅ 进行中
   - 2000 年已完成；2001-2024 后台生成中（`preprocess_gmcp_6h_fast.py`，预计 2.5h）。
   - 产出 `F:/GMCP_Precipitation_6h/gmcp_6h_YYYY.nc`。

2. **GMCP-only 端到端训练冒烟验证** ✅ 完成
   - `training_gmcp_smoke.yaml` 1 epoch 通过，loss 下降，checkpoint 落盘。

3. **GMCP-only 正式小规模训练** ⏳ 待全量标签完成
   - 全量 6h 标签就绪后，用 `training_gmcp_only.yaml` 跑 5 epoch，确认多 epoch 收敛趋势。

### 5.2 P1 — 短期推进（本月）

4. **极端事件损失函数落地**
   - 依据分析报告的类别不平衡结论，将当前 `GMCPLoss`（MSE+MAE）升级为含 focal loss / 极端事件加权 的多任务损失（复用 `src/models/losses/`）。
   - 按 25/50/100 mm/6h 阈值生成极端事件掩码。

5. **ERA5 下载突破**
   - 修复 `dask` 合并报错（改用显式 `xr.open_mfdataset` 或按年合并策略）。
   - 继续推进 2000–2017 训练集下载；若 CDS 持续受限，评估改用 WeatherBench2 1.5° 作为 ERA5 替代输入。

6. **评估管道联调**
   - 用 GMCP-only 第一个 checkpoint 跑通 `evaluate_benchmark.py`，验证 CSI/POD/FAR/RMSE 指标计算正确。

### 5.3 P2 — 中期（本季度）

7. **ERA5+GMCP 双源融合训练**（ERA5 就绪后）
8. **基线模型对比实验**（GenCast / GraphCast / Pangu 在拉齐的 2018/2019 评测集上）
9. **消融实验**（物理约束、极端分支、异质性模块各贡献）
10. **超参数调优与完整训练**

---

## 六、关键决策记录（补充）

| 决策 | 理由 | 日期 |
|------|------|------|
| 新增 GMCP-only 训练模式 | ERA5 下载受阻，避免研发停滞；先用单源跑通端到端管道 | 2026-06 |
| `encoder_spatial_size` 下采样策略 | 360×620 高分辨率直接进编码器显存爆炸；下采样后上采样还原 | 2026-06 |
| 修复 UNet skip connection | decoder 通道拼接错配会导致特征错位，影响扩散去噪质量 | 2026-06 |
| WeatherBench2 1.5° 作为快速原型 | 绕开 ERA5 原始下载，快速验证训练流程 | 2026-06 |

---

## 七、Git 提交记录（本轮）

| Commit | 类型 | 说明 |
|--------|------|------|
| `70b3842` | feat(data) | GMCP reader 与真实数据兼容性修复 |
| `cea52ab` | feat(model) | PhyDiff-Net GMCP-only 训练模式与管道 |
| `1a583a6` | feat(data) | ERA5 并行/优化下载脚本与进度记录 |
| `44e1994` | docs | 项目进展总结与日志更新 |
| `a318c14` | perf(data) | 极速预处理脚本（快 20 倍）+ 5D 输入修复 + 冒烟验证 |

所有提交已 push 至 `origin/main`。工作区干净，无未提交改动。

---

**附**：项目日志 `reports/analysis/project_log.md` 的"最后更新"仍停留在 2026-06-15，本轮 GMCP 工作尚未同步进日志，建议尽快补更。
