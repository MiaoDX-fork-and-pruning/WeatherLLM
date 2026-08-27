# 气象降水预报AI模型研究项目

## 项目目标

基于ECMWF模式数据，利用GMCP（0.1°，1-hour，2000-2024）和前沿AI技术，显著提高降水预报精度，目标是发表Nature级别论文。

## 背景

### 现有问题
- 气象大模型（Pangu、GraphCast、GenCast）在降水预报上效果不佳
- 降水时空异质性强，难以用粗分辨率模式数据准确预报
- 现有模型严重依赖ECMWF 0.25°数据，无法有效改进降水预报

### 我们的优势
- 拥有GMCP数据（0.1°，1-hour，2000-2024）：团队自研的全球多源融合降水资料
- 可在ECMWF基础上融合高分辨率降水数据
- AI技术发展提供了新的解决方案

## Multi-Agent团队

我们采用multi-agent协作的方式进行研究，不同agent负责不同工作：

### Agent角色

| Agent | 角色 | 职责 |
|-------|------|------|
| weather-planner | 任务规划专家 | 拆分任务，制定计划，建立Git标准 |
| paper-reader | 论文理解专家 | 阅读论文，分析核心点 |
| paper-reviewer | 论文审核专家 | 审核论文理解的质量 |
| weather-model-trainer | 模型训练专家 | 设计代码，进行模型训练 |
| weather-model-reviewer | 模型审核专家 | 验证模型是否满足要求 |
| algorithm-architect | 算法架构专家 | 设计算法方案 |
| algorithm-reviewer | 算法审核专家 | 审核算法方案可行性 |
| weather-researcher | 调研专家 | 完成技术调研 |
| weather-research-reviewer | 调研审核专家 | 检验调研产出质量 |

### 协作流程

```
阶段一：项目启动
    weather-planner → 制定项目计划

阶段二：文献研究
    paper-reader → 论文分析
    paper-reviewer → 审核论文理解

阶段三：技术调研
    weather-researcher → 技术调研
    weather-research-reviewer → 审核调研质量

阶段四：算法设计
    algorithm-architect → 设计算法方案
    algorithm-reviewer → 审核方案可行性

阶段五：模型实现
    weather-model-trainer → 实现模型
    weather-model-reviewer → 审核代码和实验

阶段六：迭代优化
    根据反馈持续改进
```

## 数据资源

### 数据位置
- **ERA5再分析数据（CDS原始）**: `F:\ERA5再分析数据下载`
- **WeatherBench2 ERA5**: `F:\WeatherBench2_ERA5`（1.5°、6h、13 变量、5 压力层）
- **GMCP降水数据（原始小时）**: `F:\GMCP_Precipitation`
- **GMCP 6h累计标签（预处理）**: `F:\GMCP_Precipitation_6h`（25 年 36528 窗口，31GB）

### 数据特点
| 数据集 | 分辨率 | 时间范围 | 描述 |
|--------|--------|----------|------|
| ERA5 (CDS) | 0.25° | 1979-present | ECMWF再分析数据（下载慢，基本弃用） |
| ERA5 (WeatherBench2) | 1.5° | 2018-2019（已就绪）/ 2000-2017（下载中） | GCS 分发，6h、13 变量 |
| GMCP | 0.1° | 2000-2024 | 全球多源融合降水资料 |
| GMCP 6h 标签 | 0.1° | 2000-2024 | 中国区域（360×620），6h 累计降水 |

## 论文资源

`docs/`目录下的核心论文：
- Pangu-Weather (Nature, 2023)
- GraphCast (Science, 2023)
- GenCast (Nature, 2024)
- Aurora (Nature, 2025)

## Git规范

### 提交格式
```
<type>: <description>

类型: feat, fix, refactor, docs, test, chore, perf, ci, experiment
```

### 示例
```
feat: implement ERA5 data preprocessing
fix: resolve GMCP data loading issue
experiment: run precipitation forecast comparison
```

## 使用方法

### 启动项目
告诉weather-planner你的研究目标，它会：
1. 分析目标，拆解任务
2. 分配任务给各子agent
3. 跟踪进度，协调资源

### 单独使用某个agent
```
# 论文理解
使用paper-reader分析指定论文

# 技术调研
使用weather-researcher进行调研

# 模型训练
使用weather-model-trainer进行训练
```

### 使用workflow
```
使用weather-forecast-research workflow执行完整的研究流程
```

## 项目结构

```
weather/
├── README.md              # 项目说明
├── PROJECT_RULES.md       # 项目规范（Git、目录、命名）
├── docs/                  # 论文资料
│   ├── Bi-2023-Pangu-Nature.pdf
│   ├── Lam-2023-GraphCast-Science.pdf
│   ├── Price-2024-GenCast-Nature.pdf
│   └── Bodnar-2025-Aurora-Nature.pdf
├── data/                  # 数据相关
│   ├── raw/               # 原始数据
│   ├── processed/         # 预处理后数据
│   └── scripts/           # 数据处理脚本
├── src/                   # 源代码
│   ├── data/              # 数据加载和预处理
│   ├── models/            # 模型实现
│   ├── training/          # 训练流程
│   ├── evaluation/        # 评估脚本
│   ├── utils/             # 工具函数
│   └── configs/           # 配置文件
├── models/                # 训练好的模型权重
│   ├── checkpoints/       # 模型检查点
│   └── final/             # 最终模型
├── outputs/               # 代码运行输出
│   ├── logs/              # 训练日志
│   ├── predictions/       # 预测结果
│   └── figures/           # 可视化图表
├── reports/               # 报告类产出
│   ├── analysis/          # 分析报告
│   ├── research/          # 调研报告
│   ├── reviews/           # 审核报告
│   └── experiments/       # 实验报告
└── experiments/           # 实验配置和记录
    ├── configs/           # 实验配置
    └── results/           # 实验结果
```

## 核心算法：PhyDiff-Net

### 设计目标
| 指标 | 目标值 | SOTA基准 | 提升幅度 |
|------|--------|----------|----------|
| 极端降水CSI | >0.5 | 0.4-0.45 | 10-25% |
| 强降水POD | >0.8 | 0.7-0.75 | 7-14% |
| 强降水FAR | <0.3 | 0.35-0.4 | 15-25% |
| 降水RMSE | 比GenCast降低10-15% | 基准线 | 10-15% |

### 五大创新点
1. **物理约束扩散模型**: 将大气动力学方程融入扩散过程
2. **0.1°高分辨率直接训练**: 突破传统降尺度限制
3. **极端事件感知机制**: 专门的分支和损失函数
4. **多尺度时空异质性建模**: 自适应捕捉不同尺度特征
5. **ERA5-GMCP联合预训练**: 充分利用多源数据优势

### 模型架构
```
PhyDiff-Net
├── Multi-Scale Spatiotemporal Encoder (Module A)
├── Physics-Constrained Diffusion Module (Module B)
├── Extreme Event Aware Branch (Module C)
├── Spatiotemporal Heterogeneity Module (Module D)
└── Multi-task Output Head
```

## 项目进度

### 已完成

**阶段一：研究基础（2026-06）**
- [x] SOTA性能指标分析（Pangu/GraphCast/GenCast/Aurora 四篇顶刊）
- [x] 技术调研报告
- [x] 算法设计方案（PhyDiff-Net 五模块架构）
- [x] 模型代码实现（467M 参数，前向/反向验证通过）
- [x] 评测集拉齐分析（GenCast 2019 / GraphCast·Pangu 2018）

**阶段二：数据管道（2026-06 ~ 2026-08）**
- [x] GMCP 真实数据兼容性修复（5 处不兼容：路径/变量名/坐标/时间维度/查找逻辑）
- [x] GMCP 探索性分析（无缺失无负值；≥25mm/6h 仅 0.19%，确认类别不平衡）
- [x] 6h 标签全量生成：25 年 × 36528 窗口（极速预处理，单月 787s→38s，快 20 倍）
- [x] ERA5 双源数据通路：WeatherBench2 2018-2019 就绪，GCS 下载通道打通

**阶段三：GMCP-only Baseline（2026-08）**
- [x] GMCP-only 训练模式（`use_era5` 开关 + 高分辨率下采样）
- [x] GMCPExtremeLoss 极端事件损失（归一化 MSE + 物理空间 CSI + 极端加权，9 项测试）
- [x] Baseline 训练 3 epoch（val_loss 108→102→98 稳定收敛）
- [x] 评估闭环打通，基线指标：轻中雨 CSI 0.12-0.23、RMSE 0.805mm/6h、相关 0.483
- [x] **基线结论：极端事件失败（CSI≈0），根因是缺少大气环流输入**

**阶段四：ERA5+GMCP 双源融合（进行中，2026-08）**
- [x] GMCPERA5Dataset 双源数据集（17 通道 ERA5：4 变量×3 层 + 5 地面；时间对齐验证）
- [x] 双源训练脚本 train_gmcp_era5.py + 双源评估支持
- [ ] 双源融合训练（后台运行中，2018 训练/2019 验证测试）
- [ ] 双源 vs GMCP-only 对比评估（重点：极端事件 CSI/F1 提升）

### 进行中
- [ ] ERA5+GMCP 双源融合训练（Epoch 1 完成：train_loss 68.8，收敛中）
- [ ] ERA5 2000-2017 训练数据下载（GCS 后台下载，每月 ~2 分钟）

### 待完成
- [ ] 双源模型评估对比
- [ ] 全量数据（2000-2017）完整训练
- [ ] 基线模型对比（GenCast/GraphCast/Pangu，2018/2019 拉齐评测集）
- [ ] 消融实验（物理约束/极端分支/异质性模块）
- [ ] 超参数调优
- [ ] 论文撰写

## 目标产出

### 代码产出
- 高质量的数据处理模块
- 创新的PhyDiff-Net模型架构
- 完整的训练和评估流程
- 充分的实验验证

### 报告产出
- SOTA性能分析报告
- 技术调研报告
- 算法设计方案
- 实验评估报告

## 实验计划

### 当前实验：ERA5+GMCP 双源融合（阶段四）

| 实验 | 数据 | 状态 | 目的 |
|------|------|------|------|
| E1: GMCP-only baseline | 2000-2001 训练 | ✅ 完成 | 验证管道，量化"无大气输入"的极限 |
| E2: ERA5+GMCP 双源 | 2018 训练/2019 验证测试 | 🔄 训练中 | 验证大气输入对极端事件的提升 |
| E3: 双源 vs 单源对比 | 同 E2 测试集 | ⏳ 待 E2 完成 | 量化 ERA5 贡献（消融第一项） |
| E4: 全量双源训练 | 2000-2017 训练 | ⏳ 数据下载中 | 完整容量模型 |
| E5: SOTA 对比 | 2018/2019 拉齐评测集 | ⏳ 待 E4 | 对标 GenCast/GraphCast/Pangu |

### 关键基线指标（E1，GMCP-only，3 epoch）

| 指标 | 数值 | 说明 |
|------|------|------|
| 轻雨 (≥5mm/6h) CSI | 0.162 | 基本检测能力 |
| 中雨 (≥10mm/6h) CSI | 0.117 | 可识别 |
| 大雨 (≥25mm/6h) CSI | 0.001 | **失败——双源实验的改进目标** |
| RMSE | 0.805 mm/6h | 连续值合理 |
| 相关性 | 0.483 | 空间结构中等 |

### 技术决策记录

| 决策 | 理由 | 日期 |
|------|------|------|
| WeatherBench2 替代 CDS 直下 | CDS 每小时文件下载太慢（4 天/2000 年），GCS 月度数据 ~2 分钟 | 2026-08 |
| ERA5 17 通道组合 | u/v/T/q × 850/500/200 + 地面 5 变量，覆盖水汽输送与垂直运动前兆 | 2026-08 |
| ERA5 1.5° 中国裁剪 24×42 | 模型 CrossResolutionFusion 原生支持双分辨率融合 | 2026-08 |
| netCDF4 直接区域读取 | xarray 逐文件元数据开销是瓶颈，快 20 倍 | 2026-08 |

## 下一步计划

1. [x] 启动项目，制定详细计划
2. [x] 完成文献调研
3. [x] 完成技术调研
4. [x] 设计算法方案
5. [x] 实现模型代码
6. [x] 数据管道打通（GMCP 25 年 + ERA5 双源）
7. [x] GMCP-only baseline 训练与评估
8. [ ] ERA5+GMCP 双源融合训练与评估（进行中）
9. [ ] 评估对比SOTA
10. [ ] 迭代优化直到超越
