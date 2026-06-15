# 项目目录规范

## 目录结构

```
weather/
├── docs/           # 论文、文档资料
│   ├── papers/     # 论文PDF
│   └── *.md        # 文档文件
│
├── data/           # 数据相关
│   ├── raw/        # 原始数据
│   ├── processed/  # 预处理后的数据
│   ├── scripts/    # 数据处理脚本
│   └── README.md   # 数据说明
│
├── src/            # 源代码
│   ├── data/       # 数据加载和预处理模块
│   ├── models/     # 模型定义
│   ├── training/   # 训练流程
│   ├── evaluation/ # 评估脚本
│   ├── utils/      # 工具函数
│   └── configs/    # 配置文件
│
├── models/         # 训练好的模型权重
│   ├── checkpoints/  # 模型检查点
│   └── final/      # 最终模型
│
├── outputs/        # 代码运行输出
│   ├── logs/       # 训练日志
│   ├── predictions/ # 预测结果
│   └── figures/    # 可视化图表
│
├── reports/        # 报告类产出
│   ├── research/   # 调研报告
│   ├── analysis/   # 分析报告
│   ├── experiments/ # 实验报告
│   └── reviews/    # 审核报告
│
├── experiments/    # 实验配置和记录
│   ├── configs/    # 实验配置
│   └── results/    # 实验结果
│
└── README.md       # 项目说明
```

## Agent产出规范

### paper-reader
- 输出位置: `reports/analysis/paper_analysis_*.md`
- 命名格式: `paper_analysis_{论文名}_{日期}.md`

### paper-reviewer
- 输出位置: `reports/reviews/paper_review_*.md`
- 命名格式: `paper_review_{论文名}_{日期}.md`

### weather-researcher
- 输出位置: `reports/research/research_*.md`
- 命名格式: `research_{主题}_{日期}.md`

### weather-research-reviewer
- 输出位置: `reports/reviews/research_review_*.md`
- 命名格式: `research_review_{主题}_{日期}.md`

### algorithm-architect
- 输出位置: `reports/analysis/algorithm_design_*.md`
- 命名格式: `algorithm_design_{版本}_{日期}.md`
- 代码输出: `src/models/`

### algorithm-reviewer
- 输出位置: `reports/reviews/algorithm_review_*.md`
- 命名格式: `algorithm_review_{版本}_{日期}.md`

### weather-model-trainer
- 代码输出: `src/`
- 模型输出: `models/checkpoints/`
- 日志输出: `outputs/logs/`
- 预测输出: `outputs/predictions/`
- 报告输出: `reports/experiments/training_report_*.md`

### weather-model-reviewer
- 输出位置: `reports/reviews/model_review_*.md`
- 命名格式: `model_review_{模型名}_{日期}.md`

### weather-planner
- 输出位置: `reports/analysis/project_plan.md`
- 日志位置: `reports/analysis/project_log.md`
- 更新位置: 各agent的输出

### 项目日志维护（强制执行）
- **维护者**: weather-planner
- **更新频率**: 每次重要进展后更新
- **日志内容**:
  - 已完成工作
  - 失败/问题记录
  - 已确认事项
  - 待人工处理事项
  - 关键决策记录
  - 下一步计划
  - 风险和缓解措施

## Git版本控制规范（强制执行）

### 核心原则
1. **禁止临时文件**: 不允许存在 `xxx1.py`, `xxx2.py`, `temp_xxx.py` 等临时文件
2. **所有变更必须通过Git管理**: 任何代码修改都必须commit
3. **原子提交**: 每个commit只做一个逻辑完整的修改
4. **有意义的commit message**: 清晰描述修改内容
5. **验证后立即commit并push**: 每次验证完代码功能后，立即commit并push到远程仓库

### 分支策略
```
main: 生产分支，保持稳定
├── develop: 开发分支
│   ├── feature/data-processing: 数据处理模块
│   ├── feature/model-architecture: 模型架构
│   ├── feature/training: 训练流程
│   └── feature/evaluation: 评估模块
```

### Commit Message格式
```
<type>(<scope>): <description>

类型(type):
  feat:     新功能
  fix:      修复bug
  refactor: 重构（不改变功能）
  docs:     文档更新
  test:     测试相关
  chore:    构建/工具相关
  perf:     性能优化
  experiment: 实验相关

范围(scope):
  data:     数据处理
  model:    模型定义
  training: 训练流程
  evaluation: 评估脚本
  utils:    工具函数
  configs:  配置文件

示例:
feat(data): implement ERA5 preprocessing pipeline
fix(model): resolve memory leak in attention module
experiment(training): run baseline model comparison
docs(research): add precipitation SOTA survey
refactor(utils): simplify data loading utilities
```

### 工作流程（强制执行）
```
1. 编写/修改代码
       ↓
2. 测试验证（确保代码可运行）
       ↓
3. git add <文件>
       ↓
4. git commit -m "<type>(<scope>): <详细描述>"
       ↓
5. git push（立即推送到远程仓库）
```

### Commit Message详细格式
```
<type>(<scope>): <简短描述>

<详细描述修改了什么，为什么这么修改，解决了什么问题>

- 修改点1
- 修改点2
- 修改点3

Co-Authored-By: Claude <noreply@anthropic.com>
```

示例:
```
feat(data): implement ERA5 preprocessing pipeline with spatial alignment

实现ERA5数据预处理管道，包括：
- 数据加载和格式转换
- 空间配准到0.1°网格
- 时间对齐到6小时步长
- Z-score标准化处理
- 缺失值时空插值修复

该模块为PhyDiff-Net提供标准化的ERA5输入数据。

Co-Authored-By: Claude <noreply@anthropic.com>
```

### 禁止事项
- ❌ 创建 temp_xxx.py, xxx1.py, xxx2.py 等临时文件
- ❌ 直接在main分支上修改代码
- ❌ 提交未测试的代码
- ❌ 使用模糊的commit message（如 "update", "fix"）
- ❌ commit后不push

## 文件命名规范

### 代码文件
- 使用 `snake_case.py` 命名
- 每个文件有明确的功能定义
- 示例: `era5_loader.py`, `precipitation_model.py`

### 配置文件
- 使用 `snake_case.yaml` 或 `snake_case.json`
- 示例: `training_config.yaml`, `model_config.json`

### 报告文件
- 使用 `{类型}_{主题}_{日期}.md`
- 示例: `paper_analysis_GenCast_20260615.md`

### 模型文件
- 使用 `{模型名}_{epoch}_{metric}.pth`
- 示例: `PrecipNet_epoch_50_csi_0.85.pth`
