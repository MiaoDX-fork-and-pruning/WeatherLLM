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
- **ERA5再分析数据**: `F:\ERA5再分析数据下载`
- **GMCP降水数据**: `F:\GMCP_Precipitation`

### 数据特点
| 数据集 | 分辨率 | 时间范围 | 描述 |
|--------|--------|----------|------|
| ERA5 | 0.25° | 1979-present | ECMWF再分析数据 |
| GMCP | 0.1° | 2000-2024 | 全球多源融合降水资料 |

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
├── docs/                  # 论文资料
│   ├── Bi-2023-Pangu-Nature.pdf
│   ├── Lam-2023-GraphCast-Science.pdf
│   ├── Price-2024-GenCast-Nature.pdf
│   └── Bodnar-2025-Aurora-Nature.pdf
├── data/                  # 数据说明
│   └── data.md
├── src/                   # 源代码（待创建）
│   ├── data/              # 数据处理
│   ├── models/            # 模型实现
│   ├── training/          # 训练流程
│   └── evaluation/        # 评估脚本
└── experiments/           # 实验记录（待创建）
```

## 目标产出

### 代码产出
- 高质量的数据处理模块
- 创新的模型架构实现
- 完整的训练和评估流程
- 充分的实验验证

### 论文产出
- Nature级别论文
- 创新的降水预报方法
- 充分的实验结果
- 可复现的代码

## 下一步计划

1. [ ] 启动项目，制定详细计划
2. [ ] 完成文献调研
3. [ ] 完成技术调研
4. [ ] 设计算法方案
5. [ ] 实现模型代码
6. [ ] 进行模型训练
7. [ ] 完成论文撰写
