# 项目日志

**项目名称**: PhyDiff-Net 降水预报AI模型
**开始时间**: 2026-06-15
**最后更新**: 2026-08-27

---

## 一、已完成工作

### 1.1 项目启动阶段
- [x] 创建项目目录结构
- [x] 制定PROJECT_RULES.md项目规范
- [x] 建立9个专业agent团队
- [x] 制定Git版本控制规范
- [x] 更新README.md项目说明

### 1.2 文献研究阶段
- [x] 分析Pangu-Weather (Nature, 2023) - 3D Swin Transformer
- [x] 分析GraphCast (Science, 2023) - 图神经网络
- [x] 分析GenCast (Nature, 2024) - 扩散模型
- [x] 分析Aurora (Nature, 2025) - 多尺度Transformer
- [x] 提取SOTA性能指标和基线

### 1.3 技术调研阶段
- [x] 调研AI降水预报最新技术
- [x] 分析数据融合策略
- [x] 识别超越SOTA的方向
- [x] 制定GMCP数据利用策略

### 1.4 算法设计阶段
- [x] 设计PhyDiff-Net架构
- [x] 设计四大核心模块
- [x] 制定四阶段训练策略
- [x] 设计多任务损失函数

### 1.5 代码实现阶段
- [x] 创建模型配置文件 (model_config.yaml)
- [x] 创建数据配置文件 (data_config.yaml)
- [x] 创建训练配置文件 (training_config.yaml)
- [x] 实现多尺度编码器 (encoder.py)
- [x] 实现物理约束扩散模块 (diffusion.py)
- [x] 实现极端事件感知分支 (extreme_branch.py)
- [x] 实现时空异质性建模 (heterogeneity.py)
- [x] 实现PhyDiff-Net主模型 (phydiff_net.py)
- [x] 实现损失函数 (losses/)
- [x] 实现评估指标 (evaluation/metrics.py)
- [x] 实现可视化工具 (evaluation/visualization.py)
- [x] 实现训练器类 (trainer.py)
- [x] 实现训练脚本 (train.py)

### 1.6 评测集拉齐分析（重要发现）
- [x] 分析GenCast评测集：ERA5 2019年
- [x] 分析GraphCast评测集：ERA5 2018年
- [x] 分析Pangu评测集：ERA5 2018年
- [x] **关键发现：各模型评测集不同！**
- [x] 制定拉齐策略：同时准备2018年和2019年两个评测集
- [x] 更新ERA5下载脚本注释，明确需要下载2018-2019年
- [x] 更新项目报告，说明评测集选择策略

### 1.7 多Agent并行工作（第二轮）
- [x] 数据收集agent：创建download_era5_eval.py评测数据下载脚本 (commit: 54955f8)
- [x] 数据预处理agent：创建preprocess_era5.py预处理脚本 (commit: 0cbabff)
- [x] 模型开发agent：修复bug，创建训练脚本和配置 (commit: e00e2b5)
- [x] 评估agent：创建evaluate_benchmark.py和compare_sota.py (commit: 78c2f40)
- [x] 实现推理脚本 (inference.py)
- [x] 实现工具函数 (config.py, logger.py, seed.py, checkpoint.py, device.py)
- [x] 实现数据集类 (dataset.py)
- [x] 实现数据加载器 (dataloader.py)
- [x] 修复.gitignore规则
- [x] 模型测试通过（502M参数，前向传播正常）
- [x] 创建数据预处理脚本 (scripts/preprocess_data.py)
- [x] 创建训练启动脚本 (scripts/run_training.py)
- [x] 创建data-collector agent
- [x] 完成数据收集报告 (reports/research/data_collection_report.md)
- [x] 创建ERA5下载脚本 (scripts/download_era5.py)
- [x] 创建ERA5下载指南 (reports/research/era5_download_guide.md)
- [x] 配置CDS API密钥
- [x] 接受CDS数据集许可证
- [x] 启动ERA5数据下载（2000-2005年）

### 1.10 GMCP-only Baseline 完整闭环（2026-08-12 至 2026-08-14）
- [x] 预处理性能优化：netCDF4 直接区域读取，单月 787s→38s（快 20 倍）
- [x] 修复模型 5D 输入处理（_maybe_resize_gmcp 维度错配 + squeeze）
- [x] GMCP 6h 标签全量生成：25 年 36,528 窗口（160 分钟）
- [x] GMCPExtremeLoss 极端事件损失（9 项单元测试）
- [x] GMCP-only 评估脚本，训练→评估闭环打通
- [x] Baseline 训练 3 epoch（val_loss 108→102→98）
- [x] 训练 resume 功能
- [x] Baseline 评估：轻中雨 CSI 0.12-0.23，RMSE 0.805mm/6h（冬季测试集）

### 1.11 ERA5+GMCP 双源融合（2026-08-27）
- [x] GMCPERA5Dataset 双源数据集：17 通道 ERA5（4 变量×3 层+5 地面），时间对齐验证（6 项测试）
- [x] 双源训练脚本 train_gmcp_era5.py + 双源评估支持
- [x] E2 双源首训（2018 训练，best Epoch 2 val_loss 12.20）
- [x] E3 公平对比（同 2019H2 测试集）：双源首跑全面落后单源——根因训练量仅 1/3 + 过拟合
- [x] **重要修正**：E1 冬季测试集低估极端能力，夏季同模型 heavy F1=0.138
- [x] ERA5 数据重大发现：6 月下载的 1979-2017 月度文件（234GB）一直在磁盘，仅合并步骤失败被误判丢失
- [x] 修复损坏的 2000-04 月份（进程卡死写坏 time 值，重下载）
- [x] 合并脚本 merge_era5_monthly.py（逐变量控制内存）
- [x] GMCPERA5Dataset 支持目录输入（多年度文件惰性 concat）
- [x] E4 全量训练配置就绪（2000-2016 训练 / 2017 验证 / 2019H2 测试）
- [ ] ERA5 年度文件合并（后台进行中，每年 ~3.7 分钟）
- [ ] E4 全量双源训练（等合并完成）

### 1.8 GMCP 真实数据管道打通（2026-06 至 2026-07）
- [x] 新增 `src/data/gmcp_reader.py`，修复真实 GMCP 文件格式 5 处不兼容（路径结构/变量名/坐标名/时间维度/查找逻辑）
- [x] 重写 `GMCPPreprocessor.load_data()` 基于 GMCPDataset
- [x] 完成 300 个真实 6h 窗口的探索性分析（无缺失无负值，6h 均值 0.58mm，极端事件稀少）
- [x] GMCP reader 11 项测试通过
- [x] 详见 `reports/analysis/gmcp_compatibility_fix_20260630.md` 与 `gmcp_analysis_20260630.md`

### 1.9 GMCP-only 训练通路（2026-06 至 2026-08）
- [x] PhyDiffNet 新增 `use_era5` 开关，支持 GMCP 单源模式
- [x] 新增 `encoder_spatial_size` 下采样高分辨率 GMCP 避免显存爆炸
- [x] 修复 UNet skip connection 通道拼接错配
- [x] 新增 `src/data/gmcp_sequence_dataset.py`、`scripts/train_gmcp_only.py`、`scripts/preprocess_gmcp_6h.py`
- [x] 新增 configs/training_gmcp_only.yaml 与 verify 配置
- [x] GMCP 相关 28 项测试全部通过
- [x] 全部改动 commit 并 push 至 origin/main（`70b3842`、`cea52ab`、`1a583a6`）

---

## 二、失败/问题记录

### 2.1 技术问题
| 问题 | 状态 | 解决方案 |
|------|------|----------|
| PDF读取工具需要pdftoppm | 已解决 | 使用WebSearch搜索论文信息 |
| 自定义agent类型未被系统加载 | 已解决 | 使用general-purpose agent |
| 目录结构不完整 | 已解决 | 创建完整的子目录结构 |

### 2.2 规范问题
| 问题 | 状态 | 解决方案 |
|------|------|----------|
| 临时文件命名不规范 | 已解决 | 删除temp_extract.py，制定规范 |
| commit message不详细 | 已解决 | 更新规范要求详细描述 |
| commit后不push | 已解决 | 强制要求验证后立即push |

### 2.3 Agent执行问题
| 问题 | 状态 | 解决方案 |
|------|------|----------|
| .gitignore规则太宽泛 | 已解决 | 更新.gitignore，使用更精确的规则 |
| 部分agent输出文件为空 | 已解决 | 重新启动失败的agent |
| agent执行时间过长 | 已解决 | 拆分大任务，使用更简单的任务描述 |

---

## 三、已确认事项

### 3.1 技术方案确认
- [x] **模型架构**: PhyDiff-Net (物理约束扩散网络)
- [x] **数据融合**: ERA5 (0.25°) + GMCP (0.1°) 三阶段渐进式融合
- [x] **训练策略**: 四阶段渐进式训练 (预训练→融合预训练→微调→极端事件增强)
- [x] **损失函数**: 多任务组合损失 (MSE + CSI + 物理约束 + 极端事件)

### 3.2 性能目标确认
- [x] **极端降水CSI**: >0.5 (当前SOTA约0.4-0.45)
- [x] **强降水POD**: >0.8 (当前SOTA约0.7-0.75)
- [x] **强降水FAR**: <0.3 (当前SOTA约0.35-0.4)
- [x] **降水RMSE**: 比GenCast降低10-15%

### 3.3 数据资源确认
- [x] **ERA5数据位置**: `F:\ERA5再分析数据下载`
- [x] **GMCP数据位置**: `F:\GMCP_Precipitation`
- [x] **训练时间范围**: 1979-2017年（与GenCast/GraphCast/Pangu对齐）
- [x] **验证时间范围**: 2018年ERA5数据
- [x] **测试时间范围A**: 2019年ERA5数据（与GenCast对比）
- [x] **测试时间范围B**: 2018年ERA5数据（与GraphCast/Pangu对比）

### 3.4 创新点确认
- [x] 物理约束扩散模型
- [x] 0.1°高分辨率直接训练
- [x] 极端事件感知机制
- [x] 多尺度时空异质性建模
- [x] ERA5-GMCP联合预训练

---

## 四、待人工处理事项

### 4.1 数据相关
- [ ] **下载ERA5 2018年数据**（重要！）: 用于验证集 + 与GraphCast/Pangu对比的测试集
- [ ] **下载ERA5 2019年数据**（重要！）: 用于与GenCast对比的测试集
- [ ] **确认ERA5数据格式**: 需要人工检查实际数据文件格式
- [ ] **确认GMCP数据格式**: 需要人工检查实际数据文件格式
- [ ] **数据质量检查**: 需要人工验证数据质量
- [ ] **数据下载确认**: 需要确认数据是否已下载完整

### 4.2 模型相关
- [ ] **GPU资源确认**: 需要确认可用的GPU资源
- [ ] **超参数调优**: 需要人工调整关键超参数
- [ ] **模型验证**: 需要人工验证模型输出的合理性

### 4.3 实验相关
- [ ] **基线模型对比**: 需要下载并运行GenCast等基线模型
- [ ] **评估指标验证**: 需要人工验证评估指标的正确性
- [ ] **结果可视化**: 需要人工检查可视化结果

### 4.4 论文相关
- [ ] **论文撰写**: 有专人负责
- [ ] **实验设计**: 需要人工确认实验设计
- [ ] **结果分析**: 需要人工分析实验结果

---

## 五、关键决策记录

### 5.1 架构决策
| 决策 | 理由 | 日期 |
|------|------|------|
| 选择扩散模型 | GenCast证明扩散模型在概率预报上的优势 | 2026-06-15 |
| 选择物理约束 | 提高物理一致性，改善极端事件预测 | 2026-06-15 |
| 选择双分支架构 | 专门处理极端事件，避免MSE导致的极端值低估 | 2026-06-15 |

### 5.2 数据决策
| 决策 | 理由 | 日期 |
|------|------|------|
| 直接训练0.1° | 避免降尺度信息损失，利用GMCP优势 | 2026-06-15 |
| 四阶段训练 | 渐进式学习，避免灾难性遗忘 | 2026-06-15 |
| 多任务损失 | 同时优化连续和分类指标 | 2026-06-15 |

---

## 六、下一步计划

### 6.1 短期计划 (本周)
- [x] 完成数据处理管道实现
- [x] 完成模型架构实现
- [x] 完成训练流程实现
- [x] 完成评估模块实现
- [x] 创建数据预处理脚本
- [x] 创建训练启动脚本
- [x] 完成数据收集报告
- [x] 创建ERA5下载脚本
- [x] 创建ERA5下载指南
- [x] 配置CDS API密钥
- [x] 接受CDS数据集许可证
- [x] 启动ERA5数据下载（2000-2005年）
- [x] 打通 GMCP 真实数据管道
- [x] 实现 GMCP-only 训练通路
- [ ] 生成 6h 累计标签数据（运行 preprocess_gmcp_6h.py）
- [ ] GMCP-only 端到端训练冒烟验证（verify 配置 1 epoch）
- [ ] GMCP-only 正式小规模训练（5 epoch）
- [ ] 监控ERA5下载进度

### 6.2 中期计划 (本月)
- [ ] 数据预处理和验证
- [ ] 模型小规模测试训练
- [ ] 基线模型对比实验
- [ ] 性能评估和分析

### 6.3 长期计划 (本季度)
- [ ] 完整模型训练
- [ ] 超参数调优
- [ ] 消融实验
- [ ] 论文撰写准备

---

## 七、风险和缓解措施

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| GPU资源不足 | 中 | 高 | 申请更多资源，优化代码效率 |
| 数据质量问题 | 中 | 中 | 数据清洗，异常值处理 |
| 模型收敛困难 | 低 | 高 | 调整学习率，使用梯度裁剪 |
| 极端事件样本不足 | 高 | 中 | 数据增强，过采样策略 |
| 物理约束过强 | 中 | 中 | 可学习权重，渐进式约束 |

---

## 八、参考资源

### 8.1 论文
- GenCast: https://www.nature.com/articles/s41586-024-08252-9
- Aurora: https://www.nature.com/articles/s41586-024-07744-y
- Pangu: https://www.nature.com/articles/s41586-023-06185-3
- GraphCast: Science, 2023

### 8.2 代码库
- WeatherBench2: https://arxiv.org/abs/2401.05933
- FuXi: https://github.com/ai2es/FuXi

### 8.3 数据
- ERA5: ECMWF Climate Data Store
- GMCP: 团队自研数据

---

**日志维护者**: weather-planner
**更新频率**: 每次重要进展后更新
