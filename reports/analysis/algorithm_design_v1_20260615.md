# 降水预报AI模型算法设计方案 v1

## 一、设计目标

### 核心目标
构建一个基于扩散模型与物理约束融合的高分辨率降水预报系统，通过创新的多尺度时空建模和极端事件专用机制，在中国区域实现超越GenCast的降水预报性能。

### 关键指标
| 指标 | 目标值 | SOTA基准 | 提升幅度 |
|------|--------|----------|----------|
| 极端降水CSI | >0.5 | 0.4-0.45 | 10-25% |
| 强降水POD | >0.8 | 0.7-0.75 | 7-14% |
| 强降水FAR | <0.3 | 0.35-0.4 | 15-25% |
| 降水RMSE | 比GenCast降低10-15% | 基准线 | 10-15% |

### 创新点
1. **物理约束扩散模型**：将大气动力学方程作为扩散过程的物理引导
2. **0.1°高分辨率直接训练**：突破传统降尺度限制，直接学习高分辨率降水分布
3. **极端事件感知机制**：专门的分支和损失函数处理极端降水事件
4. **多尺度时空异质性建模**：自适应捕捉不同尺度的降水特征
5. **ERA5-GMCP联合预训练**：充分利用多源数据的互补优势

---

## 二、数据融合方案

### 2.1 数据源

#### ERA5再分析数据
- **分辨率**: 0.25° (约28km)
- **时间范围**: 1979-present (45年)
- **变量**: 三维大气场（温度、风场、湿度、位势高度等）
- **优势**: 物理一致性好、全球覆盖、时间序列长
- **局限**: 分辨率较低，对局地极端降水捕捉能力有限

#### GMCP降水数据
- **分辨率**: 0.1° (约10km)
- **时间范围**: 2000-2024 (25年)
- **类型**: 中国区域台站校正融合产品
- **优势**: 高分辨率、经过地面站校正、降水精度高
- **局限**: 仅中国区域、时间序列相对较短

### 2.2 融合策略

采用**三阶段渐进式融合**方案：

```
阶段1: 空间配准 (Spatial Alignment)
├── 将ERA5双线性插值到0.1°网格
├── 建立ERA5变量与GMCP降水的相关性映射
└── 生成ERA5的0.1°降水先验场

阶段2: 特征融合 (Feature Fusion)  
├── 设计跨分辨率注意力机制
├── ERA5提供大尺度环流背景
├── GMCP提供高分辨率降水细节
└── 学习多源特征的互补表示

阶段3: 质量增强 (Quality Enhancement)
├── 利用ERA5的物理一致性约束GMCP
├── 识别并修正GMCP中的观测误差
└── 生成高质量融合训练标签
```

### 2.3 预处理流程

#### 数据质量控制
1. **缺失值处理**: 时空插值 + 物理约束修复
2. **异常值检测**: 基于物理阈值和统计分布的双重检验
3. **时间对齐**: 统一到6小时时间步长
4. **空间配准**: 统一到0.1°网格系统

#### 特征工程
```python
# 输入特征组设计
input_features = {
    'era5_3d': ['temperature', 'u_wind', 'v_wind', 'geopotential', 'relative_humidity'],
    'era5_2d': ['surface_pressure', 'total_precipitation', 'convective_precipitation'],
    'gmcp': ['precipitation_rate'],
    'static': ['topography', 'land_mask', 'latitude', 'longitude'],
    'temporal': ['hour_of_day', 'day_of_year', 'season_encoding']
}

# 输出目标
target = {
    'precipitation_0h': '0-6h累计降水',
    'precipitation_6h': '6-12h累计降水',
    'precipitation_12h': '12-24h累计降水'
}
```

#### 归一化策略
- **ERA5变量**: Z-score标准化 (基于1979-2019气候态)
- **GMCP降水**: 对数变换 + Min-Max归一化
- **地形数据**: 高程除以最大高程值

---

## 三、模型架构设计

### 3.1 整体架构

采用**PhyDiff-Net (Physics-guided Diffusion Network)**架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    PhyDiff-Net 整体架构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Encoder   │    │   Diffusion │    │   Decoder   │     │
│  │   Module    │───▶│   Module    │───▶│   Module    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Multi-Scale │    │   Physics   │    │  Extreme    │     │
│  │ Attention   │    │ Constraints │    │  Branch     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │   Output Head   │                       │
│                   │ (Multi-task)    │                       │
│                   └─────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心模块

#### Module A: 多尺度时空编码器 (Multi-Scale Spatiotemporal Encoder)

**功能**: 
从ERA5和GMCP输入中提取多尺度时空特征，同时捕捉大尺度环流背景和局地降水细节。

**架构设计**:
```python
class MultiScaleEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_scales=3):
        super().__init__()
        # 多尺度卷积分支
        self.scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim//num_scales, 
                         kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, hidden_dim//num_scales),
                nn.SiLU()
            ) for _ in range(num_scales)
        ])
        
        # 时空注意力融合
        self.temporal_attention = TemporalAttention(hidden_dim)
        self.spatial_attention = SpatialAttention(hidden_dim)
        
        # 跨分辨率融合
        self.cross_resolution_fusion = CrossResolutionFusion(
            era5_resolution=0.25,
            gmcp_resolution=0.1,
            hidden_dim=hidden_dim
        )
    
    def forward(self, era5_data, gmcp_data, timestamps):
        # 多尺度特征提取
        multi_scale_features = []
        for branch in self.scale_branches:
            scale_feat = branch(era5_data)
            multi_scale_features.append(scale_feat)
        
        # 特征拼接
        concat_features = torch.cat(multi_scale_features, dim=1)
        
        # 时空注意力
        temporal_feat = self.temporal_attention(concat_features, timestamps)
        spatial_feat = self.spatial_attention(temporal_feat)
        
        # 跨分辨率融合
        fused_features = self.cross_resolution_fusion(
            spatial_feat, gmcp_data
        )
        
        return fused_features
```

**创新点**:
- 自适应多尺度卷积核（3×3, 5×5, 7×7）捕捉不同尺度特征
- 时空注意力机制动态调整时间步的重要性权重
- 跨分辨率融合模块显式建模ERA5和GMCP的分辨率差异

**输入/输出规格**:
- 输入: ERA5数据 [B, C_era5, H/4, W/4]，GMCP数据 [B, C_gmcp, H, W]
- 输出: 多尺度特征 [B, hidden_dim, H, W]

---

#### Module B: 物理约束扩散模块 (Physics-Constrained Diffusion Module)

**功能**:
基于扩散模型生成降水预测，同时通过物理约束确保预测结果符合大气动力学规律。

**架构设计**:
```python
class PhysicsConstrainedDiffusion(nn.Module):
    def __init__(self, hidden_dim=256, num_diffusion_steps=1000):
        super().__init__()
        self.num_steps = num_diffusion_steps
        
        # U-Net去噪网络
        self.denoiser = UNet(
            in_channels=hidden_dim + 1,  # 特征 + 噪声降水
            out_channels=1,
            hidden_channels=hidden_dim,
            num_res_blocks=4,
            attention_resolutions=[16, 8]
        )
        
        # 物理约束模块
        self.physics_constraints = PhysicsConstraintModule(
            equations=['continuity', 'moisture_conservation', 'energy_balance']
        )
        
        # 条件编码器（ERA5作为条件）
        self.condition_encoder = ConditionEncoder(hidden_dim)
    
    def forward(self, x, t, condition):
        # x: 噪声降水场
        # t: 时间步
        # condition: ERA5条件
        
        # 条件编码
        cond_features = self.condition_encoder(condition)
        
        # 去噪预测
        noise_pred = self.denoiser(x, t, cond_features)
        
        # 物理约束修正
        corrected_pred = self.physics_constraints(
            x - noise_pred, 
            condition
        )
        
        return corrected_pred
    
    def sample(self, condition, shape, device):
        """生成降水预测"""
        # 从高斯噪声开始
        x = torch.randn(shape, device=device)
        
        # 逆向扩散过程
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((x.shape[0],), t, device=device)
            predicted_x0 = self.forward(x, t_tensor, condition)
            
            # 应用物理约束
            if t > 0:
                noise = torch.randn_like(x)
                x = predicted_x0 + sqrt_alpha_t * noise
            else:
                x = predicted_x0
        
        return x
```

**物理约束设计**:
```python
class PhysicsConstraintModule(nn.Module):
    def __init__(self, equations):
        super().__init__()
        self.constraints = nn.ModuleDict({
            'continuity': ContinuityConstraint(),
            'moisture_conservation': MoistureConstraint(),
            'energy_balance': EnergyConstraint()
        })
        
        # 可学习的约束权重
        self.constraint_weights = nn.Parameter(
            torch.ones(len(equations)) / len(equations)
        )
    
    def forward(self, precipitation, condition):
        total_loss = 0
        corrected = precipitation
        
        for i, (name, constraint) in enumerate(self.constraints.items()):
            constraint_loss, correction = constraint(corrected, condition)
            total_loss += self.constraint_weights[i] * constraint_loss
            corrected = corrected - 0.1 * correction  # 渐进式修正
        
        self.physical_loss = total_loss
        return corrected
```

**创新点**:
- 将大气连续性方程、水汽守恒方程融入扩散过程
- 可学习的约束权重自适应调整不同物理规律的重要性
- 渐进式修正策略避免过度约束导致的模式模糊

---

#### Module C: 极端事件感知分支 (Extreme Event Aware Branch)

**功能**:
专门处理极端降水事件，通过定制化的特征提取和损失函数提升极端事件的预测精度。

**架构设计**:
```python
class ExtremeEventBranch(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # 极端事件检测器
        self.extreme_detector = ExtremeDetector(
            thresholds={
                'heavy': 25.0,      # mm/6h
                'very_heavy': 50.0,  # mm/6h
                'extreme': 100.0     # mm/6h
            }
        )
        
        # 极端事件专用编码器
        self.extreme_encoder = ExtremeEncoder(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=4
        )
        
        # 强度预测头
        self.intensity_head = IntensityHead(hidden_dim)
        
        # 空间范围预测头
        self.extent_head = ExtentHead(hidden_dim)
    
    def forward(self, features, is_extreme=None):
        # 检测极端事件
        if is_extreme is None:
            is_extreme = self.extreme_detector(features)
        
        # 极端事件特征增强
        extreme_features = self.extreme_encoder(features * is_extreme.unsqueeze(1))
        
        # 强度和范围预测
        intensity = self.intensity_head(extreme_features)
        extent = self.extent_head(extreme_features)
        
        return intensity, extent, is_extreme
```

**极端事件损失函数**:
```python
class ExtremeEventLoss(nn.Module):
    def __init__(self, weights={'heavy': 1.0, 'very_heavy': 2.0, 'extreme': 5.0}):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred, target, extreme_mask):
        """
        pred: 预测降水 [B, 1, H, W]
        target: 真实降水 [B, 1, H, W]
        extreme_mask: 极端事件掩码 [B, 1, H, W]
        """
        losses = {}
        
        for level, weight in self.weights.items():
            level_mask = extreme_mask[level]
            
            if level_mask.sum() > 0:
                # 极端事件专用损失
                level_loss = self.compute_level_loss(
                    pred[level_mask], 
                    target[level_mask]
                )
                losses[level] = weight * level_loss
            else:
                losses[level] = torch.tensor(0.0, device=pred.device)
        
        return sum(losses.values())
    
    def compute_level_loss(self, pred, target):
        """组合损失：MSE + CSI + 分位数损失"""
        mse_loss = F.mse_loss(pred, target)
        csi_loss = 1 - self.compute_csi(pred, target)
        quantile_loss = self.quantile_loss(pred, target, q=0.95)
        
        return mse_loss + 0.5 * csi_loss + 0.3 * quantile_loss
```

**创新点**:
- 多级极端事件检测（强降水、特强降水、极端降水）
- 极端事件专用编码器捕捉极端事件的特殊模式
- 组合损失函数同时优化MSE、CSI和分位数损失
- 动态权重调整，训练后期增加极端事件权重

---

#### Module D: 时空异质性建模模块 (Spatiotemporal Heterogeneity Module)

**功能**:
建模降水场的时空异质性，包括空间非平稳性和时间变率特征。

**架构设计**:
```python
class SpatiotemporalHeterogeneity(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8):
        super().__init__()
        
        # 空间非平稳性建模
        self.spatial_heterogeneity = SpatialHeterogeneity(
            hidden_dim=hidden_dim,
            num_regions=4  # 中国四大地理区域
        )
        
        # 时间变率建模
        self.temporal_variability = TemporalVariability(
            hidden_dim=hidden_dim,
            num_frequencies=8
        )
        
        # 自适应特征调制
        self.adaptive_modulation = AdaptiveModulation(hidden_dim)
    
    def forward(self, features, timestamps):
        # 空间异质性特征
        spatial_het = self.spatial_heterogeneity(features)
        
        # 时间变率特征
        temporal_var = self.temporal_variability(features, timestamps)
        
        # 自适应调制
        modulated_features = self.adaptive_modulation(
            features, spatial_het, temporal_var
        )
        
        return modulated_features
```

**空间非平稳性建模**:
```python
class SpatialHeterogeneity(nn.Module):
    def __init__(self, hidden_dim, num_regions):
        super().__init__()
        # 区域特定特征提取
        self.region_encoders = nn.ModuleList([
            RegionEncoder(hidden_dim) for _ in range(num_regions)
        ])
        
        # 区域边界平滑
        self.boundary_smoothing = BoundarySmoothing(
            kernel_size=5,
            sigma=1.0
        )
        
        # 区域权重预测
        self.region_weight_predictor = nn.Linear(hidden_dim, num_regions)
    
    def forward(self, features):
        B, C, H, W = features.shape
        
        # 预测区域权重
        region_weights = self.region_weight_predictor(
            features.mean(dim=[2, 3])
        )  # [B, num_regions]
        region_weights = F.softmax(region_weights, dim=-1)
        
        # 各区域特征提取
        region_features = []
        for i, encoder in enumerate(self.region_encoders):
            region_feat = encoder(features)
            region_features.append(region_feat * region_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1))
        
        # 特征融合
        heterogeneity_feature = sum(region_features)
        
        return heterogeneity_feature
```

**创新点**:
- 显式建模中国四大地理区域的降水特性差异
- 自适应区域权重根据输入动态调整
- 时间变率模块捕捉不同时间尺度的降水变化模式
- 边界平滑处理避免区域边界的人工伪影

---

### 3.3 降水专用设计

#### 损失函数设计

采用**多任务组合损失函数**，针对降水预报的特殊性进行优化：

```python
class PrecipitationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 基础损失
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=10.0)
        
        # 降水专用损失
        self.csi_loss = CSILoss(thresholds=[0.1, 5.0, 10.0, 25.0, 50.0])
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        self.quantile_loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        
        # 物理约束损失
        self.physics_loss = PhysicsConstraintLoss()
        
        # 极端事件损失
        self.extreme_loss = ExtremeEventLoss()
    
    def forward(self, predictions, targets, metadata):
        """
        predictions: 模型预测 [B, T, H, W]
        targets: 真实值 [B, T, H, W]
        metadata: 包含极端事件标记等元数据
        """
        losses = {}
        
        # 1. 基础回归损失
        losses['mse'] = self.mse_loss(predictions, targets)
        losses['huber'] = self.huber_loss(predictions, targets)
        
        # 2. 分类性能损失（降水/无降水）
        pred_binary = (predictions > 0.1).float()
        target_binary = (targets > 0.1).float()
        losses['focal'] = self.focal_loss(pred_binary, target_binary)
        
        # 3. CSI损失（不同强度阈值）
        losses['csi'] = self.csi_loss(predictions, targets)
        
        # 4. 分位数损失（捕捉分布）
        losses['quantile'] = self.quantile_loss(predictions, targets)
        
        # 5. 物理约束损失
        losses['physics'] = self.physics_loss(predictions, metadata)
        
        # 6. 极端事件损失
        if metadata.get('extreme_events') is not None:
            losses['extreme'] = self.extreme_loss(
                predictions, targets, metadata['extreme_events']
            )
        
        # 加权组合
        total_loss = (
            0.2 * losses['mse'] +
            0.1 * losses['huber'] +
            0.15 * losses['focal'] +
            0.25 * losses['csi'] +
            0.1 * losses['quantile'] +
            0.1 * losses['physics'] +
            0.1 * losses.get('extreme', 0.0)
        )
        
        losses['total'] = total_loss
        return losses
```

#### 评估指标体系

```python
class PrecipitationMetrics:
    """降水预报评估指标"""
    
    def __init__(self, thresholds=[0.1, 5.0, 10.0, 25.0, 50.0]):
        self.thresholds = thresholds
    
    def compute_all_metrics(self, predictions, targets):
        """计算所有评估指标"""
        metrics = {}
        
        for threshold in self.thresholds:
            prefix = f'precip_gt_{threshold}mm'
            
            # 基础指标
            metrics[f'{prefix}_csi'] = self.csi(predictions, targets, threshold)
            metrics[f'{prefix}_pod'] = self.pod(predictions, targets, threat='pod')
            metrics[f'{prefix}_far'] = self.pod(predictions, targets, threat='far')
            metrics[f'{prefix}_hss'] = self.hss(predictions, targets, threshold)
            
            # 面积统计
            metrics[f'{prefix}_area_bias'] = self.area_bias(
                predictions, targets, threshold
            )
        
        # 连续指标
        metrics['rmse'] = self.rmse(predictions, targets)
        metrics['mae'] = self.mae(predictions, targets)
        metrics['corr'] = self.correlation(predictions, targets)
        
        # 分布指标
        metrics['crps'] = self.crps(predictions, targets)
        
        return metrics
    
    def csi(self, pred, target, threshold):
        """临界成功指数"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        hits = (pred_binary * target_binary).sum()
        false_alarms = (pred_binary * (1 - target_binary)).sum()
        misses = ((1 - pred_binary) * target_binary).sum()
        
        csi = hits / (hits + false_alarms + misses + 1e-8)
        return csi
    
    def pod(self, pred, target, threat='pod'):
        """检测概率或虚警率"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        hits = (pred_binary * target_binary).sum()
        
        if threat == 'pod':
            return hits / (target_binary.sum() + 1e-8)
        else:  # far
            false_alarms = (pred_binary * (1 - target_binary)).sum()
            return false_alarms / (hits + false_alarms + 1e-8)
```

---

## 四、训练策略

### 4.1 训练流程

采用**四阶段渐进式训练**策略：

```
阶段1: 预训练 (Pre-training)
├── 目标: 学习通用大气表征
├── 数据: ERA5全球数据 (1979-2019)
├── 任务: 重建ERA5变量
├── 时长: 50 epochs
└── 输出: 通用大气编码器权重

阶段2: 多源融合预训练 (Multi-source Fusion Pre-training)
├── 目标: 学习ERA5-GMCP融合表征
├── 数据: ERA5+GMCP中国区域 (2000-2019)
├── 任务: 降水重建 + 物理一致性约束
├── 时长: 100 epochs
└── 输出: 融合编码器权重

阶段3: 下游任务微调 (Downstream Fine-tuning)
├── 目标: 优化降水预报性能
├── 数据: ERA5+GMCP中国区域 (2000-2022)
├── 任务: 多步降水预测
├── 时长: 200 epochs
└── 输出: 预测模型权重

阶段4: 极端事件增强 (Extreme Event Enhancement)
├── 目标: 提升极端事件预测能力
├── 数据: 极端事件过采样 (阈值>25mm)
├── 任务: 极端事件分类 + 强度回归
├── 时长: 50 epochs
└── 输出: 最终模型权重
```

### 4.2 优化策略

#### 优化器选择
```python
# 使用AdamW优化器，配合权重衰减
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# 多参数组策略
optimizer = torch.optim.AdamW([
    {'params': encoder.parameters(), 'lr': 1e-4},
    {'params': diffusion.parameters(), 'lr': 5e-5},
    {'params': extreme_branch.parameters(), 'lr': 2e-4},
    {'params': decoder.parameters(), 'lr': 1e-4}
], weight_decay=0.01)
```

#### 学习率调度
```python
# 余弦退火 + 预热策略
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-4, 5e-5, 2e-4, 1e-4],  # 各参数组最大学习率
    steps_per_epoch=len(train_loader),
    epochs=total_epochs,
    pct_start=0.1,  # 10%预热
    anneal_strategy='cos'
)

# 或使用WarmupCosineScheduler
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=10,
    total_epochs=total_epochs,
    base_lr=1e-4,
    min_lr=1e-6
)
```

#### 正则化策略
```python
class RegularizationStrategy:
    def __init__(self):
        self.strategies = {
            'dropout': nn.Dropout(p=0.1),
            'spatial_dropout': nn.Dropout2d(p=0.1),
            'weight_decay': 0.01,
            'label_smoothing': 0.1,
            'gradient_clip': 1.0,
            'stochastic_depth': 0.1
        }
    
    def apply(self, model, training_step):
        # 动态调整正则化强度
        if training_step < 1000:
            # 训练初期：轻度正则化
            dropout_rate = 0.05
        elif training_step < 5000:
            # 训练中期：标准正则化
            dropout_rate = 0.1
        else:
            # 训练后期：增强正则化
            dropout_rate = 0.15
        
        return dropout_rate
```

### 4.3 数据增强

#### 空间增强
```python
class SpatialAugmentation:
    def __init__(self):
        self.transforms = [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=[-90, 90]),
            RandomCrop(size=(128, 128)),
            RandomResizedCrop(size=(128, 128), scale=(0.8, 1.2)),
            ElasticTransform(alpha=50.0, sigma=5.0),
            GridDistortion(num_steps=5, distort_limit=0.05)
        ]
    
    def __call__(self, era5_data, gmcp_data):
        # 同时对ERA5和GMCP应用相同的空间变换
        seed = np.random.randint(0, 2**32)
        
        for transform in self.transforms:
            random.seed(seed)
            era5_data = transform(era5_data)
            random.seed(seed)
            gmcp_data = transform(gmcp_data)
        
        return era5_data, gmcp_data
```

#### 时间增强
```python
class TemporalAugmentation:
    def __init__(self):
        self.transforms = [
            RandomTimeShift(max_shift=2),  # 最大2个时间步
            TemporalDropout(p=0.1, max_consecutive=2),
            TimeReverse(p=0.2)
        ]
    
    def __call__(self, time_series):
        for transform in self.transforms:
            time_series = transform(time_series)
        return time_series
```

#### 物理一致性增强
```python
class PhysicsConsistentAugmentation:
    def __init__(self):
        self.physics_constraints = PhysicsConstraints()
    
    def augment_with_physics(self, era5_data, gmcp_data):
        """确保增强后的数据保持物理一致性"""
        
        # 1. 水汽守恒增强
        if random.random() < 0.3:
            era5_data, gmcp_data = self.apply_moisture_conservation(
                era5_data, gmcp_data
            )
        
        # 2. 动量守恒增强
        if random.random() < 0.3:
            era5_data = self.apply_momentum_conservation(era5_data)
        
        # 3. 能量守恒增强
        if random.random() < 0.3:
            era5_data = self.apply_energy_conservation(era5_data)
        
        return era5_data, gmcp_data
```

---

## 五、评估方案

### 5.1 基线对比

#### 对比方法
1. **GenCast**: Google DeepMind的扩散模型气象预报
2. **Pangu-Weather**: 华为盘古气象大模型
3. **FourCastNet**: NVIDIA的FourCastNet
4. **ClimaX**: 基于Transformer的气候预测
5. **传统数值模式**: ECMWF IFS, GFS

#### 评估数据集
```python
evaluation_datasets = {
    'test_set': {
        'period': '2020-2024',
        'region': 'China',
        'resolution': '0.1°',
        'variables': ['precipitation'],
        'split': 'temporal'  # 时间分割，避免数据泄露
    },
    'extreme_events': {
        'period': '2000-2024',
        'threshold': '>25mm/6h',
        'events': ['typhoons', 'meiyu', 'summer_monsoon'],
        'metrics': ['csi', 'pod', 'far', 'volume_bias']
    },
    'cross_region': {
        'regions': ['south_china', 'yangtze', 'north_china', 'northeast'],
        'purpose': '评估空间泛化能力'
    }
}
```

#### 评估指标详细设计
```python
class ComprehensiveEvaluation:
    def __init__(self):
        self.metrics = {
            'basic': ['rmse', 'mae', 'corr', 'bias'],
            'categorical': ['csi', 'pod', 'far', 'hss', 'ets'],
            'intensity': ['quantile_score', 'crps', 'ks_test'],
            'extreme': ['extreme_csi', 'extreme_pod', 'volume_rmse'],
            'spatial': ['ssim', 'psnr', 'fss'],
            'temporal': ['temporal_correlation', 'event_duration_bias']
        }
    
    def generate_evaluation_report(self, predictions, targets):
        """生成详细评估报告"""
        report = {}
        
        for metric_group, metric_list in self.metrics.items():
            report[metric_group] = {}
            for metric in metric_list:
                report[metric_group][metric] = self.compute_metric(
                    metric, predictions, targets
                )
        
        return report
```

### 5.2 消融实验

#### 消融实验设计
```python
ablation_experiments = {
    'architecture': {
        'exp1': {'name': 'Baseline', 'components': ['encoder', 'decoder']},
        'exp2': {'name': '+MultiScale', 'components': ['encoder', 'multiscale', 'decoder']},
        'exp3': {'name': '+Physics', 'components': ['encoder', 'physics', 'decoder']},
        'exp4': {'name': '+Extreme', 'components': ['encoder', 'extreme', 'decoder']},
        'exp5': {'name': 'Full Model', 'components': ['all']}
    },
    'training': {
        'exp1': {'name': 'No Pretraining', 'pretrain': False},
        'exp2': {'name': 'ERA5 Only', 'pretrain_data': 'era5'},
        'exp3': {'name': 'GMCP Only', 'pretrain_data': 'gmcp'},
        'exp4': {'name': 'Multi-source', 'pretrain_data': 'both'}
    },
    'loss': {
        'exp1': {'name': 'MSE Only', 'loss': 'mse'},
        'exp2': {'name': '+CSI', 'loss': 'mse+csi'},
        'exp3': {'name': '+Extreme', 'loss': 'mse+csi+extreme'},
        'exp4': {'name': '+Physics', 'loss': 'mse+csi+extreme+physics'}
    }
}
```

#### 消融实验评估
```python
class AblationStudy:
    def __init__(self, base_config):
        self.base_config = base_config
    
    def run_ablation(self, experiment_name, experiment_config):
        """运行单个消融实验"""
        config = self.merge_config(self.base_config, experiment_config)
        
        # 训练模型
        model = self.train_model(config)
        
        # 评估模型
        results = self.evaluate_model(model, self.test_loader)
        
        return {
            'experiment': experiment_name,
            'config': experiment_config,
            'results': results
        }
    
    def analyze_results(self, all_results):
        """分析消融实验结果"""
        analysis = {
            'component_importance': {},
            'synergy_effects': {},
            'recommendations': []
        }
        
        # 计算各组件的贡献
        baseline_results = all_results['Baseline']['results']
        for exp_name, exp_results in all_results.items():
            if exp_name != 'Baseline':
                improvement = self.compute_improvement(
                    baseline_results, exp_results['results']
                )
                analysis['component_importance'][exp_name] = improvement
        
        return analysis
```

---

## 六、创新点总结

### 1. 物理约束扩散模型 (Physics-Constrained Diffusion)
- **创新性**: 将大气动力学方程作为扩散过程的物理引导
- **优势**: 确保预测结果符合物理规律，提升极端事件预测的物理一致性
- **实现**: 可学习的约束权重自适应调整不同物理规律的重要性

### 2. 0.1°高分辨率直接训练 (Direct High-Resolution Training)
- **创新性**: 突破传统降尺度限制，直接学习0.1°分辨率的降水分布
- **优势**: 避免降尺度过程中的信息损失，提升空间细节的预测精度
- **实现**: 跨分辨率融合模块显式建模ERA5和GMCP的分辨率差异

### 3. 极端事件感知机制 (Extreme Event Aware Mechanism)
- **创新性**: 专门的分支和损失函数处理极端降水事件
- **优势**: 显式提升极端事件的预测能力，避免模型对极端事件的低估
- **实现**: 多级极端事件检测 + 极端事件专用编码器 + 组合损失函数

### 4. 多尺度时空异质性建模 (Multi-Scale Spatiotemporal Heterogeneity)
- **创新性**: 自适应捕捉不同尺度的降水特征和时空非平稳性
- **优势**: 更好地建模中国复杂地形和气候条件下的降水特性
- **实现**: 多尺度卷积 + 时空注意力 + 区域特定编码器

### 5. ERA5-GMCP联合预训练 (Joint Pre-training Strategy)
- **创新性**: 充分利用ERA5的物理一致性和GMCP的高精度降水观测
- **优势**: 学习更鲁棒的多源数据融合表征
- **实现**: 三阶段渐进式融合 + 跨分辨率注意力机制

---

## 七、实现计划

### 阶段1: 环境搭建与数据准备 (4周)

| 任务 | 时长 | 交付物 |
|------|------|--------|
| GPU集群环境配置 | 1周 | 可用的训练环境 |
| ERA5数据下载与预处理 | 2周 | 标准化ERA5数据集 |
| GMCP数据下载与预处理 | 1周 | 标准化GMCP数据集 |
| 数据加载器实现 | 1周 | 高效数据加载pipeline |

**关键里程碑**: 数据准备完成，可以通过DataLoader高效加载

### 阶段2: 模型架构实现 (6周)

| 任务 | 时长 | 交付物 |
|------|------|--------|
| 多尺度编码器实现 | 2周 | Module A |
| 物理约束扩散模块实现 | 2周 | Module B |
| 极端事件感知分支实现 | 1周 | Module C |
| 时空异质性建模实现 | 1周 | Module D |
| 整体架构集成 | 1周 | 完整模型 |

**关键里程碑**: 模型可以通过前向传播，参数量<500M

### 阶段3: 训练策略实现 (4周)

| 任务 | 时长 | 交付物 |
|------|------|--------|
| 损失函数实现 | 1周 | 多任务损失函数 |
| 优化器与调度器实现 | 1周 | 训练策略 |
| 数据增强实现 | 1周 | 增强pipeline |
| 分布式训练实现 | 1周 | 多GPU训练支持 |

**关键里程碑**: 可以在单GPU上完成小规模训练

### 阶段4: 预训练与微调 (8周)

| 任务 | 时长 | 交付物 |
|------|------|--------|
| 阶段1预训练 | 2周 | 预训练权重 |
| 阶段2多源融合预训练 | 3周 | 融合权重 |
| 阶段3下游任务微调 | 2周 | 微调权重 |
| 阶段4极端事件增强 | 1周 | 最终权重 |

**关键里程碑**: 模型在验证集上达到目标指标的80%

### 阶段5: 评估与优化 (4周)

| 任务 | 时长 | 交付物 |
|------|------|--------|
| 基线对比实验 | 1周 | 评估报告 |
| 消融实验 | 2周 | 消融分析 |
| 超参数调优 | 1周 | 优化后模型 |

**关键里程碑**: 所有指标达到目标值

### 总时间线
```
总时长: 26周 (约6个月)
- 阶段1: 4周
- 阶段2: 6周  
- 阶段3: 4周
- 阶段4: 8周
- 阶段5: 4周
```

### 资源需求

#### 计算资源
- **训练**: 8×H100 GPU (80GB) 或 16×A100 GPU (80GB)
- **推理**: 1×H100 或 2×A100
- **存储**: 10TB (数据+模型+实验记录)
- **内存**: 512GB RAM

#### 人力需求
- **算法研究员**: 2人 (模型设计与优化)
- **工程师**: 2人 (系统实现与优化)
- **数据工程师**: 1人 (数据处理与pipeline)
- **总计**: 5人 × 6个月 = 30人月

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| GPU资源不足 | 中 | 高 | 申请更多资源，优化代码效率 |
| 数据质量问题 | 中 | 中 | 数据清洗，异常值处理 |
| 模型收敛困难 | 低 | 高 | 调整学习率，使用梯度裁剪 |
| 极端事件样本不足 | 高 | 中 | 数据增强，过采样策略 |
| 物理约束过强 | 中 | 中 | 可学习权重，渐进式约束 |

---

## 附录

### A. 核心代码结构

```
weather/
├── models/
│   ├── __init__.py
│   ├── phydiff_net.py          # 主模型
│   ├── modules/
│   │   ├── encoder.py          # 多尺度编码器
│   │   ├── diffusion.py        # 物理约束扩散
│   │   ├── extreme_branch.py   # 极端事件分支
│   │   └── heterogeneity.py    # 时空异质性
│   └── losses/
│       ├── __init__.py
│       ├── precipitation_loss.py
│       ├── csi_loss.py
│       └── physics_loss.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── augmentation.py
│   └── preprocessing.py
├── trainers/
│   ├── __init__.py
│   ├── trainer.py
│   ├── pretrainer.py
│   └── finetuner.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
├── configs/
│   ├── default.yaml
│   ├── pretrain.yaml
│   └── finetune.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
└── tests/
    ├── test_model.py
    └── test_data.py
```

### B. 关键超参数配置

```yaml
# configs/default.yaml
model:
  hidden_dim: 256
  num_scales: 3
  num_diffusion_steps: 1000
  dropout_rate: 0.1
  
training:
  batch_size: 16
  learning_rate: 1e-4
  weight_decay: 0.01
  max_epochs: 400
  warmup_epochs: 10
  
data:
  input_resolution: 0.1  # degrees
  temporal_resolution: 6  # hours
  input_sequence_length: 12  # time steps
  forecast_horizon: 4  # time steps (24 hours)
  
augmentation:
  spatial_flip: true
  spatial_rotation: true
  temporal_shift: true
  physics_consistent: true
  
loss:
  mse_weight: 0.2
  csi_weight: 0.25
  extreme_weight: 0.1
  physics_weight: 0.1
```

---

**文档版本**: v1.0  
**创建日期**: 2026-06-15  
**作者**: algorithm-architect  
**状态**: 设计完成，待实现
