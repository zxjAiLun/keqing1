# ModelV4 与 Mortal 模型设计对比分析

## 1. 概述

| 特性 | ModelV4 | Mortal |
|------|---------|--------|
| 架构类型 | ResNet + ChannelAttention + Policy/Value Head | ResNet + DQN (Deep Q-Network) |
| 训练方式 | 监督学习 (Supervised Learning) | 深度强化学习 (Deep RL) + CQL |
| 输入特征 | 37牌类型平面 + 24维标量特征 | 多通道卷积特征 (版本相关) |
| 输出 | Policy Logits + Value | Q-values for all actions |
| 典型参数 | hidden_dim=256, num_res_blocks=3 | conv_channels=192, num_blocks=40 |

---

## 2. 模型架构对比

### 2.1 Encoder/Backbone

#### ModelV4 Encoder
```
输入:
  - tile_features: (batch, 37, tile_plane_dim=32)
  - scalar_features: (batch, scalar_dim=24)

流程:
  1. tile_proj: Conv2d(32, 256, kernel_size=(37, 1))
  2. ResBlocks: 3个 SEBlock (ResidualBlock + ChannelAttention)
  3. scalar_net: Linear(24) -> Linear(256) -> Linear(256)
  4. fusion: Linear(512) -> Linear(256)
```

#### Mortal Brain (Encoder)
```
输入:
  - obs: (batch, in_channels, 34) - 版本相关
  - invisible_obs: (可选) 额外通道

流程:
  1. Conv1d(in_channels, conv_channels=192, kernel_size=3)
  2. 40个 ResBlock (conv_channels=192)
  3. Conv1d(192, 32, kernel_size=3)
  4. Flatten -> Linear(32*34, 1024)
```

**关键差异:**
- ModelV4 使用 2D 卷积处理牌类型平面 (37x32)，Mortal 使用 1D 卷积
- Mortal 的 ResBlock 数量 (40) 远超 ModelV4 (3)
- Mortal 的卷积通道数 (192) 比 ModelV4 (256) 更小，但层数更深

### 2.2 通道注意力机制

#### ModelV4 ChannelAttention (SE-Net style)
```python
# Squeeze: Global Average Pooling
y = self.gap(x).view(batch, channels)

# Excitation: FC -> ReLU -> FC -> Sigmoid
y = self.fc(y).view(batch, channels, 1, 1)

# Scale
return x * y.expand_as(x)
```
- 使用全局平均池化
- 压缩比: 8 (channels // 8)

#### Mortal ChannelAttention (Mixed style)
```python
# 混合池化: 平均 + 最大
avg_out = self.shared_mlp(x.mean(-1))
max_out = self.shared_mlp(x.amax(-1))
weight = (avg_out + max_out).sigmoid()
x = weight.unsqueeze(-1) * x
```
- 同时使用平均池化和最大池化
- 压缩比: 16 (ratio=16)

### 2.3 Policy/Value Head vs DQN

#### ModelV4
```python
# Policy Head (3层MLP)
PolicyHead: Linear(256) -> ReLU -> Linear(128) -> ReLU -> Linear(num_actions)

# Value Head (3层MLP + Tanh)
ValueHead: Linear(256) -> ReLU -> Linear(64) -> ReLU -> Linear(1) -> Tanh()
```

#### Mortal DQN
```python
# V4版本: 单一线性层
self.net = nn.Linear(1024, 1 + ACTION_SPACE)

# Q值计算使用 "Mean Q" 公式:
# Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
a_sum = a.masked_fill(~mask, 0.).sum(-1, keepdim=True)
mask_sum = mask.sum(-1, keepdim=True)
a_mean = a_sum / mask_sum
q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
```

---

## 3. 特征工程对比

### 3.1 ModelV4 特征

#### 牌类型平面 (37, 32)
| 通道 | 特征 | 编码方式 |
|------|------|----------|
| 0 | 手牌 | 1 if 持有该牌 else 0 |
| 1 | 宝牌指示器 | 1 if 是宝牌 else 0 |
| 2-5 | 各家舍牌 | 归一化计数 (count/max_count) |

#### 标量特征 (24维)
| 索引 | 特征 | 归一化方式 |
|------|------|------------|
| 0 | 亲家标志 | 1 if 亲 else 0 |
| 1 | 场风 | wind/3 |
| 2 | 本场 | min(honba/10, 1) |
| 3 | 立直棒 | min(kyotaku/5, 1) |
| 4-7 | 各家副露次数 | min(count/4, 1) |
| 8-11 | 各家巡目 | min(n_discards/20, 1) |
| 12 | 分数差 | min(max_diff/50000, 1) |
| 13 | 向听数 | shanten/7 |
| 14 | 是否听牌 | 1 if tenpai else 0 |
| 15 | 有效进张数 | n_waits/34 |
| 16 | 预计胡牌点 | min(agari/30000, 1) |
| 17 | 胡牌概率 | 启发式计算 |
| 18 | 有役/无役 | 启发式判断 |
| 19 | 副露倾向 | 启发式计算 |

### 3.2 Mortal 特征

Mortal 使用 libriichi 库生成的观测空间，特征更加丰富：
- 34xC 的 1D 卷积输入
- 包含对手的隐式信息 (invisible_obs)
- 版本 4 可能有更精细的特征设计

---

## 4. 训练策略对比

### 4.1 ModelV4 训练

```yaml
# 监督学习配置
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 256
num_epochs: 10
grad_clip: 1.0
train_split: 0.9

# 优化器
optimizer: AdamW
scheduler: CosineAnnealingLR (T_max=num_epochs, eta_min=1e-6)

# Loss
policy_loss = CrossEntropy(policy_logits, actions)
```

### 4.2 Mortal 训练

```toml
# 强化学习配置
[optim]
eps = 1e-8
betas = [0.9, 0.999]
weight_decay = 0.1
max_grad_norm = 0  # 无梯度裁剪

[optim.scheduler]
peak = 1e-4
final = 1e-4
warm_up_steps = 0
max_steps = 0

[resnet]
conv_channels = 192
num_blocks = 40

# Loss = DQN Loss + CQL Loss + Aux Loss
dqn_loss = 0.5 * MSE(q, q_target_mc)
cql_loss = logsumexp(q_all) - mean(q_selected)
next_rank_loss = CrossEntropy(aux_logits, player_ranks)
```

---

## 5. 关键参数对训练的影响

### 5.1 隐藏维度 (hidden_dim / conv_channels)

| 参数 | ModelV4 (256) | Mortal (192) | 影响 |
|------|---------------|--------------|------|
| 表达能力 | 中等 | 较弱 | 更小的维度需要更多层来补偿 |
| 内存占用 | 较高 | 较低 | 约 1.78x 差异 |
| 训练稳定性 | 较好 | 需更多层 | 大维度更稳定 |

### 5.2 残差块数量 (num_res_blocks)

| 模型 | 层数 | 影响 |
|------|------|------|
| ModelV4 | 3 | 浅层网络，训练快，但表达能力有限 |
| Mortal | 40 | 深层网络，能捕捉复杂模式，但训练慢 |

**过拟合风险:**
- 层数过多 → 过拟合风险增加
- 层数过少 → 欠拟合，特征提取不充分

### 5.3 Dropout

ModelV4: `dropout=0.1`
- 防止过拟合
- 在残差块内使用

Mortal: 未明确使用 dropout
- 依赖 BatchNorm 和权重衰减
- 可能通过 CQL 正则化

### 5.4 学习率

| 模型 | 学习率 | Scheduler |
|------|--------|-----------|
| ModelV4 | 1e-3 | CosineAnnealing (min=1e-6) |
| Mortal | 峰值 1e-4 | LinearWarmUp + CosineAnnealing |

**影响:**
- 学习率过大 → 训练不稳定
- 学习率过小 → 收敛慢
- Warmup 有助于稳定训练早期

### 5.5 权重衰减 (weight_decay)

| 模型 | weight_decay | 作用 |
|------|--------------|------|
| ModelV4 | 1e-4 | 轻微 L2 正则化 |
| Mortal | 0.1 | 强正则化 |

Mortal 的高权重_decay 配合其深层网络结构来防止过拟合。

---

## 6. 优化建议

### 6.1 ModelV4 改进方向

1. **增加残差块数量**
   ```
   当前: num_res_blocks=3
   建议: 6-12
   ```

2. **调整隐藏维度**
   ```
   当前: hidden_dim=256
   建议: 192-256 (参考 Mortal)
   ```

3. **增强特征工程**
   - 加入更多游戏状态特征
   - 考虑使用启发式特征 (如 ModelV4 的标量特征)

4. **学习率调整**
   ```
   当前: 1e-3
   建议: 使用 warmup: 1e-5 -> 1e-3 -> 1e-5
   ```

5. **引入 CQL 或其他正则化**
   - 当前仅使用 CrossEntropy
   - 可考虑加入对比学习或策略正则化

### 6.2 训练稳定性

1. **梯度裁剪**: ModelV4 的 `grad_clip=1.0` 有助于稳定训练
2. **BatchNorm momentum**: Mortal 使用 `momentum=0.01, eps=1e-3`，更保守
3. **EMA**: Mortal 可使用指数移动平均

---

## 7. 总结

| 方面 | ModelV4 优势 | Mortal 优势 |
|------|-------------|-------------|
| 训练效率 | 高 (浅层网络) | 低 (深层网络) |
| 特征设计 | 直观、可解释 | 端到端学习 |
| 训练方式 | 简单 (SL) | 复杂 (RL+CQL) |
| 最终性能 | 基准水平 | 世界领先水平 |
| 调参难度 | 低 | 高 |

**建议**: 如果资源有限，从 ModelV4 开始进行快速迭代；如果追求最高性能，建议深入研究 Mortal 的设计，并考虑迁移其训练策略 (如 CQL、Warmup Cosine Annealing)。
