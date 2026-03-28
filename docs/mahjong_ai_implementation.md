# 轻量化麻将 AI 本地训练解决方案

> 基于 v4model 改造，针对 4060 Laptop（16G VRAM + 16G RAM + i7-12700H）优化
> 目标：1v3 摸切 bot 胜率领先，可打包到 mjai 平台进行 bot 间实战模拟

---

## 一、问题诊断总结（已对照源码校正）

### 1.1 v4model 现有问题

| 问题 | 影响 | 优先级 | 源码依据 |
|---|---|---|---|
| ~~NumPy 实现，无 GPU 加速~~ **【已修正】v4model 已是 PyTorch 实现** | — | — | `src/v4model/model.py`: Conv2d/BatchNorm2d/ResidualBlock 均为 PyTorch |
| 270 维状态编码，冗余高 | 参数量大，收敛慢 | P1 | `src/bot/features.py:11`: `OBS_DIM = 270` |
| 38 个动作覆盖不完整 | 无法处理完整日麻动作 | P1 | `src/train/train_v4.py:54`: `NUM_ACTIONS = 38` |
| ~~5 层残差网络~~ **【已修正】默认 3 层** | 模型容量可按需调整 | P1 | `train_v4.py` 默认配置 `num_res_blocks: 3` |
| 无断点续训机制 | 训练中断只能重来 | P0 | `train_v4.py` 无 checkpoint save/load |
| 数据全量加载进内存 | 16GB RAM 压力大 | P0 | `train_v4.py`: `V4SupervisedDataset` 全量 list |

### 1.2 豆包建议中的问题

| 问题 | 错误描述 | 正确做法 |
|---|---|---|
| LayerNorm 用错位置 | 把 LayerNorm 用于 `(B, H, W, C)` 再 permute，语义错误 | 用 BatchNorm2d（v4model 已正确使用） |
| LazyDataset 是 O(n) 灾难 | 每次 `__getitem__` 从头扫文件 | 用预建索引文件 seek |
| 16GB RAM 装不下大数据集 | — | IterableDataset 流式读取 |
| 价值损失用最终得分做 MSE | reward 滞后，梯度噪声大 | 纯策略损失或 CQL，不做简单 MSE 回归 |
| batch_size=32 太小 | 噪声大 | 至少 64-128，梯度累积到 512+ |
| 学习率 1e-4 无 warmup | 随机初始化易震荡 | 3e-5 + warmup |

### 1.3 Mortal 的实际做法（源码校正版）

> **重要**：以下均来自 `third_party/Mortal/` 源码实测，原文档多处描述有误。

- **数据格式**：mjai 棋谱 → `(obs_channels, 34)` 二维张量，**不是** `(156, 4, 9)` 三维张量
  - v4 版本：`obs_shape(4) = (1012, 34)`，`(consts.rs:25)`
  - 空间维度是 34（日麻 34 种牌），不是 4×9
- **网络结构**：使用 `nn.Conv1d`（一维卷积），**不是** Conv2d，`(model.py:46-49)`
- **动作空间**：`ACTION_SPACE = 46`，**不是** 121，`(consts.rs:7-15)`
  - 37（弃牌/杠选择）+ 1（立直）+ 3（吃）+ 1（碰）+ 1（杠决定）+ 1（和牌）+ 1（流局）+ 1（pass）
- **训练方式**：BC（行为克隆）+ CQL（Conservative Q-Learning），**不是**纯 BC，`(train.py: min_q_weight)`
- **断点续训**：保存 `model_state_dict + optimizer_state_dict + epoch + scheduler_state_dict` ✓
- **AMP**：`torch.cuda.amp.GradScaler` + `autocast` ✓
- **梯度累积**：`accumulation_steps` ✓
- **数据懒加载**：`FileDatasetsIter(IterableDataset)` 流式读取，`(dataloader.py:10)` ✓
- **底层解析**：通过 Rust 库 `libriichi.dataset.GameplayLoader` 解析 mjai 格式，Python 端不手写解析

---

## 二、整体架构

```
v4model_new/
├── configs/
│   └── default.yaml           # 超参数配置
├── data/
│   ├── __init__.py
│   ├── preprocessor.py        # 数据预处理（JSONL → 流式）
│   └── mjai_dataset.py        # IterableDataset 加载器
├── model/
│   ├── __init__.py
│   ├── encoder.py             # ResBlock（Conv1d）+ MahjongEncoder
│   └── light_mortal.py        # 完整模型
├── train/
│   ├── __init__.py
│   ├── trainer.py             # 训练循环（AMP + 梯度累积）
│   └── checkpoint.py         # 断点续训管理
├── inference/
│   ├── __init__.py
│   └── bot.py                # mjai 协议推理 Bot
├── scripts/
│   ├── train.sh               # 启动训练
│   ├── resume.sh              # 恢复中断训练
│   └── test_model.sh         # 模型推理测试
└── requirements.txt
```

---

## 三、模型结构（对齐 Mortal 实际设计）

### 3.1 编码器（Conv1d，非 Conv2d）

```python
# model/encoder.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """残差块，输入形状 (B, channels, 34)"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + identity)

class MahjongEncoder(nn.Module):
    """
    输入: (B, obs_channels, 34) — 与 Mortal obs_shape(4)=(1012,34) 对齐
    输出: (B, hidden_dim) — 局面特征向量
    """
    def __init__(self, in_channels=1012, hidden_channels=192, num_blocks=4):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm1d(hidden_channels)
        self.blocks = nn.Sequential(*[ResBlock(hidden_channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, hidden_channels, 34) -> (B, hidden_channels, 1)

    def forward(self, x):
        x = torch.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)  # (B, hidden_channels)
        return x
```

> **说明**：如果不使用 libriichi 解析 mjai（即自行编码特征），可以灵活设计 `in_channels`。
> 关键是空间维度用 34（对应 34 种牌），用 Conv1d 沿牌种维度卷积。

### 3.2 完整模型

```python
# model/light_mortal.py
import torch
import torch.nn as nn
from .encoder import MahjongEncoder

ACTION_SPACE = 46  # 对齐 Mortal consts.rs

class LightMortal(nn.Module):
    """
    输入: obs (B, obs_channels, 34), mask (B, 46)
    输出: logits (B, 46)
    """
    def __init__(self, in_channels=1012, hidden_dim=192, num_blocks=4):
        super().__init__()
        self.encoder = MahjongEncoder(in_channels, hidden_dim, num_blocks)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_SPACE),
        )

    def forward(self, obs, mask):
        feat = self.encoder(obs)          # (B, hidden_dim)
        logits = self.policy_head(feat)   # (B, 46)
        logits = logits.masked_fill(~mask, float('-inf'))
        return logits
```

---

## 四、数据流水线

### 4.1 自定义特征编码方案（不依赖 libriichi）

若无法编译 libriichi，可自行将 mjai JSONL 转换为 `(obs_channels, 34)` 格式：

```python
# data/mjai_dataset.py
import json
from pathlib import Path
from torch.utils.data import IterableDataset
import torch
import numpy as np

class MjaiIterableDataset(IterableDataset):
    """
    流式读取 mjai JSONL 文件，不一次性加载到内存。
    每个样本: (obs, mask, action_idx)
    """
    def __init__(self, file_list, feature_encoder):
        self.file_list = file_list
        self.feature_encoder = feature_encoder  # 自定义特征编码器

    def __iter__(self):
        for path in self.file_list:
            with open(path) as f:
                events = [json.loads(line) for line in f]
            yield from self.feature_encoder.encode_game(events)
```

### 4.2 使用 libriichi（推荐，与 Mortal 完全对齐）

```python
from libriichi.dataset import GameplayLoader

loader = GameplayLoader(
    version=4,
    oracle=False,
    player_names=None,
    excludes=None,
)
for obs, mask, action, meta in loader.load_files(file_list, file_batch_size=20):
    # obs: (1012, 34), mask: (46,), action: int
    ...
```

---

## 五、训练循环（AMP + 梯度累积）

```python
# train/trainer.py
import torch
from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(model, loader, optimizer, scheduler, scaler, accumulation_steps=8):
    model.train()
    optimizer.zero_grad()
    for step, (obs, mask, action) in enumerate(loader):
        obs, mask, action = obs.cuda(), mask.cuda(), action.cuda()
        with autocast():
            logits = model(obs, mask)
            loss = torch.nn.functional.cross_entropy(logits, action) / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
```

---

## 六、断点续训

```python
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location='cuda')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['epoch']
```

---

## 七、实施优先级（校正版）

| 优先级 | 任务 | 说明 |
|---|---|---|
| P0 | IterableDataset 流式加载 | 解决内存问题，替换全量加载 |
| P0 | CheckpointManager | 断点续训 |
| P0 | AMP + 梯度累积 | 显存减半，等效大 batch |
| P1 | 观测从 270 迁移到 `(obs_channels, 34)` + Conv1d | 对齐 Mortal 实际格式 |
| P1 | 动作空间扩展到 46 | 覆盖完整日麻动作（对齐 Mortal，不是 121） |
| P2 | mjai 协议接入 + 部署 | bot 间实战评估 |
| P3 | torch.compile 编译优化 | 提速约 20% |

---

## 八、值得保留的现有代码

- `src/v4model/model.py`：ResidualBlock、ChannelAttention 可复用，改 Conv2d→Conv1d
- `src/v4model/bot.py`：mjai 协议交互逻辑、动作映射
- `src/mahjong_env/`：规则引擎，核心依赖
- `RuleBot`：baseline 对手

---

## 九、关键事实备忘（避免重蹈覆辙）

| 项目 | 错误认知 | 实际值 | 来源 |
|---|---|---|---|
| Mortal 观测张量形状 | `(156, 4, 9)` | `(1012, 34)` | `consts.rs:25` |
| Mortal 卷积类型 | Conv2d | Conv1d | `mortal/model.py` |
| Mortal 动作空间 | 121 | 46 | `consts.rs:7-15` |
| Mortal 训练方式 | 纯 BC | BC + CQL | `mortal/train.py` |
| v4model 实现框架 | NumPy | PyTorch | `src/v4model/model.py` |
| v4model 残差块数 | 5 层 | 默认 3 层 | `src/train/train_v4.py` |
