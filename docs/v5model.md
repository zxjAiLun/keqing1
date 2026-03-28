# v5model 详细设计文档

## 概述

v5model 是 keqing1 项目的当前主力模型，基于 Conv1d ResNet 架构，针对立直麻将监督学习设计。

**相对 v4model 的改进：**

| 项目 | v4 | v5 |
|------|----|----|`
| 卷积方向 | Conv2d（设计有误） | Conv1d（沿 34 牌维度） |
| 动作数 | 38（不完整） | 45（完整日麻动作空间） |
| 数据加载 | 全量进内存 | IterableDataset 流式 |
| AMP | 无 | GradScaler + autocast |
| 梯度累积 | 无 | accumulation_steps=4 |
| 断点续训 | 无 | 完整 checkpoint |
| riichienv 依赖 | 有 | 无（直接用 mahjong_env） |

---

## 文件结构

```
src/v5model/
├── __init__.py          # 公开导出
├── action_space.py      # 45 动作索引映射、ChiType 判断
├── features.py          # 特征编码（state → tile_feat + scalar）
├── model.py             # MahjongModel 网络定义
├── dataset.py           # MjaiIterableDataset 流式数据集
├── trainer.py           # 训练循环
└── bot.py               # V5Bot 推理接口

src/train/train_v5.py    # 训练入口脚本
configs/v5_default.yaml  # 默认超参配置
```

---

## 动作空间（action_space.py）

共 45 个动作，索引如下：

```python
DAHAI_OFFSET  = 0   # 0–33: 弃牌（按 tile_id/34 的牌种）
REACH_IDX     = 34  # 立直
CHI_LOW_IDX   = 35  # 吃（顺子中被吃牌在最低位）
CHI_MID_IDX   = 36  # 吃（被吃牌在中间）
CHI_HIGH_IDX  = 37  # 吃（被吃牌在最高位）
PON_IDX       = 38  # 碰
DAIMINKAN_IDX = 39  # 大明杠
ANKAN_IDX     = 40  # 暗杠
KAKAN_IDX     = 41  # 加杠
HORA_IDX      = 42  # 和牌（荣和/自摸合并）
RYUKYOKU_IDX  = 43  # 流局（九种九牌）
NONE_IDX      = 44  # 跳过/pass
```

### ChiType 判断（对齐 Mortal）

判断方法：将 `pai` 和 `consumed` 均 deaka（去赤，`tile_id % 34`），比较 `pai_rank` 与 `consumed` 的 min/max：

```
pai_rank < consumed_min  →  CHI_LOW  (35)
consumed_min ≤ pai_rank < consumed_max  →  CHI_MID  (36)
pai_rank ≥ consumed_max  →  CHI_HIGH  (37)
```

---

## 特征编码（features.py）

### 输出

- `tile_feat`：`ndarray (128, 34)`，float32
- `scalar`：`ndarray (16,)`，float32

### tile_feat 通道含义（C=128）

主要通道（其余补零）：

| 通道范围 | 含义 |
|----------|------|
| 0–3 | 自家手牌每种数量（0/1/2/3/4 张，多热编码）|
| 4–7 | 各家是否出现该牌（舍牌记录）|
| 8–11 | 各家副露是否含该牌 |
| 12–16 | dora（最多 5 张）|
| 17–20 | 各家立直状态（广播到 34 维）|
| 21 | 宝牌指示牌 |
| 22–29 | 场风/自风（one-hot，广播）|
| 30 | 向听数（归一化，广播）|
| 31 | 是否听牌（广播）|

### scalar 含义（S=16）

| 索引 | 含义 |
|------|------|
| 0–3 | 各家分数（除以 100000 归一化）|
| 4 | 本场数（除以 8）|
| 5 | 供托数（除以 8）|
| 6 | 剩余牌数（除以 70）|
| 7–10 | 各家巡目（除以 18）|
| 11 | 场风（0=东, 1=南）|
| 12 | 自风（0-3）|
| 13 | 向听数（除以 8）|
| 14 | 有效进张数（除以 34）|
| 15 | 是否听牌（0/1）|

---

## 网络结构（model.py）

```
MahjongModel
  input_proj:   Conv1d(128 → 256, k=1) + BN1d + ReLU
  ResBlock × 4: Conv1d(256, k=3, pad=1) + BN1d + ReLU
                Conv1d(256, k=3, pad=1) + BN1d
                + 残差连接 + ReLU
  global_avg_pool(dim=-1)  →  (B, 256)
  scalar_proj:  Linear(16 → 32) + ReLU
  concat:       (B, 256+32) = (B, 288)
  fc_shared:    Linear(288 → 256) + ReLU + Dropout(0.1)
  policy_head:  Linear(256 → 45)        — 未 softmax
  value_head:   Linear(256 → 64) + ReLU
                Linear(64 → 1) + Tanh   — 预测最终相对得分
```

总参数量约 **1.7M**，BN momentum=0.01, eps=1e-3。
权重初始化：Conv1d 用 Kaiming normal，Linear 用 Xavier uniform。

---

## 数据集（dataset.py）

### MjaiIterableDataset

- 继承 `torch.utils.data.IterableDataset`，流式读取 `.mjson` 文件
- 每个样本：`(tile_feat, scalar, legal_mask, action_idx, value)`
- shuffle buffer = 2000，每次随机 pop 减少顺序偏差
- 多 epoch 自动重 shuffle 文件列表

### Value Target

```python
# 扫描全局 hora/ryukyoku 事件的 deltas，累加后归一化
value_target = final_deltas[actor] / 30000.0
value_target = clip(value_target, -1.0, 1.0)
```

### split_files

```python
train_files, val_files = split_files(root_dirs, val_ratio=0.05, seed=42)
```

---

## 训练循环（trainer.py）

### 损失函数

```
total_loss = masked_CE(policy_logits, action_idx, legal_mask)
           + 0.5 × MSE(value, value_target)
```

`masked_CE`：将非法动作的 logits 设为 -1e9 后再算 cross entropy。

### 训练特性

- **AMP**：`torch.cuda.amp.autocast` + `GradScaler`
- **梯度累积**：默认 `accumulation_steps=4`，等效 batch=256
- **LR schedule**：线性 warmup（500步）+ cosine decay
- **断点续训**：checkpoint 保存 model / optimizer / scheduler / epoch / step

### Checkpoint 文件

| 文件 | 含义 |
|------|------|
| `best.pth` | 验证集 CE loss 最低的 checkpoint |
| `latest.pth` | 最新 epoch 结束时的 checkpoint |
| `train_log.jsonl` | 每 epoch 的训练/验证指标 |

---

## 推理 Bot（bot.py）

```python
from v5model.bot import V5Bot

bot = V5Bot(player_id=0, model_path="artifacts/models/modelv5/best.pth", device="cuda")

# 逐事件处理（Mjai 协议）
action = bot.react(event_dict)  # 需要响应时返回动作 dict，否则返回 None
```

### 推理流程

1. 接收 mjai 事件，通过 `apply_event` 更新 `GameState`
2. 遇到自家 `tsumo` 或他家 `dahai` 时，调用 `enumerate_legal_actions`
3. 特征编码 → 模型推理 → 非法动作 mask → argmax 选最优合法动作

---

## 启动训练

```bash
# 基本用法
uv run python src/train/train_v5.py \
  --data_dirs artifacts/converted_mjai/ds1 artifacts/converted_mjai/ds2 artifacts/converted_mjai/ds3 \
  --output_dir artifacts/models/modelv5

# 使用配置文件
uv run python src/train/train_v5.py --config configs/v5_default.yaml

# 断点续训
uv run python src/train/train_v5.py \
  --data_dirs artifacts/converted_mjai/ds1 \
  --output_dir artifacts/models/modelv5 \
  --resume

# 指定设备/seed
uv run python src/train/train_v5.py \
  --data_dirs artifacts/converted_mjai/ds1 \
  --output_dir artifacts/models/modelv5 \
  --device cuda --seed 42
```

### 监控训练

```bash
# 实时查看训练日志
tail -f artifacts/models/modelv5/train_log.jsonl

# 在 Ghostty 后台运行（& 放后台，输出重定向）
uv run python src/train/train_v5.py \
  --data_dirs artifacts/converted_mjai/ds1 \
  --output_dir artifacts/models/modelv5 \
  > artifacts/models/modelv5/stdout.log 2>&1 &

echo "PID: $!"
```
