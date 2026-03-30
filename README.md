# Keqing1 — 立直麻将 AI

基于监督学习的立直麻将策略机器人，使用 Conv1d ResNet 架构，支持 Mjai 协议推理。

## 项目状态

当前主力模型为 **keqingv2**（Conv1d ResNet ~1.7M 参数，keqingv1 + Meld Value Ranking Loss）。

主要特性：
- 完整的 45 动作监督学习框架（dahai×34 + reach/chi/pon/kan/hora/none）
- Meld Value Ranking Loss：value head 学习区分值得副露 vs 不值得副露
- 立直状态三段式 legal_actions（`reached` / `pending_reach` / 普通）
- none/pass 样本自动生成（约占总样本 16%）
- 振听系统、荣和精确判断
- 立直后摸切样本过滤（约 5.5%）
- KeqingBot 支持 Mjai 协议推理 + beam search value 重排

## 快速开始

### 安装

```bash
uv sync
```

### 预处理数据

```bash
# 普通数据（ds1-ds6, ds8-ds13 合并到 processed_v2_new/ds1）
uv run python scripts/preprocess_v2.py \
  --data_dirs artifacts/converted_mjai/ds1 artifacts/converted_mjai/ds2 ... \
  --output_dir processed_v2_new/ds1 --workers 16

# NAGA 视角数据（ds7 → processed_v2_new/ds2）
uv run python scripts/preprocess_v2.py \
  --data_dirs artifacts/converted_mjai/ds7 \
  --output_dir processed_v2_new/ds2 --workers 16 \
  --actor_name_filter 'ⓝNAGA25'
```

### 启动训练（keqingv2）

```bash
# 从 keqingv1 权重迁移（推荐）
uv run python src/train/train_v2.py \
  --config configs/keqingv2_default.yaml \
  --resume artifacts/models/keqingv1/best.pth --weights-only

# 断点续训
uv run python src/train/train_v2.py \
  --config configs/keqingv2_default.yaml \
  --resume artifacts/models/keqingv2/latest.pth
```

### 运行测试

```bash
uv run pytest
```

## 项目架构

```text
keqing1/
├── src/
│   ├── keqingv1/                 # v1 基础模型（复用）
│   │   ├── action_space.py       # 45 个动作的索引映射
│   │   ├── features.py           # 特征编码 (54,34) tile_feat + (48,) scalar
│   │   ├── model.py              # MahjongModel (~1.7M 参数)
│   │   ├── cached_dataset.py     # IterableDataset 流式加载（npz）
│   │   ├── trainer.py            # 训练循环（AMP + 梯度累积）
│   │   └── bot.py                # KeqingBot（Mjai 协议推理 + beam search）
│   │
│   ├── keqingv2/                 # v2 扩展（Meld Value Ranking Loss）
│   │   ├── cached_dataset.py     # 预编码 rank pair，batch 返回 10-tuple
│   │   └── trainer.py            # rank loss 训练循环
│   │
│   ├── mahjong_env/              # 麻将环境（游戏状态、合法动作）
│   │   ├── state.py
│   │   ├── legal_actions.py
│   │   ├── replay.py
│   │   ├── tiles.py
│   │   └── types.py
│   │
│   ├── train/
│   │   └── train_v2.py           # keqingv2 训练入口
│   │
│   └── gateway/                  # Bot 对战网关
│
├── configs/
│   └── keqingv2_default.yaml     # keqingv2 训练超参
│
├── artifacts/
│   ├── converted_mjai/           # 转换后的训练数据（ds1-ds13）
│   └── models/modelv5/           # 训练输出
│       ├── best.pth              # 最佳 checkpoint
│       ├── latest.pth            # 最新 checkpoint
│       └── train_log.jsonl       # 训练日志
│
├── tests/
└── docs/
    └── v5model.md                # v5model 详细设计文档
```

## v5model 架构

### 网络结构

```text
输入：
  tile_feat  (B, 128, 34)  — 128 通道 × 34 张牌
  scalar     (B, 16)       — 16 维标量特征

MahjongModel：
  input_proj  Conv1d(128→256, k=1) + BN + ReLU
  ResBlock×4  Conv1d(256, k=3, pad=1) + BN + ReLU  ×2（残差连接）
  global_avg_pool  →  (B, 256)
  scalar_proj  Linear(16→32) + ReLU
  concat  →  (B, 288)
  fc_shared  Linear(288→256) + ReLU + Dropout(0.1)

输出头：
  policy_head  Linear(256→45)     — 未 softmax 的动作 logits
  value_head   Linear(256→64→1)   — Tanh 输出，预测最终得分

参数量：~1.7M
```

### 动作空间（45 个）

| 索引 | 动作 |
|------|------|
| 0–33 | 弃牌（dahai），按牌种索引 |
| 34 | 立直（reach） |
| 35 | 吃（chi_low） |
| 36 | 吃（chi_mid） |
| 37 | 吃（chi_high） |
| 38 | 碰（pon） |
| 39 | 大明杠（daiminkan） |
| 40 | 暗杠（ankan） |
| 41 | 加杠（kakan） |
| 42 | 和牌（hora） |
| 43 | 流局（ryukyoku） |
| 44 | 跳过（none/pass） |

### 特征编码

**tile_feat `(128, 34)`：**

- 自家手牌（多热，每种牌 0-4 张）
- 各家已见牌、副露、立直状态
- 宝牌、场风/自风、向听数等

**scalar `(16,)`：**

- 各家分数、本场数、供托数、剩余牌数、巡目等

### 训练配置

| 参数 | 值 |
|------|----|
| batch_size | 64（梯度累积×4，等效 256）|
| learning_rate | 3e-4 |
| warmup_steps | 500 |
| LR decay | Cosine |
| AMP | 是（GradScaler） |
| value loss weight | 0.5 |
| num_epochs | 20 |

### Value Target

每局游戏中累加所有 `hora` / `ryukyoku` 事件的 `deltas`，得到该玩家的最终得分变化，除以 30000 归一化到 `[-1, 1]`。

## 数据流

```text
天凤/雀魂对局记录
      ↓ (src/convert/ + libriichi)
Mjai JSONL (.mjson)
      ↓ (mahjong_env.replay.build_supervised_samples)
监督学习样本 (state, legal_actions, label_action)
      ↓ (v5model.dataset.MjaiIterableDataset)
流式训练 batch
```

数据目录：`artifacts/converted_mjai/ds1` ~ `ds13`，共 12870 个 `.mjson` 文件。

## v5model 关键实现细节

### 立直状态处理（legal_actions.py）

`enumerate_legal_actions` 对立直状态实现三段式处理：

| 状态 | `reached=True` | `pending_reach=True` | 普通 |
|------|---------------|---------------------|------|
| 含义 | 立直已被接受 | 已宣告、待打宣言牌 | 正常打牌 |
| 合法动作 | tsumogiri + ankan | 打出后 shanten==0 的打法 | 全打法 + reach + ankan + kakan |
| reach 可选 | 否 | 否 | 是（shanten==0 时）|

`pending_reach` 字段由 `state.snapshot()` 自动提供，无需外部注入。立直宣言牌候选用 `mahjong.Shanten` 逐张判断（移除后 shanten==0）。

### 训练样本过滤

`build_supervised_samples` 过滤 `reached=True` 时的 dahai 事件（立直后摸切，无决策价值，约占 dahai 样本 5.5%）。训练数据实时从 `.mjson` 解析，无需重新生成数据。

### waits_tiles 语义

`snap["waits_tiles"]`（length-34 bool list）表示**等待牌**（摸到哪张能和），不是「可打牌」。`pending_reach` 分支中判断「打出后听牌」需重新用 Shanten 计算，不能直接用 waits_tiles 过滤。

### none/pass 样本

每个 dahai 事件后，对有鸣牌机会但主动放弃（下一事件为 tsumo）的玩家生成 none 样本，约占总样本 16%。缺少此类样本会导致模型鸣牌过激进。

## 注意事项

- `src/` 为包根路径（`pyproject.toml` 配置），导入用 `from v5model.xxx import ...`
- 训练 checkpoint 保存在 `artifacts/models/modelv5/`，`best.pth` 为最佳，`latest.pth` 支持断点续训
- GPU 不可用时自动回退到 CPU
- 数据集使用 IterableDataset 流式加载，不会全量加载进内存
- Python 执行用 `uv run python3`，系统 `python3` 无 `riichienv`/`torch`

## 致谢

- [Mortal](https://github.com/Equim-chan/Mortal) — 架构参考
- [Mjai](https://github.com/mjai-jp/mjai) — 麻将 AI 协议
- 天凤/雀魂社区 — 数据集来源
