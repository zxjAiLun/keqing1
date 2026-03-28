# v5model 详细设计文档

## 概述

v5model 是 keqing1 项目的当前主力模型，基于 Conv1d ResNet + SE Block 架构，针对立直麻将监督学习（行为克隆）设计。

**相对 v4model 的改进：**

| 项目 | v4 | v5 |
|------|----|----|`
| 卷积方向 | Conv2d（设计有误） | Conv1d（沿 34 牌维度） |
| 激活函数 | ReLU | Mish |
| Channel Attention | 无 | SE Block（reduction=8） |
| 赤宝牌特征 | 归并（丢失信息） | 独立通道 ch56-58 |
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
├── trainer.py           # 训练循环（AMP + 梯度累积）
└── bot.py               # V5Bot 推理 Bot

src/mahjong_env/
├── replay.py            # build_supervised_samples（数据标注核心）
├── state.py             # GameState + apply_event
├── legal_actions.py     # enumerate_legal_actions
└── tiles.py             # 牌名常量

tests/v5model/
└── test_features.py     # features.py 单元测试（10个用例）
```

---

## 特征编码（features.py）

### tile_feat: shape (128, 34)

| 通道 | 内容 |
|------|------|
| 0-3 | 自家手牌计数 planes（第k张同种牌） |
| 4-7 | 自家副露 presence（每组副露占一个 ch） |
| 8-11 | 他家副露 presence（上/对/下家各一组，最多4组用ch8-11均摊） |
| 12-23 | 自家弃牌历史（按出牌顺序，12张） |
| 24-35 | 他家弃牌历史（上/对/下家各4张，共12ch） |
| 36-39 | 赤宝牌标记（自家手牌/副露/弃牌/他家）—— 按牌种 34 列 |
| 40-47 | dora 指示牌（最多8张）each ch 全列=1 表示有 dora |
| 48-51 | 自家手牌是否含各 dora（逐张） |
| 52-55 | 各 dora 在已见牌中的计数 |
| 56-58 | 赤宝牌独立通道（0m/0p/0s），手牌中有则对应列=1 |
| 59-63 | 各家立直标记（全列） |
| 64-67 | 各家弃牌数量（归一化） |
| 68-71 | 各家副露数量（归一化） |
| 72-75 | 自家 keep/next shanten discard 候选 |
| 76-127 | 保留（全零） |

### scalar: shape (N_SCALAR=20)

| 索引 | 内容 | 归一化 |
|------|------|--------|
| 0 | bakaze（场风） | /3 |
| 1 | jikaze（自风） | /3 |
| 2 | kyoku（局数） | /4 |
| 3 | honba（本场） | /4 |
| 4 | kyotaku（供托） | /4 |
| 5 | 自家得分 | /40000 |
| 6 | 顺位（0=1位） | /3 |
| 7 | 向听数 | /8 |
| 8 | waits_count（等张数） | /34 |
| 9 | 是否立直 | bool |
| 10 | 副露数 | /4 |
| 11 | 弃牌数 | /24 |
| 12-14 | 他家立直（上/对/下家） | bool |
| 15 | 保留 | 0 |
| 16-19 | 风格向量 [speed, riichi, value, defense] | [-1,+1] |

---

## 副露（Meld）处理的关键细节

### 手牌状态时机

副露后手牌处于 **3n+2** 状态（例如 chi 后11张，需要再打一张变10张），**不能**直接用 HandEvaluator 计算 waits。

| 动作 | 手牌张数 | 状态 | 可算 shanten | 可算 waits |
|------|---------|------|------------|----------|
| chi 发生 | 13→11 | 3n+2 | ✓ | ✗ |
| chi 后 dahai | 11→10 | 3n+1 | ✓ | ✓ |
| 门清 dahai | 13→12 | — | ✓（打前13张） | ✓（打后12张） |

### shanten 和 waits_count 的注入来源

**训练时（`build_supervised_samples`）：**
- 使用 `riichi.state.PlayerState`（libriichi Rust 实现）维护精确状态
- `snap["shanten"] = shanten_before`：动作发生前的向听数
- `snap["waits_count"] = waits_after_cnt`：动作发生**后** PlayerState 更新的等待数
  - libriichi 只在 **dahai 后（3n+1）** 调用 `update_waits_and_furiten`，所以 `waits_after_cnt` 是打出 label 牌后的真实等待数
  - chi/pon 后（3n+2）waits 不更新，`waits_after_cnt=0` 是正确的（副露后还没打牌，等待未确定）

**推理时（`V5Bot.react`）：**
- bot.py 维护 `riichi.state.PlayerState`，每次 react 先 `update` 事件
- 在 `encode` 前注入 `snap["shanten"]` 和 `snap["waits_count"]`
- `waits_count` 来自 PlayerState.waits（dahai 后才有非零值，chi/pon 后为0）

**为什么不用 `HandEvaluator.get_waits()`：**
- HandEvaluator 只支持标准13张门清手牌
- 副露后手牌（11/10/7/4张）传入时 `is_tenpai()` 和 `get_waits()` 均返回错误结果

### Mortal/libriichi 对比

libriichi 的 `update_shanten_discards()` 在 chi/pon 后（3n+2）计算：
- `keep_shanten_discards[i]`：打第 i 张牌向听不变
- `next_shanten_discards[i]`：打第 i 张牌向听下降

`update_waits_and_furiten()` 只在 dahai 后（3n+1）调用，才更新 `state.waits`。

Mortal 的 obs_repr.rs 直接读 `state.waits`（dahai 后的值）编码特征，与我们修复后的行为一致。

---

## 动作空间（action_space.py）

45 个动作：

| 索引 | 动作 |
|------|------|
| 0-33 | dahai（弃牌，34种牌，赤归并） |
| 34 | reach（立直） |
| 35 | chi_low（被吃牌是顺子最低位） |
| 36 | chi_mid（被吃牌是顺子中间位） |
| 37 | chi_high（被吃牌是顺子最高位） |
| 38 | pon（碰） |
| 39 | daiminkan（明杠） |
| 40 | ankan（暗杠） |
| 41 | kakan（加杠） |
| 42 | hora（荣和/自摸统一） |
| 43 | ryukyoku（九种九牌） |
| 44 | none/pass |

ChiType 判断对齐 Mortal chi_type.rs：`pai_rank < lo → chi_low`，`lo ≤ pai_rank < hi → chi_mid`，`pai_rank ≥ hi → chi_high`。

---

## 训练

```bash
# 全量训练（ds1-ds13）
PYTHONPATH=src .venv/bin/python3 -m train.train_v5 --config configs/v5_default.yaml

# 断点续训
PYTHONPATH=src .venv/bin/python3 -m train.train_v5 --config configs/v5_default.yaml --resume artifacts/models/modelv5/latest.pth

# 运行测试
PYTHONPATH=src .venv/bin/python3 -m pytest -v
```

**当前训练结果（旧数据，waits_count 未修复）：**
- 最佳 val_acc=75.2%（epoch 6），之后过拟合
- acc 卡在 76% 左右，部分原因是副露后 waits_count 全为 0