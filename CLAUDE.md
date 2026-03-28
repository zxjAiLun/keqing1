# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# 安装依赖
uv sync

# 安装额外包（如 mahjong、pytest）
uv pip install mahjong pytest

# 运行所有测试
uv run python3 -m pytest tests/v5model/

# 运行单个测试文件
uv run python3 -m pytest tests/v5model/test_features.py

# 训练（配置文件方式）
uv run python src/train/train_v5.py --config configs/v5_default.yaml

# 断点续训
uv run python src/train/train_v5.py --config configs/v5_default.yaml --resume artifacts/models/modelv5/latest.pth
```

`uv run` 是唯一正确的 Python 执行方式，系统 `python3` 没有 `riichienv`/`torch`。`mahjong` 包需单独 `uv pip install mahjong`（不在 pyproject.toml 依赖中）。

## 架构概述

项目是立直麻将监督学习 Bot（行为克隆），数据流：

```
mjai JSONL (.mjson)
  → mahjong_env.replay.build_supervised_samples()  # 标注 label + legal_actions
  → v5model.dataset.MjaiIterableDataset            # 流式加载
  → v5model.features.encode()                      # (128,34) tile_feat + (20,) scalar
  → v5model.model.MahjongModel                     # Conv1d ResNet + SE Block (~1.7M 参数)
  → policy logits (45) + value scalar
```

## 牌格式约定（关键）

项目内部统一使用以下格式，**不使用** riichienv 的 `0m/0p/0s` 格式：

| 类型 | 格式 | 示例 |
|------|------|------|
| 普通数牌 | `{n}{suit}` | `5m`, `9s` |
| 赤宝牌 | `5mr`, `5pr`, `5sr` | mjai log 原始格式 |
| 字牌 | `E/S/W/N/P/F/C` | 东南西北白发中 |

`state.py` 的 `_normalize_or_keep_aka` 保留 `5mr/5pr/5sr`，去掉其他 `r` 后缀。`0m/0p/0s` 是 riichienv 内部格式，不应出现在项目代码或测试中。

## tile 格式转换（mahjong_env/tiles.py）

牌的坐标转换统一在 `mahjong_env/tiles.py` 实现，通过 `mahjong.tile.TilesConverter` 构建查找表：

- `tile_to_136(tile_str)` → tile136 整数（-1 表示未知）
- `tile_to_34(tile_str)` → tile34 整数 0-33（-1 表示未知）
- `tile_is_aka(tile_str)` → 是否赤宝牌
- tile34 与 action_space 的 `TILE_NAME_TO_IDX` 完全对齐（验证过）
- `features.py` 和 `replay.py` 和 `legal_actions.py` 均从 `tiles.py` import，不再各自维护副本
- 赤宝牌检测直接比较字符串 `'5mr'/'5pr'/'5sr'`，不再依赖 `FIVE_RED_MAN` 等常量

## 动作空间（action_space.py）

45 个动作：0-33 dahai（tile34 index），34 reach，35-37 chi（low/mid/high），38 pon，39 daiminkan，40 ankan，41 kakan，42 hora，43 ryukyoku，44 none。

chi 类型判断对齐 Mortal `chi_type.rs`：`pai_rank < lo → chi_low`，`lo ≤ pai_rank < hi → chi_mid`，`pai_rank ≥ hi → chi_high`。dahai 赤宝牌归并到对应 tile34（`5mr` → index 4）。

## 状态与副露

`mahjong_env.state.GameState` 维护 4 家状态，`apply_event` 逐事件更新。meld dict 格式：`{type, pai, consumed, target}`，consumed 和 pai 均为内部字符串格式。

`legal_actions.enumerate_legal_actions` 枚举当前合法动作：
- chi consumed 用 `_hand_has_tile`/`_pick_chi_tile` 正确处理赤宝牌等价
- pon/daiminkan consumed 用 `_pick_consumed` 优先选赤宝牌版本
- kakan 用 `_hand_has_tile` + `normalize_tile` 处理赤宝牌等价
- **打牌阶段（actor_to_move==actor）不含 none**：打牌是强制决策，pass 无语义，已删除该分支末尾的 none

## none/pass 样本生成（replay.py）

`build_supervised_samples` 在每个 `dahai` 事件处理完（apply 之后）对其他3个玩家生成 pass 样本：
- 条件：该玩家有非 none 的合法动作（chi/pon/daiminkan/hora）且下一个有意义事件是 `tsumo`（无人鸣/荣）
- label：`{"type": "none", "actor": p}`，`value_target=0.0`
- 被他人抢先（下一事件是他人鸣牌/荣）不生成（被迫 pass，非主动选择）
- 参考：Mortal `libriichi/src/dataset/gameplay.rs:392-407`
- none 样本约占总样本 16%，训练后模型 pass logit 不再异常低

## shanten/waits 注入协议

训练时 `replay.build_supervised_samples` 向 snap 注入：
- `snap["shanten"]` — 动作前向听数（libriichi PlayerState.shanten）
- `snap["waits_count"]` — 动作后进张数
- `snap["waits_tiles"]` — length-34 bool list（libriichi PlayerState.waits）

推理时（bot.py）无注入则 fallback 到 `riichienv.HandEvaluator` 自算。`HandEvaluator` 要求 `riichienv.Meld`，不接受 `mahjong.meld.Meld`。

## bot.py 推理过滤

`V5Bot.react` 在枚举 legal_actions 后，若所有合法动作均为 none（无实质选择），直接返回 none 不进入模型推理，避免无意义决策打印。

## 立直状态处理（legal_actions.py + state.py）

`state.py` 的 `snapshot()` 包含 `pending_reach` 字段（length-4 bool list），`enumerate_legal_actions` 基于此实现三段式立直处理：

1. **`reached=True`（立直已被接受）**：只返回 tsumogiri（`last_tsumo_raw`）+ 可能的 ankan，不做全量推理。
2. **`pending_reach=True`（已宣告立直、待打宣言牌）**：用 `mahjong.Shanten` 逐张计算，只保留移除后 shanten==0 的打法（打出后仍听牌），不加入 reach。
3. **普通打牌**：原有逻辑，含 reach（shanten==0 时）、ankan、kakan。

`tile_to_34` 对字牌（`E/S/W/N/P/F/C`）返回正确 tile34，对 `1z-7z` 格式返回 -1，pending_reach 分支直接用 `_tile_to_34(tile)` 不需要 normalize。

## 训练数据过滤（replay.py）

`build_supervised_samples` 的 `collect_sample` 条件新增：`not (et == "dahai" and state.players[actor].reached)`，过滤立直后摸切样本（无决策价值，约占 dahai 样本 5.5%）。

训练数据是实时从 `.mjson` 解析的（IterableDataset），无需重新生成数据，直接续训即可生效。

## scalar[16-19] style 字段

训练时恒为 0，推理时 bot.py 手动注入 `style_vec=[speed, riichi, value, defense]`（各 [-1,+1]）。
