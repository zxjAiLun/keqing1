# Keqing1 当前项目总览

更新时间：2026-04-21

最新的“当前项目状态 / 活跃工作流 / 风险 / 近期待办”请先看 `docs/project_progress.md`。
这份文档是当前主线的摘要说明，重点回答项目现在在做什么、哪些边界已经冻结、接下来证据链要怎么补齐。

## 1. 一句话判断

`keqing1` 当前是一条以轻量但足够强的麻将模型为目标的建模主线，而不是旧 runtime/bot 系统的维护仓库。

当前固定判断：
- `xmodel1` 是训练窗口主线。
- `keqingv4` 是备线与 Rust ownership 修复线。
- `keqingrl` 是并行推进的 RL 原生工作线。
- 共享语义真源继续向 Rust 收口。
- replay/runtime/bot 只作为模型训练、评测和兼容面的承载层。
- `docs/keqingv4/keqingv4_model_design_v2.md` 现在记录的是 `keqingv4` 当前 backup snapshot 合同，不是一个还要继续展开的大改提案。

## 2. 当前优先级

当前全局优先级保持为：
- `xmodel1`
- `keqingv4`
- `keqingrl`

这三条线当前是并列最高优先级，不需要固定先后顺序。

其后的支撑优先级才是：
- 共享 Rust semantic core 巩固
- `xmodel2` 等有界实验线
- runtime / replay / bot 兼容面清理

这意味着项目当前不再优先做旧模型族的补丁式修修补补，也不再把 battle/replay 侧问题当成主目标；真正的顶层工作面现在是三条并行线，而不是单线串行推进。

## 3. 数据与真源边界

稳定原始牌谱根目录只有三套：
- `artifacts/converted_mjai/ds1`
- `artifacts/converted_mjai/ds2`
- `artifacts/converted_mjai/ds3`

`processed_*/*` 目录都只是某个模型族的预处理产物，不是长期真源数据。

当前共享活跃语义面是：
- `src/mahjong_env/action_space.py`
- `src/mahjong_env/feature_tracker.py`
- `src/mahjong_env/progress_oracle.py`
- `src/training/state_features.py`

旧模型族包已经退出活跃训练/runtime 面，不允许重新把 import 指回被移除的 legacy 包。

## 4. 主线、备线与并行工作面

### 4.1 xmodel1 主线

`xmodel1` 当前目标不是继续扩结构，而是把训练窗口前的证据闭环跑完整。

代码侧当前已经具备：
- Rust-first preprocess/cache 主路径
- dedicated `hora` sample type
- full preprocess 前的随机 preflight export/probe gate
- cache schema metadata repair 工具
- ratio-based 文件采样与自动 `steps_per_epoch`
- candidate quality / special candidate summaries 的新校准
- parity oracle、slice harness、train-time gates

现在真正缺的不是新结构，而是更完整的用户执行证据：
- 最小真实 cache smoke 已经绿色通过
- 第一轮正式短训也已经完成（`artifacts/xmodel1_runs/v1b_train_20260422/`）
- 仍缺 `reach / call / none / hora` 的 slice/review acceptance

### 4.2 keqingv4 备线

`keqingv4` 现在默认不抢训练窗口，它承担两件事：
- 当 `xmodel1` 触发 kill criteria 时作为备线接管
- 继续承担共享语义面向 Rust ownership 收口的试验与修复

当前代码侧已经完成到：
- future-truth 接入 summary/value 路径
- continuation scenario/scoring/aggregation 的共享 Rust helper
- Rust-first inference scoring
- fail-closed hora truth / continuation 边界
- 从 legacy feature surface 迁移到共享活跃 surface
- `event_history(48, 5)` 成为 runtime + preprocess 显式合同
- `v4_opportunity[3]` 取代 summary magic channel
- checkpoint 改为 metadata validate + `strict=True` 的 hard cutover
- 当前 placement-rank 支持已在原线内收完当前边界：
  - raw `final_rank_target` / `final_score_delta_points_target` 已进入 cache
  - `rank_logits` / `final_score_delta` 已进入 model/trainer
  - review / inference 已暴露 `rank_probs` / `final_score_delta` / `rank_pt_value`
  - runtime action-conditioned placement rerank 已落地，但默认仍关闭（`rank_pt_lambda = 0.0`）

因此 `keqingv4` 现在的重点是维持这条显式合同 backup snapshot 的冻结，而不是继续加新结构。

### 4.3 keqingrl 并行工作线

`keqingrl` 当前不是 `xmodel1` 的替代物，也不是 `keqingv4` 的后继命名；它是一条 RL-native 的独立工作线。

当前代码侧已经完成到：
- interactive policy 合同
- variable legal-action distribution
- neural forward contract
- PPO buffer / loss / update helper
- Mahjong env wrapper
- rollout review / export
- longer-run training harness
- widened self-turn action surface：
  - `DISCARD`
  - `REACH_DISCARD`
  - `TSUMO`
  - `ANKAN`
  - `KAKAN`

因此 `keqingrl` 当前的重点不是再证明它“是不是 sidecar”，而是继续补：
- longer-run PPO evidence
- 更宽动作面
- rollout / learner / env 的合同稳定性

## 5. Rust 迁移当前含义

Rust semantic core 迁移已经不再是“是否开始做”，而是“哪些公共语义边界还没彻底归 Rust”。

当前共识是：
- state core 与 legal action structural enumeration 已大体 Rust-owned
- hora/yaku truth 已通过 Rust truth interface 流转
- replay sample construction 与 continuation evaluation 仍是主要剩余表面

当前活跃的 Python 边界主要分三类：
- parity-only oracle：`src/xmodel1/preprocess.py`
- Rust-first shell / emergency fallback surface：`src/mahjong_env/replay.py`、`src/mahjong_env/legal_actions.py`、`src/keqingv4/preprocess_features.py`、`src/inference/scoring.py`、`src/mahjong_env/scoring.py`
- 下一轮要继续压缩的 Python semantic-owner surface：`src/mahjong_env/state.py`、`src/training/state_features.py`、`src/xmodel1/features.py`、`src/xmodel1/candidate_quality.py`

这里的原则已经冻结：
- 默认公共 owner 应该是 Rust
- Python fallback 只允许在 missing capability 时触发
- unexpected drift 不能再被 Python 静默兜底

## 6. 当前真正待验证的内容

项目当前最大的未知数不是“文档怎么写”，而是模型主线证据是否能跑通。

当前最关键的验证面现在是并行的：
- `xmodel1`：
  - 审当前正式短训产物
  - 做 `reach / call / none / hora` 等 slice review acceptance
  - 再决定是否扩大为更广覆盖的后续训练
- `keqingv4`：
  - 做 focused verification
  - 继续共享 Rust ownership 推进
  - 冻结 backup snapshot 的 fail-closed 边界
- `keqingrl`：
  - 跑 longer-run PPO evidence
  - 扩宽动作面
  - 保持 rollout / learner / env 三者合同对齐

在这些并行证据没有跑出来之前，代码侧“已经准备好”不等于模型质量已经被证明。

## 7. 使用这份文档的方式

如果你要快速判断项目现状，建议按下面顺序看：
1. `AGENTS.md`
2. `docs/project_progress.md`
3. `docs/agent_sync.md`
4. `docs/todo_2026_04_24.md`
5. 这份 `docs/project_overview_current.md`

如果项目状态发生变化，应优先更新长期状态板与协作面，而不是只改这份摘要。
