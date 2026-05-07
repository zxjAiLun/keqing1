# Keqing1 当前项目总览

更新时间：2026-05-08

## 一句话判断

`keqing1` 当前正式从自研麻将模型成长线收缩为：

```text
Mortal-based enhancement / fine-tuning / selfplay / evaluation toolkit
```

主线不再是 `xmodel`、`keqingv3/v3.1`、`keqingv4` 或 KeqingRL 独立 policy 成长。
后续默认投入应围绕 Mortal 生态完成小步增强、微调、自对战、评测和可视化工具化。

## Active Mainline

当前主线由三层组成：

```text
Core runtime:
  Mortal + riichienv + libriichi

Tools:
  riichienv 4-Mortal selfplay replay generation
  Mortal replay sidecar / Q-value / mask export
  replay review GUI
  checkpoint/selfplay evaluation utilities

Archive:
  xmodel
  keqingv3 / keqingv3.1 / keqingv4 growth attempts
  KeqingRL independent Mortal Action-Q imitation
  legacy BattleManager selfplay and rulebase/PPO rescue routes
```

必须保留并继续维护的核心参考：

- `third_party/Mortal/`
- `artifacts/mortal_training/`
- `plans/mortal_training_runbook_2026_04_28.md`
- `docs/mortal_action_contract.md`
- `scripts/mortal/generate_riichienv_selfplay_replays.py`
- `scripts/mortal/materialize_replay_sidecars.py`
- `src/replay_ui/`
- `src/inference/mortal_bot.py`

## Archived Experiments

KeqingRL Action-Q imitation is now archived as an engineering experiment, not
the active strength route. The frozen state is preserved by tag
`archive-keqingrl-mortal-imitation-202605`; removed working-tree files should
not be treated as missing active dependencies.

`xmodel1` / `xmodel2` / `keqingv3` / `keqingv3.1` / `keqingv4` are archive-only
or frozen compatibility assets. They are not active baselines, teachers, or full
retrain candidates.

## Frozen Compatibility

`rust/keqing_core/` and `docs/rust_refactor/` remain useful as compatibility and
research references:

- old Keqing replay and `ActionSpec` contract context
- GUI/tool parsing support where still needed
- historical implementation reference for self-contained experiments

They should not compete with `riichienv + Mortal/libriichi` as the default main
environment or legal owner.

## Do Not Restart By Default

- 200G supervised preprocess pipelines
- xmodel supervised full retrain
- keqingv3 / keqingv3.1 feature expansion
- KeqingRL independent policy imitation as the strength route
- materialize-heavy policy distillation as the main path
- legacy BattleManager selfplay
- rulebase/PPO-only rescue
- paired eval as a blocker before Mortal workflow progress
- discard-only no-pass probes as Mortal teacher strength evidence
- large preprocessing pipelines not directly compatible with Mortal training

## Current Read First

1. `docs/project_progress.md`
2. `docs/agent_sync.md`
3. `docs/docs_index.md`
4. `docs/mortal/archive_decisions_2026_05.md`
5. `docs/mortal_action_contract.md`
6. `plans/mortal_training_runbook_2026_04_28.md`

Historical KeqingRL and xmodel documents may explain prior decisions, but they
must not override this Mortal-based mainline unless this status board is updated
in the same change.
