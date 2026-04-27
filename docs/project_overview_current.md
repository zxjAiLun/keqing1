# Keqing1 当前项目总览

更新时间：2026-04-24

## 一句话判断

`keqing1` 当前主线已经切到 `KeqingRL-Lite`，并且 teacher source 分析已经提前到主线：

```text
rulebase prior
+ neural_delta
+ rollout-native actor-critic
+ discard-only PPO first
+ fixed-seed review/eval
+ Mortal discard-Q topK teacher analysis
```

`xmodel1` / `keqingv4` 不再作为成长主线返工，也不能作为 teacher source；它们只保留为 frozen assets、baselines、runtime/Rust 资产库。

## 当前优先级

1. Mortal teacher source audit and discard-Q topK adapter
2. `keqingrl` contract / rule-prior / PPO 主线 only where needed to consume teacher signals
3. Rust public semantics and rulebase scoring
4. rollout review / fixed-seed evaluation
5. xmodel1 / keqingv4 asset extraction only when directly needed, never as teacher source

不要默认继续推进：

- 200G 级别 supervised preprocess
- xmodel1 full retrain
- keqingv4 新 aux head
- cache schema cutover 作为强度主策略
- 继续只在 KeqingRL 内部调 PPO / gate / penalty 而不引入新的 topK 排序 teacher

## KeqingRL-Lite Contract

核心策略：

```python
raw_rule_scores = rule_scorer(public_obs, legal_actions)
prior_logits = raw_rule_scores - raw_rule_scores.max(dim=-1, keepdim=True).values
prior_logits = prior_logits.clamp(min=-10.0, max=0.0) / prior_temperature
logits = rule_score_scale * prior_logits + neural_delta
```

关键数据必须随 rollout 保存：

- ordered legal action specs
- chosen action index
- chosen action canonical key
- obs tensors
- legal action features and mask
- raw rule scores
- prior logits
- rule context
- neutral style context
- old log-prob and old value
- behavior policy version

## Development Plan

The live implementation plan is:

- `plans/keqingrl_lite_mainline_2026_04_24.md`

The live design doc is:

- `docs/keqingrl/keqingrl_model_design_v1.md`

## Read First

1. `docs/project_progress.md`
2. `docs/agent_sync.md`
3. `plans/mortal_teacher_contract_2026_04_28.md`
4. `plans/mortal_training_runbook_2026_04_28.md`
5. `docs/keqingrl/keqingrl_model_design_v1.md`
6. `plans/keqingrl_lite_mainline_2026_04_24.md`
7. latest `docs/todo_*.md`
