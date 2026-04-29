# Keqing1 当前项目总览

更新时间：2026-04-28

## 一句话判断

`keqing1` 当前主线已经切到 `KeqingRL-Lite`，并且 teacher source 分析已经提前到主线：

```text
rulebase prior
+ neural_delta
+ rollout-native actor-critic
+ KeqingCore/Rust legal-action ownership
+ fixed-seed review/eval
+ Mortal action-Q teacher over KeqingRL legal ActionSpec rows
+ opportunity-qualified terminal/action coverage gates
```

`xmodel1` / `keqingv4` 不再作为成长主线返工，也不能作为 teacher source；它们只保留为 frozen assets、baselines、runtime/Rust 资产库。

## 2026-04-28 理解修正

之前把 Mortal discard-only / no-move / topK gate 结果读成 teacher 强弱判断，这个口径现在标记为错误：

```text
错误理解：discard-only 或 terminal-poor rollout 没有通过 movement/topK gate，说明 Mortal teacher 没用。
最新理解：这类结果只是不合格诊断；没有覆盖被测试的终局/立直/副露/响应机会，就不能判断 teacher 强度。
```

当前固定口径：

- `discard-only` 只保留为合同、mask、teacher consumption 诊断，不作为麻将强度路线。
- `score_changed` 不是和牌覆盖率；立直棒和流局听牌支付也会改分。
- 实际 `hora` / `RON` / `TSUMO` 是受牌山运气影响的 outcome metric，不作为默认 gate。
- 默认 gate 必须看 opportunity coverage：legal terminal/agari rows 与 prepared legal terminal/agari rows。
- 看到“全流局”“没有 ron”“teacher 没撼动 topK”之类结果时，先用 MJAI replay 导出工具回放具体局面，再更新结论。

## 当前优先级

1. KeqingCore/Rust 继续作为 legal action owner；KeqingRL 只消费有稳定 `ActionSpec` 身份的合法动作。
2. Mortal 只用训练出来的 Mortal checkpoint 作为 Q/ranking teacher，不使用 `xmodel` / `xmodel1` / `keqingv4`。
3. 从 `mortal-discard-q` 诊断升级到 `mortal-action-q`：在 KeqingRL legal actions 上打分，仍然 fail closed。
4. teacher probe 先过 opportunity-based terminal/action coverage，再解释 topK movement、fresh gate、paired eval。
5. response-window 的 PASS/RON/PON/CHI mask parity 缺口先调查清楚，再开放训练。

不要默认继续推进：

- 200G 级别 supervised preprocess
- xmodel1 full retrain
- keqingv4 新 aux head
- cache schema cutover 作为强度主策略
- 继续只在 KeqingRL 内部调 PPO / gate / penalty 而不引入新的 topK 排序 teacher
- 把 discard-only no-pass 当成 Mortal teacher 强弱证据
- 用实际和牌数或 `score_changed` 单独判定 teacher gate

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
- `plans/mortal_training_runbook_2026_04_28.md`

The live design doc is:

- `docs/keqingrl/keqingrl_model_design_v1.md`
- `docs/mortal_action_contract.md`
- `docs/keqingrl/mortal_training_workflow.md`

## Read First

1. `docs/project_progress.md`
2. `docs/agent_sync.md`
3. `docs/mortal_action_contract.md`
4. `docs/keqingrl/mortal_training_workflow.md`
5. `plans/mortal_training_runbook_2026_04_28.md`
6. `docs/keqingrl/keqingrl_model_design_v1.md`
7. `plans/keqingrl_lite_mainline_2026_04_24.md`
8. latest `docs/todo_*.md`

`plans/mortal_teacher_contract_2026_04_28.md` is now historical discard-only
diagnostic context. It must not override the active Mortal action contract.
