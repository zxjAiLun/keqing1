# Keqing1 当前项目总览

更新时间：2026-05-04

## 一句话判断

`keqing1` 当前模型成长主线已经从 KeqingRL-Lite 的 PPO/paired-eval 诊断，
切到 **Mortal Action-Q imitation**：

```text
KeqingCore/Rust legal-action ownership
+ KeqingRL ActionSpec canonical identity
+ Mortal action-Q teacher over those legal rows
+ CE/KL imitation toward Mortal decisions
+ readable replay / audit diagnostics
```

`xmodel1` / `keqingv4` 不再作为成长主线返工，也不能作为 teacher source；它们只保留为 frozen assets、baselines、runtime/Rust 资产库。

当前 handoff：

```text
docs/keqingrl/keqingrl_mortal_action_q_handoff_2026_05_04.md
```

当前最佳 KeqingRL imitation checkpoint：

```text
reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt
teacher_kl=0.401925
teacher_agreement=0.686201
mapping=5422/5422
fail_closed=0
```

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
3. 当前训练默认走 `scripts/run_keqingrl_mortal_imitation.py`，目标是学 Mortal 在 KeqingRL legal rows 上的决策分布。
4. `mortal-discard-q`、paired diag、小样本 `hora/score_changed` 都只保留为诊断，不作为当前默认 gate。
5. response-window 的 PASS/RON/PON/CHI 已进入 imitation scope；KAN family 继续 out-of-scope，直到单独完成 id42/kan-select 合同。
6. 新实验优先围绕当前最佳 checkpoint 做小假设，例如 `teacher_topk=4`，不要盲目继续 `lr=0.003/0.0025` 链。

不要默认继续推进：

- 200G 级别 supervised preprocess
- xmodel1 full retrain
- keqingv4 新 aux head
- cache schema cutover 作为强度主策略
- 继续只在 KeqingRL 内部调 PPO / gate / penalty 而不引入新的 topK 排序 teacher
- 把 discard-only no-pass 当成 Mortal teacher 强弱证据
- 用实际和牌数或 `score_changed` 单独判定 teacher gate
- 把 paired eval 重新设为当前 imitation 训练的默认 blocker
- 训练 KAN family，除非先补完 Mortal id42 / `at_kan_select` 合同

## KeqingRL Contract

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

The live implementation/handoff surface is:

- `docs/keqingrl/keqingrl_mortal_action_q_handoff_2026_05_04.md`
- `scripts/run_keqingrl_mortal_imitation.py`

The live design/reference docs are:

- `docs/keqingrl/keqingrl_model_design_v1.md`
- `docs/mortal_action_contract.md`
- `docs/keqingrl/mortal_training_workflow.md`
- `plans/mortal_training_runbook_2026_04_28.md`

## Read First

1. `docs/project_progress.md`
2. `docs/agent_sync.md`
3. `docs/keqingrl/keqingrl_mortal_action_q_handoff_2026_05_04.md`
4. `docs/mortal_action_contract.md`
5. `docs/keqingrl/mortal_training_workflow.md`
6. `docs/riichi_dev_mortal_handoff.md`
7. `plans/mortal_training_runbook_2026_04_28.md`
8. `docs/keqingrl/keqingrl_model_design_v1.md`
9. `plans/keqingrl_lite_mainline_2026_04_24.md`
10. latest `docs/todo_*.md`

`plans/mortal_teacher_contract_2026_04_28.md` is now historical discard-only
diagnostic context. It must not override the active Mortal action contract.
