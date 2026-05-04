# Keqing1 Project Progress

Updated: 2026-05-04

This is the primary live status board for the repository.

## Executive Summary

- `keqingrl` is now the only model-growth mainline.
- The active growth path is now Mortal Action-Q imitation over KeqingCore legal
  `ActionSpec` rows. PPO/paired-eval probes are historical diagnostics unless a
  later status update explicitly promotes them again.
- `xmodel1` and `keqingv4` are frozen assets/baselines. They remain useful for candidate features, after-state ideas, runtime/review adapters, and Rust ownership work, but they are no longer the default training path.
- Teacher source rule: only trained Mortal checkpoints may be used as teacher sources. Do not use `xmodel`, `xmodel1`, or `keqingv4` logits/checkpoints/rollouts as teacher outputs.
- Do not spend default mainline effort on full `xmodel1` preprocess/retrain, new `keqingv4` aux heads, or cache schema churn.
- `rulebase` is now both a compatibility bot and the initial policy prior/autopilot baseline.
- Rollout data is first-class: ordered legal actions, canonical action keys, raw rule scores, prior logits, old log-prob/value, rule/style context, and policy version are part of the learner contract.
- Current handoff: `docs/keqingrl/keqingrl_mortal_action_q_handoff_2026_05_04.md`.
- Current best KeqingRL imitation checkpoint:
  `reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt`
  with `teacher_kl=0.401925`, `teacher_agreement=0.686201`,
  `mapping=5422/5422`, and fail-closed count `0`.

## Current Global Judgment

The repository has moved from supervised-cache work, through KeqingRL-Lite PPO
diagnostics, into a direct Mortal imitation workflow:

```text
KeqingCore legal action enumeration
-> KeqingRL ActionSpec canonical identity
-> Mortal action-Q scores over those legal rows
-> KeqingRL policy CE/KL imitation
-> readable replay / audit diagnostics
```

The old rule-prior delta formula remains part of the policy architecture:

```python
prior_logits = (raw_rule_scores - raw_rule_scores.max(dim=-1, keepdim=True).values)
prior_logits = prior_logits.clamp(min=-10.0, max=0.0) / prior_temperature
final_logits = rule_score_scale * prior_logits + neural_delta
```

`raw_rule_scores` preserve rulebase argmax parity. `prior_logits` are policy prior logits. Clipping must not change the raw best action except true raw-score ties.

The current training objective is not rank reward and not short-run paired
strength. It is imitation fidelity to Mortal's action-Q distribution over the
same legal actions KeqingRL can actually execute.

## Active Workstream: Mortal Action-Q Imitation

Dedicated script:

```text
scripts/run_keqingrl_mortal_imitation.py
```

Primary outputs:

```text
imitation_summary.csv
imitation_summary.md
imitation_iterations.csv
mortal_action_mapping_audit.csv
mortal_action_mapping_examples.jsonl
checkpoint_summary.csv
checkpoint_iterations.csv
changed_decisions.csv
changed_decisions.readable.md
```

Current training scope:

```text
self-turn: DISCARD / REACH_DISCARD / TSUMO / RYUKYOKU
response:  PASS / RON / PON / CHI
autopilot: TSUMO / RON / RYUKYOKU
out:       DAIMINKAN / ANKAN / KAKAN
```

KAN remains out of scope because Mortal id `42` is a coarse family decision
while KeqingRL legal actions distinguish kan forms and tile choices.

### Latest Imitation Chain

The current best chain started from the earlier `cfg3_iter8` checkpoint, then
continued through all-seat imitation. All-seat collection improved throughput
because each selfplay episode contributes learner rows for seats `0 1 2 3`
instead of only seat `0`.

Best known checkpoint:

```text
reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt
sha256: 0e2acefc502472e804afebd272a8fb0e30093434a69125b9ec09c19ac17ad1a2
```

Best metrics:

```text
teacher_kl=0.401925
teacher_ce=0.809471
teacher_agreement=0.686201
mapping=5422/5422
fail_closed=0
rank_ge5=0
```

Follow-up `lr=0.003` continuation and `lr=0.0025` probe did not beat this
checkpoint. The next useful hypothesis is likely a bounded `teacher_topk=4`
probe or deeper readable replay audit, not another blind continuation.

### Mortal Teacher Correction

The 2026-04-28 Mortal probes corrected an earlier interpretation mistake:

```text
WRONG: discard-only / terminal-poor no-move runs prove Mortal teacher weakness.
RIGHT: such runs are unqualified unless the batch covers the action opportunities being tested.
```

Current gate policy:

- `discard-only` Mortal probes are infrastructure diagnostics, not strength evidence.
- `score_changed` is not an agari proxy; riichi sticks and ryukyoku payments can change scores without `hora`.
- Actual `hora` count is an outcome metric affected by wall luck. It must not be a default teacher gate.
- The default `--terminal-coverage-gate` is opportunity-based: legal terminal/agari rows and prepared legal terminal/agari rows.
- Outcome thresholds only gate when `--terminal-coverage-outcome-gate` is explicitly enabled.
- Surprising results must be checked with `scripts/export_keqingrl_mjai_replay.py` before changing conclusions.
- `mortal-action-q` probes now report `contract_scoreboard.*` plus
  `mortal_action_mapping_audit.*`; response mask gaps block contract
  qualification instead of being interpreted as teacher weakness.
- Current response-stage probes keep KAN family out of learner scope. Extra
  Mortal id `42` is therefore audit-only when
  `--no-mortal-teacher-strict-extra-mask` is set; missing controlled legal
  actions still fail closed.
- 2026-04-29 `mortal-action-q` contract probe passed after fixing Mortal event
  bridge handling for sequential response windows after riichi acceptance:
  mapping coverage was complete, fail-closed count was zero, and the
  opportunity gate passed. The first paired seat-rotation proxy remained
  rule-prior equivalent (`delta_vs_zero=0`, `top1_changed=0`), so that run only
  validated plumbing.
- Later 2026-04-29 movement calibration found a fresh-qualified Mortal
  action-Q checkpoint at
  `reports/keqingrl_mortal_action_q_candidate_cfg3_fresh_seedshift_20260429_source93_step20000`:
  `lr=0.0085`, `epochs=3`, `clip=0.2`, `teacher_ce_coef=10`,
  `rule_score_scale=0.25`, train `top1_changed=0.0314`, fresh
  `top1_changed=0.0293`, `approx_kl=0.0134`, `clip_fraction=0.205`,
  fail-closed count `0`, and `qualified_for_eval=True`.
- The paired diag32 proxy at
  `reports/keqingrl_mortal_action_q_paired_eval_20260429_source93_step20000_cfg3_seedshift_diag32`
  showed `delta_vs_zero=+0.046875` against both `rule_prior_greedy` and
  `rulebase`, with eval-diagnostic `top1_changed=0.0184`,
  `changed_rank_mean=2.28`, `rank_ge5=0`, and `changed_margin_p50=3`.
  Treat this as historical movement-bearing proxy evidence, not the current
  training gate.
- After user direction to prioritize learning Mortal decisions over more
  validation, a direct imitation-heavy follow-up was run from the cfg3
  checkpoint:
  `reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_cfg3_iter8`.
  It used `episodes=64`, `iterations=8`, `lr=0.006`, `epochs=3`,
  `clip=0.2`, `teacher_ce_coef=10`, `rule_kl=0`, `entropy=0`, and the same
  Mortal action-Q scope. Final checkpoint:
  `checkpoint_config_000/policy_final.pt`, SHA
  `645e41e5730c439e2355deb934bb2a0a388a84ea776073c8415e6c40daeb3b7e`.
  Final train stats: mapping `1204/1204`, fail-closed `0`,
  `top1_changed=0.0407`, `changed_rank_mean=2.10`, `rank_ge5=0`,
  `approx_kl=0.0125`, `clip_fraction=0.1395`.
- 2026-04-30 direct imitation superseded the `cfg3_iter8` checkpoint. Use the
  current best all-seat checkpoint listed above for new KeqingRL imitation work.

### Contract Status

Landed or being landed:

- structured `ActionSpec` with derived `canonical_key`
- `RuleContext` and `RewardSpec`
- neutral `StyleContext`
- rollout metadata for action-order safety and policy versioning
- `MaskedCategorical` fail-closed semantics
- Rust rulebase scoring API
- `RulePriorPolicy` and `RulePriorDeltaPolicy`
- PPO rule-KL metrics and delta metrics

### Non-Negotiable Rules

- PPO never re-enumerates legal actions to interpret an old action index.
- `legal_action_specs[action_index].canonical_key` must match stored `chosen_action_canonical_key`.
- `rule_score`, observation building, action feature building, and review components must consume public/actor observations only.
- GAE/returns are computed per `(episode_id, actor_id)`, not over one global chronological step stream.
- Checkpoint and rollout contract versions fail closed on mismatch.
- Phase 1-7 use neutral style only: `[0, 0, 0, 0, 0]`.

## Frozen Asset Lines

### xmodel1

Frozen role:

- candidate/action feature ideas
- response/after-state design reference
- Rust export/profile lessons
- baseline checkpoints and review slices

Not default work:

- full supervised preprocess reruns
- new cache schema revisions
- more aux heads as the main strength strategy

### keqingv4

Frozen role:

- backup runtime/checkpoint line
- Rust ownership repair examples
- inference/review adapter reference
- placement/rank-pt implementation reference

Not default work:

- new model-family fork
- additional head stacking
- replacing `keqingrl` as the growth line

## Immediate Development Order

1. Keep KeqingCore/Rust as legal-action owner and fail closed on `ActionSpec` / mask mismatch.
2. Use Mortal trained checkpoints only as Q/ranking teachers over KeqingRL legal actions.
3. For current model growth, train Mortal Action-Q imitation directly; do not
   use paired eval as the default blocker.
4. Treat discard-only probes as contract diagnostics only.
5. Keep KAN family out of training scope until a separate id-42/kan-select
   contract is implemented.
6. Use readable replay/audit artifacts to inspect changed decisions before
   claiming strength.

## Current Known Environment Note

Use the project `uv` environment for tests and scripts. If the local `.venv`
gets corrupted or imports fail, recreate it with:

```bash
rm -rf .venv
uv sync
```
