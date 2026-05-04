# Keqing1 Agent Sync

Updated: 2026-05-04

## Short Handoff Summary

- `keqingrl` is the only model-growth mainline.
- The active growth path is Mortal Action-Q imitation over KeqingCore legal
  `ActionSpec` rows. The older KeqingRL-Lite PPO/paired-eval work is
  diagnostic context, not the current training gate.
- `xmodel1` and `keqingv4` are frozen assets/baselines, not active growth lines.
- Only trained Mortal checkpoints may be used as teacher sources. Never use `xmodel`, `xmodel1`, or `keqingv4` as teacher logits/checkpoints/rollout sources.
- Mortal teacher probe gates must be opportunity-based by default; do not gate on actual `hora` or `score_changed` unless explicitly running an outcome diagnostic.
- Do not default to supervised full preprocess/retrain work.
- Current handoff: `docs/keqingrl/keqingrl_mortal_action_q_handoff_2026_05_04.md`.
- Current best imitation checkpoint:
  `reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt`
  (`teacher_kl=0.401925`, `teacher_agreement=0.686201`, fail-closed `0`).
- The active implementation surface is `scripts/run_keqingrl_mortal_imitation.py`
  plus the 2026-05-04 handoff. `plans/keqingrl_lite_mainline_2026_04_24.md`
  remains contract/history context.
- The active design doc is `docs/keqingrl/keqingrl_model_design_v1.md`, with
  current status overridden by the 2026-05-04 handoff when they differ.

## Read Before Work

1. `docs/project_progress.md`
2. `docs/agent_sync.md`
3. `docs/keqingrl/keqingrl_model_design_v1.md`
4. `docs/keqingrl/keqingrl_mortal_action_q_handoff_2026_05_04.md`
5. `docs/mortal_action_contract.md`
6. `docs/keqingrl/mortal_training_workflow.md`
7. `plans/keqingrl_lite_mainline_2026_04_24.md`
8. `git status --short`

## Current Alignment

The main policy contract is:

```text
KeqingCore legal ActionSpec list
Mortal action-Q scores for those legal rows
policy logits over the same legal rows
Mortal imitation CE/KL
audit/replay for changed decisions
```

`ActionSpec.canonical_key` and legal-action order are public learner contracts.

## Coordination Rules

- Never reinterpret an old rollout `action_index` by rebuilding legal actions.
- PPO batch construction must verify the stored `chosen_action_canonical_key`.
- `rule_score`, observation building, action feature building, and review components must not consume hidden information.
- `MaskedCategorical` must fail closed for all-masked rows, non-finite logits, and illegal selected indices.
- Returns/GAE are per actor trajectory, not global chronological order.
- `RulePriorDeltaPolicy` zero-delta behavior must match the stored rule prior distribution, not only greedy action.
- Historical discard-only PPO filters policy-visible actions to learner-controlled
  action types after full-action rulebase/autopilot arbitration; current
  imitation uses the broader scoped action set listed in the handoff.
- Forced terminal/autopilot actions such as `TSUMO`, `RON`, and `RYUKYOKU` stay rule-controlled in early phases.
- Discard-only Mortal teacher probes are contract diagnostics only, not Mahjong strength evidence.
- `score_changed` is not agari coverage. It can be caused by riichi sticks or ryukyoku tenpai payments.
- Default terminal coverage gates use legal opportunity rows: legal terminal/agari rows and prepared legal terminal/agari rows.
- Actual `TSUMO/RON` outcome counts are diagnostic unless `--terminal-coverage-outcome-gate` is explicitly enabled.
- Before reinterpreting surprising Mortal probe results, export the exact seed with `scripts/export_keqingrl_mjai_replay.py` and inspect the readable MJAI trace.
- Use `contract_scoreboard.*` as the Mortal action-Q go/no-go surface; use `mortal_action_mapping_audit.*` to debug response mask parity gaps.
- For the current `PASS/RON/PON/CHI` response-stage probe, KAN is out of learner scope. Run with `--no-mortal-teacher-strict-extra-mask` so extra Mortal id `42` is audit-only; missing controlled legal actions remain fail-closed.
- Historical 2026-04-29 `mortal-action-q` status:
  - Bridge-fix contract probe passed contract and opportunity gates.
  - Conservative movement grid still had no top1 movement.
  - A calibrated candidate now exists at
    `reports/keqingrl_mortal_action_q_candidate_cfg3_fresh_seedshift_20260429_source93_step20000/checkpoint_config_000/policy_final.pt`.
  - Candidate hyperparams: `rule_score_scale=0.25`, `lr=0.0085`,
    `epochs=3`, `clip=0.2`, `teacher_ce_coef=10`,
    `quality_max_prior_margin_p50=3.5`.
  - Candidate gates: fail-closed `0`, train `top1_changed=0.0314`, fresh
    `top1_changed=0.0293`, `approx_kl=0.0134`, `clip_fraction=0.205`,
    `qualified_for_eval=True`.
  - Paired diag32 proxy:
    `reports/keqingrl_mortal_action_q_paired_eval_20260429_source93_step20000_cfg3_seedshift_diag32`
    showed `delta_vs_zero=+0.046875`, eval-diagnostic
    `top1_changed=0.0184`, `changed_rank_mean=2.28`, `rank_ge5=0`.
  - This is movement-bearing proxy evidence only. It is no longer the default
    training gate after user direction to learn Mortal decisions directly.
- User direction after that: stop using validation as the blocker and train to
  imitate Mortal decisions. A follow-up imitation-heavy run from the calibrated
  cfg3 checkpoint is saved at
  `reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_cfg3_iter8/checkpoint_config_000/policy_final.pt`.
  It used `episodes=64`, `iterations=8`, `lr=0.006`, `epochs=3`,
  `clip=0.2`, `teacher_source=mortal-action-q`, `teacher_ce_coef=10`,
  `rule_kl=0`, `entropy=0`, and the same full action scope. Final training
  summary: `mapping=1204/1204`, fail-closed `0`, `top1_changed=0.0407`,
  `changed_rank_mean=2.10`, `rank_ge5=0`, `approx_kl=0.0125`,
  `clip_fraction=0.1395`, checkpoint SHA
  `645e41e5730c439e2355deb934bb2a0a388a84ea776073c8415e6c40daeb3b7e`.
- 2026-04-30 all-seat imitation superseded the `cfg3_iter8` checkpoint. Current
  best checkpoint:
  `reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt`,
  SHA `0e2acefc502472e804afebd272a8fb0e30093434a69125b9ec09c19ac17ad1a2`.
  Metrics: `teacher_kl=0.401925`, `teacher_ce=0.809471`,
  `teacher_agreement=0.686201`, mapping `5422/5422`, fail-closed `0`,
  `rank_ge5=0`.
- Follow-up `lr=0.003` continuation and `lr=0.0025` probe did not beat the
  current best. Do not continue that branch blindly; prefer a bounded
  `teacher_topk=4` probe or readable replay audit.
- Current imitation performance improvements include deferred Mortal runtime,
  batched teacher eval, incremental MortalObservationBridge cache,
  post-rollout bridge materialization, checkpoint-row dedupe, iteration-specific
  checkpoints, and all-seat learner collection.
- KAN remains out of training scope. Extra Mortal id `42` is audit-only under
  `--no-mortal-teacher-strict-extra-mask`; missing controlled legal actions
  still fail closed.
- Style context is carried through but fixed neutral for the current imitation
  line unless a future status update opens style conditioning.
- Checkpoint metadata must include policy/action/observation/action-feature/env/rule-score contract versions.
- Do not describe `xmodel1` or `keqingv4` as the active growth mainline unless the status board is updated in the same change.

## Frozen Asset Boundaries

`xmodel1` can be used for:

- candidate feature references
- after-state/response design references
- old baseline comparison

`keqingv4` can be used for:

- runtime adapter references
- review/scoring references
- Rust ownership examples

Neither should receive default full retrain/cache-schema work as part of the KeqingRL-Lite mainline.

## Minimum Handoff Output

When finishing a work chunk, leave:

- files changed
- tests run
- tests blocked
- contract risks still open
- next concrete step
