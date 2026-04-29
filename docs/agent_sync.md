# Keqing1 Agent Sync

Updated: 2026-04-28

## Short Handoff Summary

- `keqingrl` is the only model-growth mainline.
- `KeqingRL-Lite` is the active design: rulebase prior + neural delta + rollout-native PPO.
- `xmodel1` and `keqingv4` are frozen assets/baselines, not active growth lines.
- Only trained Mortal checkpoints may be used as teacher sources. Never use `xmodel`, `xmodel1`, or `keqingv4` as teacher logits/checkpoints/rollout sources.
- Mortal teacher probe gates must be opportunity-based by default; do not gate on actual `hora` or `score_changed` unless explicitly running an outcome diagnostic.
- Do not default to supervised full preprocess/retrain work.
- The active implementation plan is `plans/keqingrl_lite_mainline_2026_04_24.md`.
- The active design doc is `docs/keqingrl/keqingrl_model_design_v1.md`.

## Read Before Work

1. `docs/project_progress.md`
2. `docs/agent_sync.md`
3. `docs/keqingrl/keqingrl_model_design_v1.md`
4. `plans/keqingrl_lite_mainline_2026_04_24.md`
5. `docs/mortal_action_contract.md`
6. `docs/keqingrl/mortal_training_workflow.md`
7. `git status --short`

## Current Alignment

The main policy contract is:

```text
ActorObservation/PublicObservation
ordered legal ActionSpec list
raw_rule_scores
prior_logits
neural_delta
masked legal-action distribution
rollout step
PPO update
```

`ActionSpec.canonical_key` and legal-action order are public learner contracts.

## Coordination Rules

- Never reinterpret an old rollout `action_index` by rebuilding legal actions.
- PPO batch construction must verify the stored `chosen_action_canonical_key`.
- `rule_score`, observation building, action feature building, and review components must not consume hidden information.
- `MaskedCategorical` must fail closed for all-masked rows, non-finite logits, and illegal selected indices.
- Returns/GAE are per actor trajectory, not global chronological order.
- `RulePriorDeltaPolicy` zero-delta behavior must match the stored rule prior distribution, not only greedy action.
- Discard-only PPO filters policy-visible actions to learner-controlled action types after full-action rulebase/autopilot arbitration.
- Forced terminal/autopilot actions such as `TSUMO`, `RON`, and `RYUKYOKU` stay rule-controlled in early phases.
- Discard-only Mortal teacher probes are contract diagnostics only, not Mahjong strength evidence.
- `score_changed` is not agari coverage. It can be caused by riichi sticks or ryukyoku tenpai payments.
- Default terminal coverage gates use legal opportunity rows: legal terminal/agari rows and prepared legal terminal/agari rows.
- Actual `TSUMO/RON` outcome counts are diagnostic unless `--terminal-coverage-outcome-gate` is explicitly enabled.
- Before reinterpreting surprising Mortal probe results, export the exact seed with `scripts/export_keqingrl_mjai_replay.py` and inspect the readable MJAI trace.
- Style context is carried through but fixed neutral until neutral discard PPO is stable.
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
