# Keqing1 Project Progress

Updated: 2026-04-28

This is the primary live status board for the repository.

## Executive Summary

- `keqingrl` is now the only model-growth mainline.
- The active design is `KeqingRL-Lite`: rulebase prior + neural delta + actor-critic + rollout-native PPO.
- `xmodel1` and `keqingv4` are frozen assets/baselines. They remain useful for candidate features, after-state ideas, runtime/review adapters, and Rust ownership work, but they are no longer the default training path.
- Teacher source rule: only trained Mortal checkpoints may be used as teacher sources. Do not use `xmodel`, `xmodel1`, or `keqingv4` logits/checkpoints/rollouts as teacher outputs.
- Do not spend default mainline effort on full `xmodel1` preprocess/retrain, new `keqingv4` aux heads, or cache schema churn.
- `rulebase` is now both a compatibility bot and the initial policy prior/autopilot baseline.
- Rollout data is first-class: ordered legal actions, canonical action keys, raw rule scores, prior logits, old log-prob/value, rule/style context, and policy version are part of the learner contract.

## Current Global Judgment

The repository is moving from a supervised-cache workflow to an interactive-policy workflow:

```text
ActorObservation/PublicObservation
-> ordered legal actions
-> Rust rulebase scorer
-> raw_rule_scores
-> centered/clipped prior_logits
-> RulePriorDeltaPolicy
-> rollout buffer
-> PPO update
-> review/eval
```

The core formula is:

```python
prior_logits = (raw_rule_scores - raw_rule_scores.max(dim=-1, keepdim=True).values)
prior_logits = prior_logits.clamp(min=-10.0, max=0.0) / prior_temperature
final_logits = rule_score_scale * prior_logits + neural_delta
```

`raw_rule_scores` preserve rulebase argmax parity. `prior_logits` are policy prior logits. Clipping must not change the raw best action except true raw-score ties.

## Active Workstream: KeqingRL-Lite

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
3. Qualify teacher probes with opportunity-based terminal/action coverage before interpreting topK movement.
4. Treat discard-only probes as contract diagnostics only.
5. Investigate full response-window Mortal mask parity gaps before enabling PASS/RON/PON/CHI training.

## Current Known Environment Note

The workstation `.venv` is broken. Recreate it before running Python tests:

```bash
rm -rf .venv
uv sync
```
