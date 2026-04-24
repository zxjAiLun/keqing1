# KeqingRL-Lite Mainline Plan

Updated: 2026-04-24

## Summary

Switch the active growth path to `keqingrl`:

```text
rulebase prior
+ neural delta
+ rollout-native actor-critic
+ discard-only PPO first
```

`xmodel1` and `keqingv4` are frozen assets/baselines, not default retraining targets.

## Public Formula

```python
raw_rule_scores = rule_scorer(public_obs, legal_actions)
prior_logits = raw_rule_scores - raw_rule_scores.max(dim=-1, keepdim=True).values
prior_logits = prior_logits.clamp(min=clip_min, max=0.0) / prior_temperature
final_logits = rule_score_scale * prior_logits + neural_delta
```

Defaults:

```yaml
clip_min: -10.0
clip_max: 0.0
prior_temperature: 1.0
rule_score_scale: 1.0
prior_kl_eps: 1e-4
gamma: 1.0
gae_lambda: 0.95
```

## Contract Requirements

- `ActionSpec.canonical_key` is derived with `field(init=False)`.
- Legal action order is a rollout contract.
- PPO cannot rebuild legal actions or features to train old rollouts.
- `raw_rule_scores` and `prior_logits` are stored separately.
- Hidden information is forbidden in rule score, obs building, action feature building, and review components.
- Checkpoint and rollout metadata must carry contract versions.

## Phase List

### P0. Documentation Switch

Update long-lived docs so new sessions start from the KeqingRL-Lite mainline.

### P1. Core Contracts

Implement:

- canonical action keys
- `RuleContext`
- `RewardSpec`
- neutral `StyleContext`
- rollout contract metadata
- action-order assertion

### P2. Rulebase Scoring

Implement batched Rust scoring:

```text
score_rulebase_actions(observation, legal_actions) -> all scores
```

Return raw score, priority, tie-break rank, and components. Argmax must match `choose_rulebase_action`.

### P3. RulePriorPolicy / RulePriorDeltaPolicy

Zero-delta policy must match rule-prior logits and probabilities exactly on legal actions.

### P4. Env Observe / Autopilot Arbitration

For discard-only, policy sees only filtered `DISCARD` actions after full-action rulebase arbitration.

### P5. PPO Safety

Use stored `old_log_prob`, `old_value`, `raw_rule_scores`, `prior_logits`. Track rule KL, agreement, entropy, clip fraction, and delta metrics.

### P6. Critic Pretraining

Generate rollouts from zero-delta rule-prior sampling by default. Train only value/rank heads.

### P7. Discard-Only PPO

Run fixed-seed smoke with terminal rank-pt reward and neutral style.

### P8. Review / Latency

Review JSONL exposes rule score, prior logit, neural delta, final logit, probabilities, contexts, and action canonical keys.

Latency smoke reports Rust scoring call latency, Python policy forward latency, env step latency, decisions/sec, and games/sec.

### P9. Progressive Unlock

Unlock order:

```text
DISCARD
-> TSUMO/RON or autopilot terminal
-> REACH_DISCARD
-> PASS
-> PON
-> CHI
-> KAN family
```

### P10. Fixed-Seed / Duplicate Evaluation

Separate eval from training rollouts. Compare against rulebase and frozen snapshots with seat rotation. Report rank pt, rank, 4th rate, win rate, deal-in rate, call rate, and riichi rate.

## Verification

```bash
uv run pytest tests/test_keqingrl_actions.py tests/test_keqingrl_distribution.py -q
uv run pytest tests/test_keqingrl_rule_score.py tests/test_keqingrl_policy_lite.py -q
uv run pytest tests/test_keqingrl_env_contract.py tests/test_keqingrl_selfplay.py -q
uv run pytest tests/test_keqingrl_ppo_toy.py tests/test_keqingrl_training.py tests/test_keqingrl_review.py -q
cargo test --manifest-path rust/keqing_core/Cargo.toml rulebase
```
