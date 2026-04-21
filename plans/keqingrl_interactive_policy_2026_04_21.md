# keqingrl Interactive Policy Blueprint

Updated: 2026-04-21

## Objective

Create a new `keqingrl` family for Mahjong interactive policy learning with this contract:

```text
obs + legal_actions + rule_context
-> action distribution
-> sample / greedy action
-> log_prob / entropy / value
-> rollout trajectory
-> RL learner update
```

This plan is multi-session. It does not move the current mainline/backup judgment:

1. `xmodel1` remains the training-window mainline
2. `keqingrl` is the new interactive RL family
3. `xmodel2` remains the bounded offline placement auxiliary family
4. `keqingv4` remains the backup line and Rust-ownership repair path

## Execution Mode

- Current environment: direct mode
- GitHub workflow assumptions: none required
- Principle: keep every step focused and reversible

## Repository Truth Snapshot

- `xmodel1` is CE-first and frozen on `xmodel1_discard_v3`; do not mutate its public contract for RL work.
- `xmodel2` already exists as an offline `decomposed EV + placement auxiliary v1` scaffold; do not overload its name with self-play policy work.
- `keqingv4` remains the backup line and shared Rust cleanup surface.
- Public legal actions, replay snapshots, and hora/scoring truth are already Rust-first public surfaces or moving that way.
- There is currently no dedicated rollout-native learner family in the repository.

## Invariants

- Do not reopen `xmodel1` architecture while its repaired re-export/retrain evidence is still pending.
- Do not build `keqingrl` inside `training.train_model(...)`.
- Do not make the fixed 45-way flat logit interface the public contract for `keqingrl`.
- Keep rollout storage separate from supervised caches.
- Treat action ordering as a contract, not incidental metadata.
- Prefer correctness and reviewability over throughput in early phases.

## Step Graph

```text
S0 boundary freeze + contract skeleton
 -> S1 action contract + legality tests
 -> S2 random policy + rollout record smoke
 -> S3 neural forward contract
 -> S4 PPO learner on toy env
 -> S5 discard-only Mahjong env loop
 -> S6 partial full-action rollout
 -> S7 opponent pool + eval/review surfaces
```

Parallelism note:

- This plan is mostly serial because action ordering, rollout replay, and env stepping share one contract.
- Safe parallel work is limited to test-only coverage or doc/review tooling after the action contract is frozen.

## Step List

### S0. Boundary Freeze And Contract Skeleton

Context brief:

- The new family should exist in its own namespace and docs before any learner implementation starts.
- The repository currently lacks a rollout-native policy contract.

Primary write scope:

- `docs/project_progress.md`
- `docs/agent_sync.md`
- `docs/todo_2026_04_24.md`
- `docs/keqingrl/keqingrl_model_design_v1.md`
- `plans/keqingrl_interactive_policy_2026_04_21.md`
- `src/keqingrl/`

Tasks:

1. Freeze the family name as `keqingrl`.
2. Define the live contract for policy input/output, sampling, and rollout records.
3. Register the new family in the status board without moving the mainline/backup boundary.
4. Land minimal code scaffolding for actions, distributions, and rollout records.

Verification:

```bash
uv run pytest tests/test_keqingrl_actions.py tests/test_keqingrl_distribution.py tests/test_keqingrl_policy.py -q
```

Exit criteria:

- The family exists in code and docs.
- The repository now has an explicit place for RL-native contracts.

Rollback:

- Remove `src/keqingrl/` and the new docs, leaving the long-lived status board consistent.

### S1. Action Contract And Mahjong Env Bridge

Context brief:

- RL work fails early if action encode/decode and env stepping are unstable.
- Existing Mahjong code already has legal-action enumeration and MJAI action surfaces that can be reused.

Primary write scope:

- `src/keqingrl/actions.py`
- `src/keqingrl/env.py`
- `src/mahjong_env/legal_actions.py`
- `src/mahjong_env/types.py`
- focused tests for round-trip legality

Tasks:

1. Finalize `ActionSpec` encode/decode and MJAI conversion rules.
2. Build a `MahjongEnv` wrapper that exposes ordered legal actions and actor-relative observations.
3. Add legality tests that prove sampled actions can step the env.
4. Fail loudly if action ordering drifts between observation and step.

Verification:

```bash
uv run pytest tests/test_keqingrl_actions.py tests/test_keqingrl_env_contract.py -q
```

Exit criteria:

- Random legal actions can be converted and stepped without illegal-action drift.

Rollback:

- Keep the action contract helpers and remove only the env wrapper layer.

### S2. Random Policy And Rollout Record Smoke

Context brief:

- Before any learner, the project needs proof that legal action lists, sampling, and rollout logging line up.

Primary write scope:

- `src/keqingrl/policy.py`
- `src/keqingrl/distribution.py`
- `src/keqingrl/rollout.py`
- tests for random legality and replayable rollout logs

Tasks:

1. Land a random or zero-logit policy with shared greedy/sample interface.
2. Record `action_index`, `log_prob`, `entropy`, `value`, `policy_version`, and rewards in rollout steps.
3. Prove that greedy/sample always select legal actions.

Verification:

```bash
uv run pytest tests/test_keqingrl_distribution.py tests/test_keqingrl_policy.py tests/test_keqingrl_rollout.py -q
```

Exit criteria:

- The repository can generate rollout-native records without a learner.

Rollback:

- Remove random policy entrypoints but keep the pure contract dataclasses.

### S3. Neural Forward Contract

Context brief:

- The first model milestone is not strength. It is finite, replayable outputs on variable legal-action sets.

Primary write scope:

- `src/keqingrl/obs.py`
- `src/keqingrl/policy.py`
- focused forward-shape tests

Tasks:

1. Add a first state encoder + action encoder forward path.
2. Produce `action_logits[B, A]`, `value[B]`, and `rank_logits[B, 4]`.
3. Verify masking, entropy, and greedy decode on variable `A`.

Verification:

```bash
uv run pytest tests/test_keqingrl_policy_forward.py tests/test_keqingrl_distribution.py -q
```

Exit criteria:

- Neural forward works on batched variable legal-action lists.

Rollback:

- Keep contracts and revert only the neural forward implementation.

### S4. PPO Learner On Toy Env

Context brief:

- PPO should first be validated on a toy environment where reward improvement is easy to diagnose.

Primary write scope:

- `src/keqingrl/ppo.py`
- `src/keqingrl/buffer.py`
- a toy env under tests or fixtures

Tasks:

1. Implement PPO ratio/value/entropy losses against rollout records.
2. Add return/advantage computation and minibatch updates.
3. Prove the learner can improve reward on a toy task.

Verification:

```bash
uv run pytest tests/test_keqingrl_ppo_toy.py -q
```

Exit criteria:

- PPO update works on rollout-native data without Mahjong complexity in the loop.

Rollback:

- Remove the toy env harness while keeping rollout buffers and shared losses.

### S5. Discard-Only Mahjong Env Loop

Context brief:

- Full-action Mahjong is a poor first RL proving ground.
- Discard-only control gives a smaller surface while still validating the interactive family.

Primary write scope:

- `src/keqingrl/env.py`
- `src/keqingrl/selfplay.py`
- `src/keqingrl/rewards.py`
- focused legality and episode-completion tests

Tasks:

1. Let `keqingrl` control discards only.
2. Keep `reach / hora / calls` rule-driven or disabled.
3. Backfill terminal rewards from `pt_map`.

Verification:

```bash
uv run pytest tests/test_keqingrl_env_contract.py tests/test_keqingrl_terminal_rewards.py -q
```

Exit criteria:

- A discard-only interactive loop can complete episodes and produce rollout rewards.

Rollback:

- Revert to random policy control while preserving env contract helpers.

### S6. Partial Full-Action Rollout

Context brief:

- After discard-only stability, open actions gradually instead of jumping straight to all responses and kans.

Primary write scope:

- `src/keqingrl/actions.py`
- `src/keqingrl/env.py`
- `src/keqingrl/policy.py`

Tasks:

1. Open `reach` and `hora`.
2. Open `pass`, then call families.
3. Add legality and action-order tests for every newly opened action type.

Verification:

```bash
uv run pytest tests/test_keqingrl_actions.py tests/test_keqingrl_env_contract.py -q
```

Exit criteria:

- Newly opened action types remain reviewable and legal in rollout traces.

Rollback:

- Disable the last opened action family and keep the rest of the contract stable.

### S7. Opponent Pool, Evaluation, And Review Surfaces

Context brief:

- Self-play only against the latest policy is unstable and hard to review.

Primary write scope:

- `src/keqingrl/opponent_pool.py`
- `src/keqingrl/eval.py`
- `src/replay_ui/` or existing review exporters only if needed

Tasks:

1. Add checkpoint/rule-bot/opponent-pool sampling.
2. Export top-k legal-action review traces with logits/probs/value/rank.
3. Define first evaluation metrics beyond reward:
   - average rank
   - pt utility
   - win rate
   - deal-in rate

Verification:

```bash
uv run pytest tests/test_keqingrl_eval.py -q
```

Exit criteria:

- `keqingrl` outputs are inspectable enough to debug policy collapse and action-order bugs.

Rollback:

- Remove evaluation glue while preserving rollout logs and raw review artifacts.

## Adversarial Review Checklist

Before closing any major `keqingrl` slice, verify:

1. Does it keep `keqingrl` out of the imitation trainers?
2. Does it preserve ordered legal-action identity from observe -> sample -> step -> replay?
3. Is every new learner assumption backed by a focused test?
4. Does it keep `xmodel1` mainline evidence collection unblocked?
5. Is a rollback obvious if the new slice destabilizes legality or replayability?

## Anti-Patterns

- Adding RL heads to `xmodel1` or `keqingv4` and calling that a new family
- Reusing supervised cache rows as rollout storage
- Keeping only chosen MJAI actions and trying to reconstruct old log-probs later
- Opening full-action Mahjong before discard-only legality is stable
- Treating fixed 45-way logits plus mask as the long-term `keqingrl` public contract

## Plan Mutation Protocol

If the plan changes:

1. Update `docs/project_progress.md` first
2. Update `docs/agent_sync.md` if the coordination boundary changed
3. Update this plan with:
   - what changed
   - why
   - which downstream steps moved or split
