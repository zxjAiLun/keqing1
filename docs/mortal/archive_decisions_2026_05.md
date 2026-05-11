# Mortal Mainline Archive Decisions

Updated: 2026-05-10

## Decision

The project is contracted from a self-developed Mahjong model mainline into a
Mortal-based enhancement, fine-tuning, selfplay, evaluation, and tooling
project.

Active mainline:

- Mortal + riichienv + libriichi runtime
- Mortal/libriichi observation encoding + Mortal Brain + Dueling DQN checkpoint
  development
- Mortal selfplay replay generation
- Mortal replay sidecar / Q-value / mask export
- replay review GUI
- future Mortal checkpoint comparison and fine-tuning utilities
- future policy/value/rank heads only when attached to Mortal-compatible
  encoding

Archived routes:

- xmodel supervised full retrain and preprocess/cache schema work
- keqingv3 / keqingv3.1 independent growth
- KeqingRL independent policy imitation
- legacy BattleManager selfplay
- rulebase/PPO-only rescue paths
- new custom Mahjong observation encoders for strength work

## Implementation Boundary

Phase 0 freezes direction before new model work. KeqingRL, old materialize
sidecars, self-written replay core, xmodel, and keqingv routes are archive-only
for strength development. Keep only tools that still produce useful MJAI,
`json.gz`, replay review, or metrics artifacts.

Phase 1 moves useful Mortal tools into `scripts/mortal/`, adds fixed-seed
Mortal selfplay tests, and standardizes metric export. Deprecated wrappers at
old paths are retained only for command compatibility.

Phase 2 may mark KeqingRL entrypoints as archived or move them under
`archive/keqingrl_mortal_imitation_202605/`, but only after imports/tests are
handled deliberately.

Phase 3 may move xmodel/keqingv documents and scripts under archive directories.

Phase 4 may split active and archived tests or add pytest archive markers so
default CI does not carry frozen routes.

## Current Tool Paths

- `scripts/mortal/generate_riichienv_selfplay_replays.py`
- `scripts/mortal/materialize_replay_sidecars.py`
- `scripts/mortal/export_decision_review_cases.py`
- `scripts/mortal/one_vs_three_smoke.py`
- `scripts/mortal/ab_match.py`
- `scripts/mortal/eval_metrics.py`

## Phase 0 Freeze Directory

| Category | Handling |
| --- | --- |
| Self-written game core / rules | Archive for strength work. Keep only when needed by UI/gateway compatibility or regression tests. |
| KeqingRL / legacy policy route | Archive. These paths are not environment setup blockers and should not be revived for new training. |
| Old materialize sidecars | Archive, except Mortal replay-review sidecars under `scripts/mortal/`. |
| xmodel / keqingv export formats | Archive for model development. Retain only as historical conversion code until removed from build/test scope. |
| Data download / organizing scripts | Cherry-pick keep when they produce or repair MJAI input data. |
| Replay conversion to Mortal data | Keep when the output is Mortal-compatible `*.json.gz`. |
| Replay review / visualization | Keep when it consumes MJAI/Mortal-review artifacts. |
| Mortal / libriichi / RiichiEnv adapter layer | New mainline. Phase 1 evaluation should use these contracts. |

## Phase 1 Commands

Mortal native fixed-seed smoke:

```bash
uv run python scripts/mortal/one_vs_three_smoke.py \
  --challenger artifacts/mortal_training/mortal.pth \
  --champion artifacts/mortal_training/mortal.pth \
  --seed-start 10000 \
  --seed-key 8192 \
  --seed-count 1 \
  --output-dir artifacts/eval/one_vs_three_smoke
```

RiichiEnv A/B match:

```bash
uv run python scripts/mortal/ab_match.py \
  --model-a artifacts/mortal_training/mortal.pth \
  --model-b artifacts/mortal_serving/backups/mortal_step30000_before_step41200_20260430_201553.pth \
  --games 4 \
  --seat-mode one-vs-three \
  --seed 10000 \
  --output-dir artifacts/eval/riichienv_ab_match
```

Both commands write `metrics.json` using schema
`keqing.mortal.eval.metrics.v1`.

Deprecated wrappers are intentionally retained:

- `scripts/generate_mortal_riichienv_replays.py`
- `scripts/materialize_mortal_replay_sidecars.py`
- `scripts/export_keqingrl_mortal_review_cases.py`
