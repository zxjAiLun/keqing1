# Mortal Mainline Archive Decisions

Updated: 2026-05-09

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

Phase 0 freezes direction in documentation without deleting code.

Phase 1 moves useful Mortal tools into `scripts/mortal/` and leaves deprecated
wrappers at old paths for command compatibility.

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

Deprecated wrappers are intentionally retained:

- `scripts/generate_mortal_riichienv_replays.py`
- `scripts/materialize_mortal_replay_sidecars.py`
- `scripts/export_keqingrl_mortal_review_cases.py`
