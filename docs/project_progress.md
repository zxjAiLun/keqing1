# Keqing1 Project Progress

Updated: 2026-05-10

This is the primary live status board for the repository.

## Executive Summary

- Active mainline is now Mortal-based enhancement / fine-tuning / selfplay /
  evaluation / tooling.
- Core runtime direction is `Mortal + riichienv + libriichi`, with
  Mortal/libriichi v4 observation as the default strength contract.
- KeqingRL independent Mortal Action-Q imitation is archived experiment context,
  not the current strength route.
- `xmodel1`, `xmodel2`, `keqingv3`, `keqingv3.1`, and `keqingv4` are
  archive-only or frozen compatibility assets. They are not teachers,
  baselines, or retrain candidates for new mainline work.
- Do not spend default effort on 200G supervised preprocess, xmodel full
  retrain, cache schema cutovers, materialize-heavy distillation, legacy
  BattleManager selfplay, or rulebase/PPO-only rescue.
- Current useful tools have been promoted under `scripts/mortal/`.
- New strength work must use Mortal/libriichi encoding and Mortal-compatible
  Brain/DQN checkpoints. Do not create a parallel game-board encoding by
  default.
- The two Mortal deep research reports are synchronized into
  `docs/mortal/deep_research_sync_2026_05_10.md`; use that as the concise
  architecture and development-plan summary.

## Current Global Judgment

The repository has proved that maintaining a separate self-developed model
growth line is lower leverage than building around Mortal's existing
observation, EV table, DQN, selfplay, and training stack.

The default project loop is now:

```text
Mortal/libriichi obs encoding
-> Mortal Brain + Dueling DQN checkpoint
-> riichienv 4-Mortal selfplay
-> replay pool / Q-mask sidecars
-> behavior and checkpoint evaluation
-> replay review GUI
-> Mortal-native continued training / fine-tuning / reward experiments
```

## Current Mortal Architecture Reading

Mortal is a native-first hybrid stack, not a Python materialize project:

```text
json.gz replay
-> Rust libriichi replay/state/legal-action/obs pipeline
-> obs_repr(version=4) with online single-player EV tables
-> Python reward bridge using GRP/rank-point deltas
-> Python Brain + Dueling DQN + AuxNet training
-> selfplay/evaluation/replay review tooling
```

The high-value boundary is Mortal/libriichi's `version=4` observation contract:
`(1012, 34)` observation planes, 46-action masks, single-player EV table
features, and GRP-derived rank-point rewards. New model work should attach
heads or reward changes to this backbone instead of rebuilding feature encoding.

## Active Workstreams

### Mortal Selfplay Pipeline

Current entrypoint:

```text
scripts/mortal/generate_riichienv_selfplay_replays.py
```

Purpose:

- generate 4-Mortal half-game/full-game replay pools quickly
- export MJAI replay files and Mortal decision traces
- support future selfplay evaluation and replay review

### Mortal Replay Review Pipeline

Current entrypoints:

```text
scripts/mortal/materialize_replay_sidecars.py
src/replay_ui/
```

Purpose:

- run Mortal review over existing replays
- export `q_values`, masks, and decision traces
- inspect policy/Mortal/rulebase disagreements in the GUI
- support regression checks for action mapping and checkpoint behavior

### Mortal Fine-Tuning / Evaluation

Current references:

```text
plans/mortal_training_runbook_2026_04_28.md
docs/mortal_action_contract.md
```

Future work should prefer small changes to Mortal's existing training workflow:

- keep `version=4` fixed until a deliberate compatibility design says otherwise
- continue or fine-tune Mortal Brain + DQN checkpoints
- adjust data source composition through Mortal-native gzipped MJAI loaders
- run reward-only experiments first through `pts`, `RewardCalculator`, and GRP
  rank-point scalarization
- add future policy/value/rank heads only on top of Mortal-compatible encoding
- compare checkpoints under the same seeds and `riichienv` mode
- report behavior stats such as rank, point, agari, houjuu, fuuro, riichi, and
  point per round

## Development Priority

1. Stabilize the native Mortal baseline: build/import `libriichi`, prepare
   gzipped MJAI data, and run finite GRP/DQN smoke chunks.
2. Run reward-only experiments before changing model shape.
3. Add a policy head beside the existing DQN/value path; keep Q fallback until
   calibration and behavior diagnostics are measured.
4. Treat style prior / style mix as a second-stage research branch after the
   baseline loop is reproducible.
5. Consider RiichiEnv compatibility only through a Mortal-compatible
   observation/action parity layer.

## Archived Workstreams

### KeqingRL Action-Q Imitation

Archived as an engineering experiment. The frozen state is preserved by tag
`archive-keqingrl-mortal-imitation-202605`; removed working-tree files should
not be treated as active dependencies.

### xmodel / keqingv

Archive-only or frozen compatibility:

- no full retraining
- no preprocessing restart
- no new active baseline status
- no teacher role

## Non-Negotiable Boundaries

- Only trained Mortal checkpoints can be used as Mortal teacher/checkpoint
  sources for the active workflow.
- Mortal/libriichi owns observation encoding and legal-action semantics for
  strength work. Custom KeqingRL/keqingv feature encoders are archive context
  unless a new design explicitly reopens them.
- `xmodel`, `xmodel1`, `keqingv4`, and KeqingRL checkpoints must not be promoted
  back to active teachers without updating this status board.
- `rust/keqing_core/` is frozen compatibility/research reference. It should not
  compete with `riichienv + Mortal/libriichi` as the main environment.
- Historical docs and dated todos are context only. They must not override this
  status board.

## Minimum Handoff Output

When finishing a work chunk, leave:

- files changed
- tests run
- tests blocked
- contract risks still open
- next concrete step
