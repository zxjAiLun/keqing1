# Keqing1 Project Progress

Updated: 2026-05-08

This is the primary live status board for the repository.

## Executive Summary

- Active mainline is now Mortal-based enhancement / fine-tuning / selfplay /
  evaluation / tooling.
- Core runtime direction is `Mortal + riichienv + libriichi`.
- KeqingRL independent Mortal Action-Q imitation is archived experiment context,
  not the current strength route.
- `xmodel1`, `xmodel2`, `keqingv3`, `keqingv3.1`, and `keqingv4` are
  archive-only or frozen compatibility assets. They are not teachers,
  baselines, or retrain candidates for new mainline work.
- Do not spend default effort on 200G supervised preprocess, xmodel full
  retrain, cache schema cutovers, materialize-heavy distillation, legacy
  BattleManager selfplay, or rulebase/PPO-only rescue.
- Current useful tools have been promoted under `scripts/mortal/`.

## Current Global Judgment

The repository has proved that maintaining a separate self-developed model
growth line is lower leverage than building around Mortal's existing
observation, EV table, DQN, selfplay, and training stack.

The default project loop is now:

```text
riichienv 4-Mortal selfplay
-> replay pool / sidecars
-> behavior and checkpoint evaluation
-> replay review GUI
-> small Mortal fine-tuning or tool improvements
```

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

- adjust data source composition
- adjust reward / GRP / rank-point handling
- compare checkpoints under the same seeds and `riichienv` mode
- report behavior stats such as rank, point, agari, houjuu, fuuro, riichi, and
  point per round

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
