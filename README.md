# Keqing1

Keqing1 is now a Mortal-based Riichi Mahjong toolkit.

The active project direction is:

```text
Mortal + riichienv + libriichi
-> Mortal selfplay replay pools
-> Mortal Q/mask sidecars
-> replay review GUI
-> Mortal-native checkpoint evaluation / fine-tuning / selfplay training
```

The old self-developed growth routes (`keqingrl`, `xmodel*`, `keqingv*`) are
frozen or removed from the active runtime. Do not restart large supervised
preprocess, cache schema cutovers, xmodel retraining, or KeqingRL imitation as
the default path.

The model-development boundary is now explicit: do not design a new Mahjong
board encoding for strength work. Strong-model work should reuse
Mortal/libriichi observation encoding, Mortal's Brain encoder, and the Mortal
Dueling DQN action-value framework. Future policy or value heads should attach
to that Mortal-compatible backbone instead of reviving an independent
KeqingRL observation stack.

## Workspace Layout

- `training/` owns Mortal training, self-play, evaluation, research notes, and
  runbooks.
- `workbench/` owns the control-plane backend, replay UI, participant/ladder
  operations, and local launchers.
- `data/` is the Git-ignored local root for mutable Workbench state. Set
  `KEQING_DATA_ROOT` to use a different root.
- `src/` retains the shared Python runtime and inference code used by both
  areas during this first split.

Existing `artifacts/` data is intentionally left in place. Model checkpoint,
dataset, and training-run path migration happens consumer by consumer rather
than by copying or deleting existing data.

## Environment

Install Python dependencies with:

```bash
uv sync
```

Install replay UI dependencies with:

```bash
cd workbench/replay_ui
npm install
```

## Active Entry Points

Generate one or more Mortal self-play hanchan logs:

```bash
uv run python training/mortal/selfplay_native.py \
  --model artifacts/mortal_serving/mortal.pth \
  --output-dir data/replays/mortal_selfplay_smoke \
  --seed-start 0 \
  --games 1
```

Run local replay/review service:

```bash
uv run python workbench/main.py local --port 8000
```

Run gateway only:

```bash
uv run python workbench/main.py --gateway-port 11600 tenhou
```

Supported active bot names:

- `mortal`
- `rulebase`

## Key Directories

- `third_party/Mortal/`: upstream Mortal/libriichi code
- `training/`: active Mortal workflow utilities, training helpers, and research notes
- `workbench/`: local control-plane backend, launchers, and replay UI
- `data/`: Git-ignored local mutable data; see `data/README.md`
- `artifacts/`: legacy local Mortal training/checkpoint artifacts, retained during migration
- `src/inference/mortal_bot.py`: Mortal checkpoint-backed runtime wrapper
- `workbench/replay_ui/`: replay and decision review GUI
- `src/mahjong_env/`: shared Mahjong semantics still used by tooling
- `rust/keqing_core/`: frozen compatibility/research reference
- `docs/`: current status boards and workflow notes

## Current Read First

1. `docs/project_overview_current.md`
2. `training/docs/mortal/current_mainline.md`
3. `training/docs/mortal/ladder_publisher_integration.md`
4. `training/plans/mortal_training_runbook_2026_04_28.md`

## Verification

Focused active checks:

```bash
uv run pytest -q
cd workbench/replay_ui && npm run build
cargo test --manifest-path rust/keqing_core/Cargo.toml
```
