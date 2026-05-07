# Keqing1

Keqing1 is now a Mortal-based Riichi Mahjong toolkit.

The active project direction is:

```text
Mortal + riichienv + libriichi
-> Mortal selfplay replay pools
-> Mortal Q/mask sidecars
-> replay review GUI
-> checkpoint evaluation / fine-tuning utilities
```

The old self-developed growth routes (`keqingrl`, `xmodel*`, `keqingv*`) are
frozen or removed from the active runtime. Do not restart large supervised
preprocess, cache schema cutovers, xmodel retraining, or KeqingRL imitation as
the default path.

## Environment

Install Python dependencies with:

```bash
uv sync
```

Install replay UI dependencies with:

```bash
cd src/replay_ui
npm install
```

## Active Entry Points

Generate 4-Mortal RiichiEnv selfplay replays:

```bash
uv run python scripts/mortal/generate_riichienv_selfplay_replays.py \
  --model artifacts/mortal_serving/mortal.pth \
  --output-dir artifacts/replays/mortal_selfplay_smoke \
  --games 1
```

Materialize Mortal review sidecars for existing MJAI replays:

```bash
uv run python scripts/mortal/materialize_replay_sidecars.py \
  --replay-dir artifacts/replays \
  --model artifacts/mortal_serving/mortal.pth \
  --recursive
```

Run local replay/review service:

```bash
uv run python src/main.py local --port 8000
```

Run gateway only:

```bash
uv run python src/main.py --gateway-port 11600 tenhou
```

Supported active bot names:

- `mortal`
- `rulebase`

## Key Directories

- `third_party/Mortal/`: upstream Mortal/libriichi code
- `artifacts/mortal_training/`: local Mortal training/checkpoint artifacts
- `scripts/mortal/`: active Mortal workflow utilities
- `src/inference/mortal_bot.py`: Mortal checkpoint-backed runtime wrapper
- `src/replay_ui/`: replay and decision review GUI
- `src/mahjong_env/`: shared Mahjong semantics still used by tooling
- `rust/keqing_core/`: frozen compatibility/research reference
- `docs/`: current status boards and workflow notes

## Verification

Focused active checks:

```bash
uv run pytest -q
cd src/replay_ui && npm run build
cargo test --manifest-path rust/keqing_core/Cargo.toml
```
