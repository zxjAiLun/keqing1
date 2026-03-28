# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable):**
```bash
pip install -e .
```

**Run tests:**
```bash
pytest
# Single test file:
pytest tests/test_replay_log.py
# Single test:
pytest tests/test_bot_protocol.py::test_name
```

**Run the bot (stdin/stdout Mjai protocol):**
```bash
python src/main_bot.py --player-id 0 --checkpoint artifacts/sl/best.npz
```

**Supervised learning training:**
```bash
python -m train.train_sl --data-dir dataset/mjai --out-dir artifacts/sl
```

**Debug bot policy output:**
```bash
BOT_DEBUG_POLICY=1 python src/main_bot.py --player-id 0
```

## Architecture

`src/` is the package root (set via `pyproject.toml` `package-dir = {"" = "src"}` and pytest `pythonpath`).

### Inference flow

`src/main_bot.py` is the entry point for Mjai protocol inference. It reads JSON lines from stdin and writes responses to stdout. `MjaiPolicyBot` (`src/bot/mjai_bot.py`) owns the full inference loop:
1. Maintains `GameState` (from `mahjong_env/state.py`) and applies each incoming event via `apply_event`.
2. Calls `enumerate_legal_actions` to get valid moves.
3. Vectorizes state with `vectorize_state_py` (`bot/features.py`, OBS_DIM=270) — pure Python, no numpy, to work inside the mjai Docker environment.
4. Runs `_NumpylessPolicyValueModel` (an embedded pure-Python inference clone inside `mjai_bot.py`) and selects the highest-scoring legal action.
5. Falls back to `rule_bot.fallback_action` for edge cases.

Checkpoints are saved as `.npz` (numpy, for training/dev) and exported to `.json` (numpy-free, for production Docker deployment). The bot prefers `.json` at the same path.

### Model

`src/model/policy_value.py` — `MultiTaskModel` (aliased as `PolicyValueModel`) wraps `ResNetEncoder` (two residual blocks, default hidden_dim=256) and adds three heads:
- **Policy logits** over the action vocabulary
- **Value scalar** (expected placement)
- **Auxiliary logits** (secondary supervision)

All weights are plain numpy arrays; there is no PyTorch/JAX dependency.

### Training

`src/train/train_sl.py` — supervised learning from Mjai JSONL logs. Uses `SupervisedMjaiDataset` and `build_supervised_samples` to replay game logs and extract (observation, action label) pairs. Training uses advantage-weighted policy gradient (`_weighted_policy_grad`) blended with cross-entropy.

`src/train/train_rl.py` — reinforcement learning loop against self-play or other bots.

### Data pipeline

Raw logs (Tenhou `.xml` or Majsoul) → `src/convert/` (via `libriichi_bridge.py` which calls the Rust `libriichi` library from `third_party/Mortal`) → Mjai JSONL → `src/train/dataset.py`.

`third_party/Mortal` is a git submodule containing the Rust `libriichi` crate, used only for data conversion. It is not required at inference time.

### Key data contract

The observation vector layout is documented in `src/bot/features.py`:
- `[0:34]` own hand histogram
- `[34:170]` all-player discard histograms
- `[170:174]` actor one-hot
- `[174:182]` riichi flags, scores
- `[182:268]` round info, last discard, dora, meld/discard counts, oya id

Action vocabulary is built by `src/model/vocab.py` and serialized inside the checkpoint.

## Notes

- All modules import with `src/` as root (e.g., `from bot.mjai_bot import ...`), not `from src.bot...`.
- Checkpoint format: `.npz` for development, `.json` for production Docker (no numpy required at inference).
- Some tests require actual Mjai log files in `dataset/`.
- Avoid importing packages that conflict with ROS system packages.
