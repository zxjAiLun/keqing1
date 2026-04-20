# Keqing1

Keqing1 is organized around two active model lines plus shared semantic/runtime infrastructure:

- `xmodel1`: current training-window mainline
- `keqingv4`: current backup line
- `rulebase`: compatibility baseline for runtime/replay surfaces

Legacy model families are no longer active runtime or training targets.

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

## Current Entry Points

Run local replay/review service:

```bash
uv run python src/main.py local --port 8000
```

Run gateway only:

```bash
uv run python src/main.py --gateway-port 11600 tenhou
```

Run local service plus gateway:

```bash
uv run python src/main.py --port 8000 --gateway-port 11600 serve
```

Launch live bots:

```bash
uv run python scripts/launch_tenhou_bots.py --room L2147 --count 1 --bot xmodel1 --start-gateway
```

Supported bot names on active surfaces are:
- `xmodel1`
- `keqingv4`
- `rulebase`

## Current Data Flow

Stable raw replay roots:
- `artifacts/converted_mjai/ds1`
- `artifacts/converted_mjai/ds2`
- `artifacts/converted_mjai/ds3`

Model-specific preprocess outputs are artifacts, not source-of-truth datasets.

Current mainline flow:

```text
converted mjai replays
-> preprocess_xmodel1.py
-> processed_xmodel1/*
-> train_xmodel1.py
-> artifacts/models/xmodel1
-> runtime/review/slice evaluation
```

Current xmodel1 contract notes:
- active schema: `xmodel1_discard_v3`
- active context field: `history_summary[20]`
- active candidate dims: `candidate=22`, `flag=8`, `special=19`
- training resume requires full v3 checkpoint metadata; old no-`cfg` checkpoints are runtime-loadable only
- runtime tensor construction enters through `keqing_core.build_xmodel1_runtime_tensors(...)` and may fall back only when the native capability is missing

Backup flow:

```text
converted mjai replays
-> preprocess_keqingv4.py
-> processed_v4/*
-> train_keqingv4.py
-> artifacts/models/keqingv4
```

## Key Directories

- `src/mahjong_env/`: shared Mahjong semantics and neutral action/progress utilities
- `src/training/`: shared training infrastructure and neutral state feature encoders
- `src/xmodel1/`: current mainline model family
- `src/keqingv4/`: current backup model family
- `src/inference/`: runtime adapters and bot entrypoints
- `src/replay/`: replay API, storage, rendering, and review entrypoints
- `rust/keqing_core/`: Rust semantic core
- `tests/`: parity, regression, and focused model tests
- `docs/`: design docs, status boards, and execution notes

## Tracking

Read these first when resuming work:
1. `AGENTS.md`
2. `docs/project_progress.md`
3. `docs/agent_sync.md`
4. the latest `docs/todo_*.md`

`.omx` is retired and should not be used.

## Verification

Typical focused validation commands:

```bash
uv run pytest tests/test_inference_contracts.py tests/test_progress_oracle_rust.py -q
uv run pytest tests/test_keqingv4_preprocess.py tests/test_keqingv4_train_script.py -q
uv run pytest tests/test_bot_model_version.py tests/test_selfplay_rulebase_seats.py -q
```
