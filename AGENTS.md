# Repository Guidelines

## Project Structure & Module Organization
`src/` contains runtime code. Key modules are `src/gateway/` for live battle APIs and bot turns, `src/mahjong_env/` for rules/state/legal actions, `src/replay/` for offline replay generation, and `src/replay_ui/` for the React/Vite frontend. Models and generated outputs live under `artifacts/`. Dataset inputs are under `dataset/`. Tests are in `tests/v5model/`. Historical code remains in `archive/`; prefer current `src/` implementations for new work.

## Build, Test, and Development Commands
- `uv run pytest` runs the Python test suite configured in `pyproject.toml`.
- `uv run pytest tests/v5model/test_battle_meld_flow.py` runs a focused backend regression.
- `uv run python -m py_compile src/gateway/battle.py src/gateway/api/battle.py` performs quick syntax validation.
- `cd src/replay_ui && npm install` installs frontend dependencies.
- `cd src/replay_ui && npm run dev` starts the Vite dev server.
- `cd src/replay_ui && npm run build` performs a production frontend build.
- `cd src/replay_ui && npm run lint` runs ESLint.

## Coding Style & Naming Conventions
Use 4-space indentation in Python and follow existing type-hinted, dataclass-heavy style. Keep Python modules `snake_case`, classes `PascalCase`, and React components/files `PascalCase` where appropriate. Prefer small, explicit state helpers over implicit UI logic. In frontend code, preserve the existing inline-style and utility-driven patterns unless a file already uses a different convention.

## Testing Guidelines
Backend tests use `pytest`. Add regressions beside related files in `tests/v5model/`, named `test_<feature>.py` and `test_<behavior>()`. Frontend changes should at minimum pass `npm run build`; add lint validation for larger UI refactors. For battle/replay UI work, verify both live `battle` and `game-replay` flows when changing shared state fields.

## Commit & Pull Request Guidelines
Recent history favors short imperative subjects such as `add replay UI game board view & diff navigation buttons` and `refactor: archive deprecated code, reorganize project structure`. Use concise commits scoped to one change. PRs should include:
- a brief problem/solution summary
- affected modules or routes
- test/build commands run
- screenshots or short recordings for UI changes

## Architecture Notes
Treat `mahjong_env` and `gateway` as state truth. Frontend rendering should consume serialized state rather than re-deriving game rules locally. When adding UI animation states, prefer backend-provided fields if timing must match real game transitions.

Operational runbook for the selfplay / battle / replay line lives at:
- `docs/selfplay_battle_replay_runbook.md`

## Battle, Selfplay, and Replay Invariants
- Treat `src/gateway/battle.py` + `src/mahjong_env/state.py` + `src/mahjong_env/legal_actions.py` as the only gameplay truth. `riichienv` may still be used for auxiliary shanten/waits analysis, but it must not become a second rules source for live battle or selfplay flow.
- Keep `scripts/selfplay.py` as a thin orchestrator. It must not manually mutate `room.state` to simulate draws, melds, scores, or progression. All gameplay mutations must go through `BattleManager`.
- Every emitted event should be consumed by each bot at most once. Do not re-broadcast the same semantic event through multiple ad hoc paths. When refactoring selfplay, preserve the invariant that bot-local `GameState` and server `room.state` stay in sync.
- When applying bot actions to the server, canonicalize against server-side legal actions first. This is especially important for red fives (`5mr/5pr/5sr`) and meld `consumed` tiles. The bot may choose a normalized action; the server must execute the exact legal representation from current state.
- Discard response arbitration must preserve Mahjong priority:
  - `hora > pon/daiminkan > chi`
  - for equal priority, the player closer in turn order after the discarder wins
- `kakan` is a two-step flow:
  - declaration event `kakan`
  - response window for `chankan`
  - acceptance event `kakan_accepted` only if nobody wins by robbing the kan
  Do not collapse this back into a single immediate state mutation.
- Legal `hora` must require real yaku, not just waits matching. If changing `legal_actions`, keep `can_hora` / real scoring checks aligned with `BattleManager.hora()`.
- Full-game selfplay defaults to `hanchan`, not single `kyoku`. Preserve renchan, south-four ending, west extension, honba, kyotaku, and agari-yame style termination semantics unless intentionally changing match rules.
- Real scoring is expected. Do not reintroduce fixed-score placeholder resolution in `hora()` or `ryukyoku()`. `mahjong_env.scoring` is the scoring source for han/fu/yaku/cost.
- Selfplay outputs are part of the workflow, not debug-only artifacts. Preserve:
  - default benchmark output under `artifacts/selfplay_benchmarks/<model>_<timestamp>`
  - incremental `stats.json`
  - incremental `progress.jsonl`
  - default saved replay manifests under `replays/manifest.json`
  - optional anomaly exports under `anomaly_replays/manifest.json`
- Replay UI relies on backend aggregation of replay manifests. If moving routes or directories, keep `/api/selfplay/replay-collections` working and make sure both normal `replays` and `anomaly_replays` remain discoverable by the UI.
- When changing fields shared by battle, replay, and UI, verify all three paths:
  - live battle (`/battle` or battle API)
  - replay decision view (`/replay`)
  - game board replay (`/game-replay`)
- Replay accuracy metrics must be computed from backend-normalized replay data, not inferred ad hoc in the UI. In particular, response-window actions may need normalization before scoring:
  - `chosen.type == "none"` with `gt_action == null` and non-`none` response candidates means the real action was an explicit `none`/pass
  - `chosen.type in {"chi","pon","daiminkan","ankan","kakan","hora"}` with `gt_action == null` and a `none` candidate means the real action was that chosen response
  Keep `POST /api/replay` and `GET /api/replay/{id}` on the same normalization path so freshly uploaded reviews and saved replays produce identical stats.
- Frontend review screens should render `实际: —` only for truly unknown ground truth. Do not use `null gt_action` as a silent synonym for “过” or “没有副露”; fix the backend replay normalization instead.
- Hora display must not infer dora-family han from a flattened `yaku` name list alone. Backend scoring should expose structured `yaku_details` including exact `Dora`, `Ura Dora`, and `Aka Dora` han counts, and ReplayUI should prefer that structured data. Frontend fallback inference is only for legacy replay files.
- Replay board semantics for opponent discards must preserve two phases:
  - `pre`: draw completed, tile not yet in river
  - `post`: tile is already in river
  Do not collapse opponent discard replays into a single after-action frame if the board phase controls are expected to show “摸牌后 / 打牌后”.
- Revealed opponent hands in replay mode must model a separate draw slot on the right side instead of sorting the draw tile into the 13-tile hand. After discard, the board should preserve a visible gap:
  - tsumogiri: gap remains in the draw slot
  - tedashi: gap remains in the discarded tile position
- Concealed opponent hands in replay mode should mirror the same timing semantics as revealed hands. If the discarded tile position is unknown, use a stable pseudo-random gap location rather than silently snapping back to a packed 13-tile block.
- Before finishing selfplay/battle/replay changes, run focused regressions. At minimum prefer the current set of battle/selfplay tests in `tests/v5model/`, plus `cd src/replay_ui && npm run build` when changing replay UI or shared replay API shapes.

## Model Iteration Notes
- Treat `src/keqingv1/` and `src/keqingv2/` as compatibility surfaces. Do not change their feature dimensions, model input shapes, or checkpoint expectations when working on new model ideas.
- Build new model iterations under their own module namespace, e.g. `src/keqingv3/`, even if they temporarily reuse code from `keqingv1`. New feature layouts, heads, or input dimensions must not be introduced directly into `keqingv1`.
- Shared training infrastructure belongs in `src/training/` and shared sample/state analysis belongs in `src/mahjong_env/`. Version-specific feature encoders and model definitions belong in the corresponding versioned package.
- When changing shanten, waits, ukeire, or other progress-analysis logic, prefer extending the shared analyzer in `src/mahjong_env/replay.py` and then consuming it from versioned feature modules. Do not maintain separate partially duplicated analyzers in multiple model packages.
- For `keqingv3`, the hot path now lives in `src/keqingv3/progress_oracle.py` plus `src/keqingv3/feature_tracker.py`. Optimize there first before touching training or I/O code.
- `keqingv3` progress-analysis default policy is now:
  - `> 2 shanten`: fast path only, compute standard-shanten-based `ukeire`
  - `2 shanten`: allow lightweight structure-based improvement judgment
  - `1 shanten`: keep explicit `good_shape` enumeration
- Do not reintroduce heavy same-shanten improvement logic above `2 shanten` unless there is a measured win-rate benefit. The preprocessing cost is too high.
- `shape_score` in `keqingv3` is intentionally lightweight and only used for `2 shanten` same-shanten improvement ranking.
- Before attempting Rust, first check whether Python-side tracker reuse, caching, and reduced trigger frequency have already moved preprocessing into an acceptable range.
- Any model-iteration change that affects feature tensors must update:
  - the versioned feature module
  - the matching versioned model module
  - at least one regression test covering tensor shapes and a known reference hand
- Before finishing a model-iteration change, verify both syntax and focused tests for the affected version. At minimum run `uv run python -m py_compile ...` on edited Python files and `uv run pytest` on the relevant regression tests.
