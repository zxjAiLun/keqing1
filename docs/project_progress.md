# Keqing1 Project Progress

Updated: 2026-04-20

This is the primary status board for the repository.

## Current Global Judgment
- The project remains focused on building a lightweight but strong Mahjong model.
- `xmodel1` is the current training-window mainline.
- `keqingv4` is the current backup line and Rust-ownership repair path.
- Raw replay truth datasets remain `artifacts/converted_mjai/ds1`, `ds2`, and `ds3`.
- `.omx` is retired. Current tracking lives in `docs/project_progress.md`, `docs/agent_sync.md`, and the latest dated `docs/todo_*.md`.
- Legacy model families have been removed from active runtime/training code paths.
- Legacy model families and their dedicated configs/scripts have also been removed from the active tree; shared neutral modules are the only maintained import surfaces.
- This cleanup batch has also archived legacy runtime docs and deploy templates, retired `processed_v3` to `processed_v3.retired`, and removed old cache roots from active defaults.

## Active Workstreams

### P0: xmodel1 Mainline
- Goal: keep `xmodel1_discard_v2` stable and ready for user-executed preprocess, smoke train, slice review, and runtime validation.
- Current code-side status:
  - Rust-first preprocess/cache path is in place.
  - Production preprocess ownership is fully in Rust via `scripts/preprocess_xmodel1.py` and `keqing_core.build_xmodel1_discard_records(...)`.
  - `xmodel1` export now preserves `hora` as a dedicated special sample type end-to-end (Rust export + Python parity oracle + review category), so terminal win decisions are no longer dropped before training.
  - The preprocess launcher now runs a random-file preflight export/probe gate per shard before the full export, so per-file cache contract drift fails fast instead of surfacing only after the full run.
  - Existing Rust exports that only missed per-file `schema_name/schema_version` metadata can now be repaired in place via `scripts/repair_xmodel1_cache_schema.py`; they do not require a full recompute.
  - The default `xmodel1` train config now uses ratio-based random file sampling (`files_per_epoch_ratio`) instead of the old fixed `files_per_epoch_count=350`, so the per-epoch slice scales with the Rust-exported dataset.
  - The default `xmodel1` train config no longer pins `steps_per_epoch`; when the config value is `<= 0`, the launcher now auto-derives epoch steps from the sampled file slice and current `batch_size`.
  - Candidate-quality and special-candidate summaries have been recalibrated around after-state value proxy, yaku-break, and risk terms, and focused minimal regressions now pin `simple_discard / riichi / pon_call`.
  - `src/xmodel1/preprocess.py` remains only as a Python parity oracle used by contract tests; it is no longer a production export or manifest-writing fallback.
  - Candidate feature parity, slice-report harness, preprocess gates, and train-time gates are in place.
- User-owned execution remains:
  - full preprocess on `ds1 + ds2 + ds3`
  - smoke train
  - review / slice acceptance / strength judgment

### P1: keqingv4 Backup
- Goal: continue Rust ownership of preprocess, typed summaries, and runtime projection without consuming the main training window.
- Current role:
  - backup line if xmodel1 hits kill criteria
  - shared-core and Rust semantic consolidation track
- Current code-side status:
  - Phase A train-ready gate is green.
  - Phase B is no longer only a direction note; part of the code-side landing is already in place.
  - Bounded future-truth has been wired into the keqingv4 summary/value path, with recursive strengthening capped at `shanten <= 2` and non-recursive behavior kept above that cutoff.
  - Shared Rust continuation helpers have landed for scenario construction, per-scenario scoring, and weighted aggregation; `src/inference/scoring.py` now prefers the Rust continuation path and keeps Python only as a missing-capability fallback.
  - Hora truth and continuation/scoring bridges are now fail-closed on unexpected native drift; Python emergency code is no longer allowed to silently absorb schema or semantic errors on those paths.
  - keqingv4 runtime/preprocess dependencies have been pulled off legacy `keqingv1` / `keqingv3` feature surfaces onto shared active surfaces such as `mahjong_env.action_space` and `training.state_features`.
  - preprocess and evidence scripts have been updated alongside that shift, including legacy `workers` config compatibility in `scripts/preprocess_keqingv4.py`, a new random preflight gate that samples a few real `.mjson` files before full export, and calibration harnesses that now consume the shared action-space surface.
  - Focused coverage exists on the active path: preprocess parity, inference adapter behavior, model/training smoke, and continuation-contract tests all have corresponding test surfaces in the current tree.
  - `keqingv4` can re-enter the training window if xmodel1 hits kill criteria, but it remains the backup line by default.

### P2: Shared Rust Core
- Shared priorities across `xmodel1` and `keqingv4`:
  - hora/scoring truth consolidation
  - event-history emission
  - opponent tenpai labels
  - after-state hand-value and summary primitives
- Current code-side status:
  - replay sample public ownership is now Rust-only
  - continuation scoring is moving into shared Rust pure helpers
  - public legal-action enumeration now prefers a Rust public bridge end-to-end; Python keeps only missing-capability fallback and compatibility shaping on that path
  - `DefaultDecisionContextBuilder` now prefers Rust replay snapshots for runtime/model decision snapshots and falls back to Python `state.snapshot(actor)` only when the replay-snapshot capability is missing
  - `mahjong_env.scoring.py` now treats Rust hora truth as the public owner path and allows Python scoring fallback only for missing capability, not for unexpected truth drift
  - legacy preprocess helper surfaces are no longer active defaults
  - a new multi-session execution plan for the next Rust-ownership push is now frozen at `plans/rust_ownership_push_2026_04_19.md`
  - that plan keeps the current priority order intact and sequences the next push as:
    - `S0` boundary freeze and guardrails
    - `S1` keqingv4 fail-closed cleanup
    - `S2` legal action public-owner consolidation
  - current Python boundary notes:
    - parity-only oracle: `src/xmodel1/preprocess.py`
    - Rust-first shell / emergency-fallback surfaces: `src/mahjong_env/replay.py`, `src/mahjong_env/legal_actions.py`, `src/keqingv4/preprocess_features.py`, `src/inference/scoring.py`, `src/mahjong_env/scoring.py`
    - remaining active Python semantic-owner surfaces for the next Rust push: `src/mahjong_env/state.py`, `src/training/state_features.py`, `src/xmodel1/features.py`, `src/xmodel1/candidate_quality.py`
  - `S0` guardrails now explicitly require fail-closed tests for typed-summary shape drift, continuation scenario schema drift, continuation scoring schema drift, continuation aggregation schema drift, and hora-truth fallback boundaries
- Rule: shared Rust semantics should land in common core first, then be consumed by model-specific export/runtime layers.

### P3: Legacy Cleanup Follow-Through
- Legacy runtime docs now live under `docs/archive/legacy_runtime/`.
- Legacy deploy templates now live under `deploy/archive/legacy_models/`.
- Removed legacy active-tree surfaces now include `src/keqingv1/`, `src/keqingv2/`, `src/keqingv3/`, `src/keqingv31/`, their dedicated preprocess/train scripts, and their old default configs.
- Active deploy/runtime naming is now:
  - `xmodel1`
  - `keqingv4`
  - `rulebase`
- `processed_v3` has been retired to `processed_v3.retired`; active configs and scripts should not point at it anymore.
- The remaining follow-through is observational:
  - keep archive-only policy stable
  - do not restore legacy aliases or service names
  - permanently delete `processed_v3.retired` after a short observation window if no consumer resurfaces

## Current Priority Order
1. xmodel1 preprocess/train/review readiness
2. xmodel1 architecture stability and evaluation quality
3. keqingv4 preprocess/runtime Rust ownership
4. shared Rust semantic core consolidation
5. compatibility-surface cleanup around replay/runtime/gateway/UI

## Current Risks
- The user still needs to execute the real preprocess, train, and review loops; code-side gates alone do not prove model quality.
- Existing `xmodel1` caches exported before the dedicated `hora` sample-type change do not contain those rows; metadata repair cannot synthesize them, so a real re-export is required before retraining.
- Some historical docs still mention old model lines for comparison or historical context. They should not be treated as active runbooks.
- Rust migration is not complete; the remaining work is no longer old-line compatibility, but shared semantic ownership and long-tail continuation/scoring cleanup.
- `xmodel1` still carries a deliberate Python parity oracle for candidate arrays, and focused contract tests remain the place where real Rust/Python drift is measured.
- The current local Python environment still lacks `torch`, which blocks some gateway/runtime focused tests here.

## Near-Term Next Actions
1. User reruns `xmodel1` full preprocess gate on `ds1 + ds2 + ds3`; old exports must be replaced rather than schema-repaired if they were built before the dedicated `hora` sample-type change.
2. User runs `xmodel1` smoke train after preprocess passes.
3. User runs slice/review acceptance on `reach / call / none / hora / chi-position` slices.
4. Keep `keqingv4` on non-training Rust-ownership work by default; only switch it back into preprocess/train/eval if `xmodel1` hits kill criteria.
5. For `keqingv4`, the next concrete code-side move is Phase B verification rather than new structure: rerun the focused preprocess/inference/special-calibration suites against the current Rust continuation/future-truth path, then freeze that as the current backup snapshot.
6. Start the next shared Rust push from `plans/rust_ownership_push_2026_04_19.md`, but keep the first execution slice bounded to:
   - `S0` boundary freeze and guardrails
   - `S1` keqingv4 fail-closed cleanup
   - `S2` legal action public-owner consolidation
7. If `processed_v3.retired` stays unused after observation, delete it permanently in the next cleanup pass.
