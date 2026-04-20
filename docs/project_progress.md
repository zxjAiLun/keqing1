# Keqing1 Project Progress

Updated: 2026-04-20

This is the primary status board for the repository.

## Executive Summary
- The repository is in a model-window execution phase, not a legacy runtime maintenance phase.
- `xmodel1` remains the only active mainline for the current training window, and its code-side public contract is frozen on `xmodel1_discard_v3`.
- The current xmodel1 blocker is no longer architecture churn or schema uncertainty; it is the missing user-run evidence loop: full preprocess, smoke train, and slice review.
- `keqingv4` remains a frozen backup line plus Rust-ownership repair path. Its near-term work is verification and boundary hardening, not opening a competing training window.
- Shared Rust work should continue only insofar as it hardens public semantic ownership and fail-closed boundaries without consuming the mainline training window.
- Replay/runtime/bot surfaces should be treated as compatibility layers around model work, not as independent product priorities.

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
- Goal: keep `xmodel1_discard_v3` as the only active xmodel1 contract, then reopen full preprocess, smoke train, slice review, and runtime validation on that frozen boundary.
- Current code-side status:
  - Rust-first preprocess/cache path is in place.
  - Production preprocess ownership is fully in Rust via `scripts/preprocess_xmodel1.py` and `keqing_core.build_xmodel1_discard_records(...)`.
  - The mainline public schema identity is now `xmodel1_discard_v3` / `schema_version = 3`.
  - Raw public `event_history` has been removed from the xmodel1 cache/runtime contract and replaced with fixed-shape `history_summary[20]`.
  - The old `legacy_kakan2` drift was rooted in raw-history slicing on pre-normalized events while parity used normalized events; xmodel1 now builds history context from normalized events on both sides, so the legacy whitelist is no longer part of the target state.
  - Candidate and special candidate fields have been compressed to the smaller v3 slot tables; v2-only rank buckets and `score_delta_target` are no longer part of the public cache contract.
  - The model/runtime/train path now consumes `history_summary` directly; the old xmodel1 history transformer path is no longer the target architecture.
  - The Python parity oracle, cached loader, train script smoke fixtures, and adapter/runtime batch helpers now all gate on the same v3 field set.
  - `xmodel1` training now runs on the shared `training.train_model(...)` epoch loop; xmodel1 keeps a thin task wrapper for its candidate/special/value losses plus metadata/log shaping.
  - New xmodel1 checkpoints now write full metadata (`model_version`, `cfg`, schema and tensor dims, hidden size, block count, dropout); runtime keeps read-only legacy loading for old no-`cfg` checkpoints but resume stays fail-closed across the schema cutover.
  - `RuntimeBot`/`KeqingModelAdapter` now infer legacy xmodel1 structure from weight shapes instead of assuming default hidden depth.
  - Shared beam continuation scoring now uses batched `forward_many(...)` when the adapter exposes it, reducing repeated per-candidate forwards without changing decision semantics.
  - The xmodel1 runtime tensor entrypoint is now centralized behind `keqing_core.build_xmodel1_runtime_tensors(...)`, and the native Rust implementation is now the active owner for runtime candidate/special/history-summary tensors when the extension is available; Python fallback remains allowed only for missing capability on that path.
  - The preprocess launcher still runs a preflight export/probe gate per shard, and the code-side v3 contract freeze plus fixed-subset profile gate are now green.
  - Xmodel1 discard candidate analysis has been deliberately lowered again on the v3 path: special-waits fallback, one-shanten draw metrics, and pinfu/iipeikou break tracking no longer dominate preprocess compute, while the public v3 schema stays unchanged.
  - Fixed-subset evidence on `ds1 + ds2 + ds3` (15 files / 8843 samples) is now:
    - `total_wall=1.386s`
    - `discard_candidate_analysis=1.891s`
    - `npz_write=9.670s`
    - compared with the earlier profile on the same subset:
      - `total_wall=53.048s`
      - `discard_candidate_analysis=545.964s`
  - Rust export now has env-gated stage profiling (`XMODEL1_EXPORT_PROFILE=1`) plus cooperative interrupt/resume work, so the next evidence loop can measure real wall-time and stop safely.
  - `src/xmodel1/preprocess.py` remains only as a Python parity oracle used by contract tests; it is not a production exporter.
- User-owned execution remains:
  - full preprocess on `ds1 + ds2 + ds3` only after the gate passes
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

## Continuation Guidance
- The next meaningful project-level state change should come from evidence, not from more structural churn.
- Treat the following as the current acceptance chain for the mainline:
  1. rerun full preprocess on `ds1 + ds2 + ds3` with the frozen v3 contract
  2. run xmodel1 smoke train on the new export
  3. review slice quality on `reach / call / none / hora`
- Until those three steps complete, the correct global summary is "code-side mainline is ready for evidence collection", not "model quality is already proven".
- Any new Rust or backup-line work should be kept bounded enough that it does not delay that evidence chain.

## Current Risks
- The user still needs to execute the real preprocess, train, and review loops; code-side gates alone do not prove model quality.
- Existing `xmodel1` caches exported before the dedicated `hora` sample-type change do not contain those rows; metadata repair cannot synthesize them, so a real re-export is required before retraining.
- `xmodel1_discard_v2` is no longer the rerun target. Any old v2 cache or checkpoint must now fail closed and be replaced rather than silently reused.
- The new v3 contract is now aligned across Python loader/runtime/train surfaces, but a real full re-export is still mandatory before retraining because production preprocess remains Rust-owned and old caches cannot be upgraded in place.
- Some historical docs still mention old model lines for comparison or historical context. They should not be treated as active runbooks.
- Rust migration is not complete; the remaining work is no longer old-line compatibility, but shared semantic ownership and long-tail continuation/scoring cleanup.
- `xmodel1` still carries a deliberate Python parity oracle for candidate arrays, and focused contract tests remain the place where real Rust/Python drift is measured.
- The current local Python environment still lacks `torch`, which blocks some gateway/runtime focused tests here.

## Near-Term Next Actions
1. Run the focused v3 contract suite plus the runtime-recompute consistency cases; do not reopen full preprocess before that is green.
2. Rerun full preprocess on `ds1 + ds2 + ds3` on the current green `xmodel1_discard_v3` contract, then smoke train, then slice/review acceptance on `reach / call / none / hora`.
3. Keep `XMODEL1_EXPORT_PROFILE=1` available for spot checks on resumed shards or any future preprocess regression report; the fixed-subset gate is no longer the blocker.
4. Keep `keqingv4` on non-training Rust-ownership work by default; only switch it back into preprocess/train/eval if `xmodel1` hits kill criteria.
5. For `keqingv4`, the next concrete code-side move is Phase B verification rather than new structure: rerun the focused preprocess/inference/special-calibration suites against the current Rust continuation/future-truth path, then freeze that as the current backup snapshot.
6. Start the next shared Rust push from `plans/rust_ownership_push_2026_04_19.md`, but keep the first execution slice bounded to:
   - `S0` boundary freeze and guardrails
   - `S1` keqingv4 fail-closed cleanup
   - `S2` legal action public-owner consolidation
7. If `processed_v3.retired` stays unused after observation, delete it permanently in the next cleanup pass.
