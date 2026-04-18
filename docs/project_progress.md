# Keqing1 Project Progress

Updated: 2026-04-19

This is the primary status board for the repository.

## Current Global Judgment
- The project remains focused on building a lightweight but strong Mahjong model.
- `xmodel1` is the current training-window mainline.
- `keqingv4` is the current backup line and Rust-ownership repair path.
- Raw replay truth datasets remain `artifacts/converted_mjai/ds1`, `ds2`, and `ds3`.
- `.omx` is retired. Current tracking lives in `docs/project_progress.md`, `docs/agent_sync.md`, and the latest dated `docs/todo_*.md`.
- Legacy model families have been removed from active runtime/training code paths.
- This cleanup batch has also archived legacy runtime docs and deploy templates, retired `processed_v3` to `processed_v3.retired`, and removed old cache roots from active defaults.

## Active Workstreams

### P0: xmodel1 Mainline
- Goal: keep `xmodel1_discard_v2` stable and ready for user-executed preprocess, smoke train, slice review, and runtime validation.
- Current code-side status:
  - Rust-first preprocess/cache path is in place.
  - Production preprocess ownership is fully in Rust via `scripts/preprocess_xmodel1.py` and `keqing_core.build_xmodel1_discard_records(...)`.
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
  - Phase B direction is fixed: bounded future-truth remains at `shanten <= 2`, while non-recursive truth strengthening and shared continuation scoring cleanup continue in Rust.
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
  - legacy preprocess helper surfaces are no longer active defaults
- Rule: shared Rust semantics should land in common core first, then be consumed by model-specific export/runtime layers.

### P3: Legacy Cleanup Follow-Through
- Legacy runtime docs now live under `docs/archive/legacy_runtime/`.
- Legacy deploy templates now live under `deploy/archive/legacy_models/`.
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
- Some historical docs still mention old model lines for comparison or historical context. They should not be treated as active runbooks.
- Rust migration is not complete; the remaining work is no longer old-line compatibility, but shared semantic ownership and long-tail continuation/scoring cleanup.
- `xmodel1` still carries a deliberate Python parity oracle for candidate arrays, and focused contract tests remain the place where real Rust/Python drift is measured.
- The current local Python environment still lacks `torch`, which blocks some gateway/runtime focused tests here.

## Near-Term Next Actions
1. User runs `xmodel1` full preprocess gate on `ds1 + ds2 + ds3`.
2. User runs `xmodel1` smoke train after preprocess passes.
3. User runs slice/review acceptance on `reach / call / none / hora / chi-position` slices.
4. Keep `keqingv4` on non-training Rust-ownership work by default; only switch it back into preprocess/train/eval if `xmodel1` hits kill criteria.
5. If `processed_v3.retired` stays unused after observation, delete it permanently in the next cleanup pass.
