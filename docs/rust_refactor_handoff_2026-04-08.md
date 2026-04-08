# Rust Refactor Handoff (2026-04-08)

## Current State

Rust refactor remains scoped only to the `keqingv3` progress-analysis path, and **this round is now in a good stopping state**.

Implemented and working:
- `rust/keqing_core/` exists and builds a wheel through `rust/keqing_core/build.py`
- installed wheel layout is:
  - `keqing_core/__init__.py`
  - `keqing_core/_native.<platform-extension>`
- `src/keqing_core/__init__.py` supports:
  - source-development mode (`PYTHONPATH=src`) while loading the installed native extension
  - installed-wheel mode
  - optional Rust enable/disable switch via `keqing_core.enable_rust(True/False)`
- `src/keqingv3/progress_oracle.py` now uses `keqing_core` as the unified shanten integration layer
- batch native API exists:
  - `keqing_core.standard_shanten_many(...)`
- explicit `3n+2` native candidate-summary seam exists:
  - Rust expands ordered candidate discards and returns scalar candidate summaries
  - Python retains final ranking, cache ownership, fallback, and `NormalProgressInfo` materialization
- hottest standard-shanten loops inside `_summarize_3n1_cached(...)` already use the batch path
- Rust crate internals are now layered into:
  - `counts.rs`
  - `standard.rs`
  - `progress_batch.rs`
  - `py_module.rs`

Not changed:
- `src/keqingv3/features.py`
- `src/keqingv3/feature_tracker.py`
- battle / replay / selfplay truth paths
- `riichienv` remains the shanten semantics baseline

## Verified Commands

Validated successfully on the current Linux + `uv` workflow:

```bash
cd /media/bailan/DISK1/AUbuntuProject/project/keqing1/rust/keqing_core
cargo test

cd /media/bailan/DISK1/AUbuntuProject/project/keqing1
uv run python rust/keqing_core/build.py
uv pip install --reinstall rust/keqing_core/target/wheels/keqing_core-*.whl
uv run python -m py_compile src/keqing_core/__init__.py src/keqingv3/progress_oracle.py tests/test_progress_oracle_rust.py scripts/benchmark_progress_oracle_rust.py rust/keqing_core/build.py
PYTHONPATH=src uv run pytest tests/test_progress_oracle_rust.py tests/test_normal_progress.py tests/test_features_v3.py
PYTHONPATH=src uv run python scripts/benchmark_progress_oracle_rust.py
```

Latest observed results:
- `cargo test`: pass
- `pytest`: `22 passed`
- benchmark:
  - `calc_standard_shanten_from_counts()`: about `2.22x`
  - `analyze_normal_progress_from_counts()`: about `1.70x`
  - `3n+1`: about `1.65x`
  - `3n+2`: about `1.56x`

## Important Files

- `rust/keqing_core/build.py`  
  The only supported wheel build / rewrite entrypoint.
- `src/keqing_core/__init__.py`  
  Unified Python wrapper for native loading, seam capability checks, and runtime switching.
- `rust/keqing_core/src/py_module.rs`  
  PyO3 bindings, including batch shanten API and `3n+2` candidate summaries.
- `rust/keqing_core/src/progress_batch.rs`  
  Rust-side ordered candidate expansion / summary seam for `3n+2`.
- `src/keqingv3/progress_oracle.py`  
  Current integration target; Python still owns policy/ranking/fallback.
- `tests/test_progress_oracle_rust.py`  
  Rust/Python parity coverage for current integration, including stale-seam and tie-order regressions.
- `scripts/benchmark_progress_oracle_rust.py`  
  Current benchmark entrypoint.

## What Was Cleaned Up

Removed old agent-generated scripts that were no longer useful or were based on outdated assumptions:
- `scripts/align_shanten.py`
- `scripts/compare_shanten.py`
- `scripts/validate_riichi.py`
- `scripts/profile_progress.py`
- `scripts/profile_features.py`
- `scripts/profile_features_detail.py`
- `scripts/profile_features_real.py`

These should not be reintroduced.

## Recommended Next Step

This round can stop here.

If future work resumes, prefer one of these directions:
- stabilize the capability seam for training / preprocessing reuse
- decide whether Rust should remain opt-in or become the default on supported Linux environments
- only then consider deeper `3n+1` extraction if there is a measured win-rate or preprocessing-cost reason

## Explicit Do-Not-Do List

- Do not introduce `third_party/Mortal/libriichi` as a runtime dependency
- Do not create a parallel `src/libriichi/`
- Do not change battle / replay / selfplay truth logic in this line of work
- Do not replace `riichienv` semantics in this stage
- Do not go back to manual `site-packages` patching
- Do not resume helper-by-helper Rust substitution unless a benchmark proves it is better than the batch path
- Do not move final `3n+2` ranking / `NormalProgressInfo` assembly into Rust without a new planning pass

## Workspace Notes

- There is an unrelated `SPEC.md` deletion in `git status`; this was not part of the Rust refactor work
- Current meaningful worktree items for this line are mainly under:
  - `rust/keqing_core/`
  - `src/keqing_core/`
  - `src/keqingv3/progress_oracle.py`
  - `tests/test_progress_oracle_rust.py`
  - `scripts/benchmark_progress_oracle_rust.py`
