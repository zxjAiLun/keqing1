# Rust Progress Status

Current status: **this refactor round is complete and verified**.

## Completed scope

Implemented:
- `keqing_core` wheel build and package rewrite via `rust/keqing_core/build.py`
- platform-neutral native extension discovery for both source-shadowing and installed-wheel mode
- package-local native extension layout:
  - `keqing_core/__init__.py`
  - `keqing_core/_native*.so|pyd`
- unified `keqing_core` shanten integration for `keqingv3.progress_oracle`
- batch native API `keqing_core.standard_shanten_many(...)`
- explicit `3n+2` native candidate-summary seam:
  - Rust expands ordered discard candidates and returns scalar candidate summaries
  - Python retains ranking, fallback, cache ownership, and final `NormalProgressInfo` construction
- Rust crate layering split:
  - `counts.rs`
  - `standard.rs`
  - `progress_batch.rs`
  - `py_module.rs`
- regression coverage for:
  - `calc_standard_shanten_from_counts()`
  - `keqing_core.standard_shanten_many(...)`
  - `analyze_normal_progress_from_counts()`
  - rust toggle round-trip
  - `3n+2` parity
  - `3n+2` tie-order stability
  - stale native seam guard
- benchmark script for Python vs Rust-enabled paths with split `3n+1` / `3n+2` reporting

## Out of scope

- no changes to `features.py` responsibilities
- no changes to `feature_tracker.py` responsibilities
- no changes to battle/replay/selfplay truth paths
- no replacement of `riichienv` shanten semantics

## Validation completed

- `cargo test`
- `uv run python -m py_compile src/keqing_core/__init__.py src/keqingv3/progress_oracle.py tests/test_progress_oracle_rust.py scripts/benchmark_progress_oracle_rust.py rust/keqing_core/build.py`
- `PYTHONPATH=src uv run pytest tests/test_progress_oracle_rust.py tests/test_normal_progress.py tests/test_features_v3.py`
- `uv run python rust/keqing_core/build.py`
- `uv pip install --reinstall rust/keqing_core/target/wheels/keqing_core-*.whl`
- wheel install and import verification in both installed-wheel and source-shadowing mode

## Latest measured result

- `calc_standard_shanten_from_counts()`: about `2.22x`
- `analyze_normal_progress_from_counts()`: about `1.70x`
- `3n+1`: about `1.65x`
- `3n+2`: about `1.56x`

## Interpretation

- The main fragility in this line was packaging/discovery and partial-native fallback, not gameplay correctness
- `progress_oracle` now treats Rust as a capability layer rather than embedding policy into the extension
- The `3n+2` path is fast enough to stop this round without widening scope into a larger Phase 3 migration
- If future work resumes, the next likely target is training/preprocessing reuse of the same capability seam, not more helper-by-helper substitutions
