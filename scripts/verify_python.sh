#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[verify_python] uv is required but not found in PATH" >&2
  exit 1
fi

# In CI we want a deterministic synced environment before executing checks.
if [[ "${CI:-}" == "true" || "${VERIFY_SYNC:-0}" == "1" ]]; then
  uv sync --locked --group dev
fi

uv run python -m py_compile \
  src/keqing_core/__init__.py \
  src/keqingv3/progress_oracle.py \
  tests/test_progress_oracle_rust.py \
  scripts/benchmark_progress_oracle_rust.py \
  rust/keqing_core/build.py

cargo test --manifest-path rust/keqing_core/Cargo.toml

uv run python rust/keqing_core/build.py
WHEEL_PATH="$(
  ls -t rust/keqing_core/target/wheels/keqing_core-*.whl \
    | grep -v '\.rewritten\.whl$' \
    | head -n 1
)"
if [[ -z "$WHEEL_PATH" || ! -f "$WHEEL_PATH" ]]; then
  echo "[verify_python] no installable keqing_core wheel found" >&2
  exit 1
fi
uv pip install --reinstall "$WHEEL_PATH"

PYTHONPATH=src uv run pytest \
  tests/test_progress_oracle_rust.py \
  tests/test_normal_progress.py \
  tests/test_features_v3.py
