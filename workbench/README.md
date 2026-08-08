# Workbench

This directory owns the control-plane application:

- `gateway/`, `participants/`, and `replay/` — backend/API behavior;
- `replay_ui/` — React interface;
- `convert/`, `tools/`, `scripts/`, and `configs/` — Workbench operations.

Run it with `uv run python workbench/main.py local --port 8000`.
Runtime state is rooted at `KEQING_DATA_ROOT`, which defaults to the sibling
repository directory `data/`.
