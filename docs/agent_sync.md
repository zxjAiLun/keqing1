# Keqing1 Agent Sync

Updated: 2026-04-20

This document defines the minimum shared context for multi-agent work.

## Read Before Work
Before starting a task, read:
1. `AGENTS.md`
2. `docs/project_progress.md`
3. `docs/agent_sync.md`
4. the latest dated snapshot, currently `docs/todo_2026_04_20.md`
5. `git status --short`

If code reality and document wording disagree, trust the latest code and evidence first, then update the documents.

## Current Alignment
- `xmodel1` is the training-window mainline.
- `keqingv4` is the backup line and Rust-ownership repair track.
- Preprocess, training, review, and model-strength acceptance are executed by the user.
- Agents should prioritize code alignment, Rust semantic ownership, contract hardening, gates, harnesses, and documentation sync.
- `xmodel1` preprocess/review semantic contract now includes a dedicated `hora` special sample type; terminal win actions are no longer allowed to exist only in runtime legal sets while being absent from training caches.
- Public legal actions, replay snapshots, and hora-truth/scoring are now Rust-first public surfaces; Python fallback is allowed only for missing capability, not for unexpected bridge drift.
- `.omx` is retired. Do not read from it, write to it, or treat it as memory.
- Legacy model families have been removed from active runtime/training surfaces. Do not restore legacy aliases or compatibility imports.
- Legacy runtime notes and deploy templates are archive-only surfaces now:
  - `docs/archive/legacy_runtime/`
  - `deploy/archive/legacy_models/`

## Coordination Rules
- When changing `src/inference`, training sample construction, or review/export behavior, check the other two surfaces for contract drift.
- When changing public legal actions, runtime snapshots, or scoring/hora truth, preserve fail-closed behavior for unexpected Rust bridge drift; missing capability is the only valid fallback trigger.
- If a change alters `xmodel1` cache row semantics rather than only metadata, do not treat `scripts/repair_xmodel1_cache_schema.py` as sufficient evidence; document whether a real re-export is mandatory.
- Do not overwrite user changes or unrelated dirty worktree changes.
- Keep the xmodel1-mainline / keqingv4-backup boundary stable unless the status board and design docs are updated together.
- Do not point active configs, scripts, or docs back at `processed_v3` or other retired preprocess roots.
- Do not reintroduce imports from removed legacy model packages; active shared surfaces are `mahjong_env.action_space`, `mahjong_env.feature_tracker`, `mahjong_env.progress_oracle`, and `training.state_features`.

## When To Update Status
Update status documents when any of the following changes:
- mainline/backup judgment
- a global blocker is added or resolved
- an experiment becomes a real workflow
- training/inference/review semantic contracts change
- a new cross-agent risk appears
- a historical surface is archived, retired, or reactivated

## Write-Back Order
1. change code or gather evidence
2. update `docs/project_progress.md`
3. update `docs/agent_sync.md` if coordination rules changed
4. update the latest `docs/todo_YYYY_MM_DD.md` snapshot if daily execution status changed

## Document Roles
- `AGENTS.md`: repository-level execution rules and priorities
- `docs/project_progress.md`: current project status, workstreams, priorities, and risks
- `docs/agent_sync.md`: coordination rules and handoff expectations
- `docs/todo_YYYY_MM_DD.md`: dated execution slices and near-term sequencing

## Minimum Handoff Output
When an agent finishes a work chunk, leave:
- what changed
- what evidence was collected
- what is still blocked or unblocked
- the next most reasonable action
