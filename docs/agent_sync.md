# Keqing1 Agent Sync

Updated: 2026-04-21

This document defines the minimum shared context for multi-agent work.

## Short Handoff Summary
- The repository should currently be read as a model-building workspace with one active training-window line: `xmodel1`.
- `xmodel1` is already through the current code-side contract freeze; the current full export is available and a first real-cache smoke train on that export is green, so the next missing evidence is slice/review plus broader train coverage, not another schema redesign.
- `keqingv4` is not competing for the training window. It is the backup snapshot and the safer place to continue Rust-ownership cleanup under the now-closed explicit contract.
- Shared Rust work is valuable only when it reduces semantic ambiguity or preprocess/runtime cost without reopening the xmodel1 boundary.

## Read Before Work
Before starting a task, read:
1. `AGENTS.md`
2. `docs/project_progress.md`
3. `docs/agent_sync.md`
4. the latest dated snapshot, currently `docs/todo_2026_04_22.md`
5. `git status --short`

If code reality and document wording disagree, trust the latest code and evidence first, then update the documents.

## Current Alignment
- `xmodel1` is the training-window mainline.
- `keqingv4` is the backup line and Rust-ownership repair track.
- Preprocess, training, review, and model-strength acceptance are executed by the user.
- Agents should prioritize code alignment, Rust semantic ownership, contract hardening, gates, harnesses, and documentation sync.
- `xmodel1` preprocess/review semantic contract now includes a dedicated `hora` special sample type; terminal win actions are no longer allowed to exist only in runtime legal sets while being absent from training caches.
- `xmodel1_discard_v2` is no longer the active rerun target. The mainline is now frozen on `xmodel1_discard_v3`, and the code-side v3 contract freeze plus fixed-subset profile gate are green, so full preprocess is no longer blocked on that gate.
- The xmodel1 public contract now uses `history_summary` instead of raw `event_history`; v2-only rank buckets and `score_delta_target` are not part of the v3 public cache path.
- The xmodel1 train path now goes through the shared `training.train_model(...)` loop; xmodel1-specific code should stay limited to batch unpacking, extra losses, and summary/checkpoint shaping.
- Xmodel1 runtime tensor construction now enters through `keqing_core.build_xmodel1_runtime_tensors(...)`, and the native Rust bridge is the active owner when available; Python candidate-quality builders remain parity/fallback surfaces and should only run on missing capability.
- New xmodel1 checkpoints must carry full metadata; runtime may infer old no-`cfg` checkpoints for read-only loading, but resume across the v3 cutover is not allowed.
- Current xmodel1 preprocess internals use decision-level discard-analysis reuse plus the v3 schema compression path; the latest pass also lowered discard-candidate semantics again by removing special-waits fallback, one-shanten draw metrics, and pinfu/iipeikou break tracking from the hot path. `legacy_kakan2` history drift is expected to be fixed at the normalized-event boundary, not tolerated by whitelist.
- Public legal actions, replay snapshots, and hora-truth/scoring are now Rust-first public surfaces; Python fallback is allowed only for missing capability, not for unexpected bridge drift.
- `keqingv4` runtime history is now explicit: `DefaultDecisionContextBuilder` injects Python-built `event_history(48, 5)` into both `runtime_snap` and `model_snap`, and parity/smoke paths build the same tensor from `sample.events` plus `sample.event_index`.
- `keqingv4` opportunity tags are now explicit `v4_opportunity[3]` cache rows. Do not recover opportunity semantics from summary magic indices such as `[..., 13]`.
- `keqingv4` checkpoints are now hard-cutover fail-closed: inference and resume both require metadata validation and `strict=True` state-dict loads; metadata-less checkpoints are intentionally rejected.
- `.omx` is retired. Do not read from it, write to it, or treat it as memory.
- Legacy model families have been removed from active runtime/training surfaces. Do not restore legacy aliases or compatibility imports.
- Legacy runtime notes and deploy templates are archive-only surfaces now:
  - `docs/archive/legacy_runtime/`
  - `deploy/archive/legacy_models/`

## Coordination Rules
- When changing `src/inference`, training sample construction, or review/export behavior, check the other two surfaces for contract drift.
- When changing public legal actions, runtime snapshots, or scoring/hora truth, preserve fail-closed behavior for unexpected Rust bridge drift; missing capability is the only valid fallback trigger.
- If a change alters `xmodel1` cache row semantics rather than only metadata, do not treat `scripts/repair_xmodel1_cache_schema.py` as sufficient evidence; document whether a real re-export is mandatory.
- For xmodel1 preprocess work, treat any v3 contract change as re-export-required; do not create a pseudo-upgrade or mixed v2/v3 training path.
- For xmodel1 checkpoint work, distinguish runtime legacy-read compatibility from training resume compatibility; inference may infer old shapes, training must still fail closed without full v3 metadata.
- For keqingv4 runtime/preprocess work, keep parity aligned with production: do not add a padding fallback for missing `event_history`, and do not accept partial cache rows that omit `v4_opportunity`.
- For keqingv4 trainer work, treat `v4_opportunity` as the only opportunity-label contract; if a summary field still carries an internal feature at index `13`, that is not a public trainer signal.
- For keqingv4 checkpoint work, keep the hard cutover intact: no metadata means no inference load and no resume load.
- The current xmodel1 full export is already available on the frozen v3 contract. Do not reopen any v2 path, and keep `XMODEL1_EXPORT_PROFILE=1` ready for resumed-shard or regression spot checks.
- Do not overwrite user changes or unrelated dirty worktree changes.
- Keep the xmodel1-mainline / keqingv4-backup boundary stable unless the status board and design docs are updated together.
- Do not point active configs, scripts, or docs back at `processed_v3` or other retired preprocess roots.
- Do not reintroduce imports from removed legacy model packages; active shared surfaces are `mahjong_env.action_space`, `mahjong_env.feature_tracker`, `mahjong_env.progress_oracle`, and `training.state_features`.
- If you are only writing summaries or status updates, prefer updating the long-lived docs so the next session can continue from them directly instead of relying on chat context.
- When describing "current status", distinguish clearly between:
  - code-side readiness already landed
  - evidence still waiting on user execution
- Do not describe `keqingv4` Rust cleanup as a mainline switch unless kill criteria have actually been hit and the status board is updated in the same change.

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
