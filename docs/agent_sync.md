# Keqing1 Agent Sync

Updated: 2026-04-21

This document defines the minimum shared context for multi-agent work.

## Short Handoff Summary
- The repository should currently be read as a model-building workspace with three equal-priority top-tier workstreams: `xmodel1`, `keqingv4`, and `keqingrl`.
- `xmodel1_discard_v3` remains the frozen baseline contract, but it is no longer the active training contract for meld quality.
- A verified replay case now shows the current `v3` meld path is structurally weak even with aligned preprocess/runtime/action mapping:
  - replay `replay_719cee70_1776764202`
  - UI `step43`
  - `event_index=82`
  - legal `chi` vs `none`
  - GT `chi`
  - model choice `none`
- The active mainline task is now the in-place response-action refactor cut over as `xmodel1_discard_v5` inside `src/xmodel1/`, not a new family.
- `keqingrl` is now a top-priority parallel RL line with a distinct contract. It is not the same answer as the `xmodel1` meld refactor, but it is no longer a lower-priority sidecar.
- `xmodel2` remains the bounded lower-priority experiment line.
- `pt_map` now has a dedicated runtime utility surface and is not the current model-family contract boundary.
- `keqingv4` is not competing for the training window. It is the backup snapshot and the safer place to continue Rust-ownership cleanup under the now-closed explicit contract.
- `docs/keqingv4/keqingv4_model_design_v2.md` now documents that explicit `keqingv4` backup snapshot as a current contract note; do not read it as an open rearchitecture plan.
- `xmodel1`, `keqingv4`, and `keqingrl` now share equal project priority. Do not impose a serial order across those three lines unless a concrete dependency forces one.
- Shared Rust work is valuable only when it reduces semantic ambiguity or preprocess/runtime cost without reopening the xmodel1 boundary.

## Read Before Work
Before starting a task, read:
1. `AGENTS.md`
2. `docs/project_progress.md`
3. `docs/agent_sync.md`
4. the latest dated snapshot, currently `docs/todo_2026_04_24.md`
5. `git status --short`

If code reality and document wording disagree, trust the latest code and evidence first, then update the documents.

## Current Alignment
- `xmodel1` is the training-window mainline.
- `xmodel1_discard_v3` is the frozen baseline and debugging reference.
- The active xmodel1 contract is `xmodel1_discard_v5`: a response-only action branch inside `xmodel1` with no parallel `special_*` tensor path.
- The response branch owns:
  - `hora`
  - `chi_low / chi_mid / chi_high`
  - `pon`
  - `daiminkan`
  - `none`
- The current cut does not yet rebuild self-turn `reach / hora / ankan / kakan / dahai`.
- `keqingrl` is a distinct interactive RL family and now shares equal priority with `xmodel1` and `keqingv4`.
- `xmodel2` remains the bounded lower-priority experiment family.
- `keqingv4` is the backup line and Rust-ownership repair track.
- the live `keqingv4` design note is now a current backup-contract snapshot, not a future-v4 proposal surface.
- `xmodel1`, `keqingv4`, and `keqingrl` are equal-priority workstreams. Their roles differ, but agents should not force a fixed top-down order across them.
- Preprocess, training, review, and model-strength acceptance are executed by the user.
- Agents should prioritize code alignment, Rust semantic ownership, contract hardening, gates, harnesses, and documentation sync.
- `xmodel1` preprocess/review semantic contract is now response-only on the active path: terminal win actions such as `hora` must stay inside the exported response candidate set and cannot exist only in runtime legal sets.
- The repaired xmodel1 response contract is now: whenever a non-discard training row still has legal `dahai` choices at the same decision point, the exported row must retain the same-turn discard candidate arrays so `action_ce` can supervise direct action competition instead of relying on a removed `special_*` path.
- `xmodel1_discard_v2` is no longer the active rerun target. The baseline remains frozen on `xmodel1_discard_v3`, and the code-side `v5` response-only contract plus fixed-subset profile gate are green, so the next blocker is no longer schema wiring but the required full re-export/retrain.
- The xmodel1 public contract now uses `history_summary` instead of raw `event_history`; v2-only rank buckets and `score_delta_target` are not part of the active cache path.
- The xmodel1 train path now goes through the shared `training.train_model(...)` loop; xmodel1-specific code should stay limited to batch unpacking, extra losses, and summary/checkpoint shaping.
- The current `processed_xmodel1_v1b` export and `artifacts/xmodel1_runs/v1b_train_20260422/` checkpoint now count as debugging evidence for that contract bug, not as valid acceptance artifacts for self-turn special-vs-discard behavior. A real full re-export and retrain are mandatory before further strength claims.
- The current experiment split is now explicitly mapped as:
  - `xmodel1`: active `v5` response-only refactor
  - `xmodel2`: offline decomposed-EV placement-support experiment
  - `keqingrl`: interactive policy / actor-critic line
  - `src/inference/pt_map.py`: placement utility only
  - after-state scorer / reanalysis / self-play: future style-migration path
- `keqingrl` now has action contracts, a neural forward path, PPO batch/loss helpers, terminal reward helpers, a discard-only-named Mahjong env wrapper, generic seat-policy collectors, opponent-pool helpers, rollout/PPO smoke coverage, a multi-episode PPO iteration surface, rollout review/export helpers, and a longer-run training harness.
- The current learned action surface in `keqingrl` is now:
  - `DISCARD`
  - `REACH_DISCARD`
  - `TSUMO`
  - `ANKAN`
  - `KAKAN`
  - `RYUKYOKU`
- The current learned off-turn response surface in `keqingrl` is now:
  - `RON`
  - `CHI`
  - `PON`
  - `DAIMINKAN`
  - `PASS`
- `REACH` is now represented in `keqingrl` only through the composite `REACH_DISCARD` action; bare Mahjong `"reach"` still must not enter rollout/training/review surfaces by itself.
- `keqingrl` self-play collection is now generic under the hood: discard-only-named wrappers remain for compatibility, but collector/training can now assign learner seats explicitly and sample opponents from an opponent pool.
- `keqingrl` PPO updates on mixed-policy episodes must only train on learner-controlled seats; opponent actions may appear in rollout history but must not enter the learner batch.
- `keqingrl` still does not have longer-run behavioral strength evidence on Mahjong. The bounded four-player action surface now includes learned `RYUKYOKU`; the next gaps are longer-run evidence on the new opponent-pool path, fuller self-play coverage, and later optional surfaces such as `NUKI`.
- Xmodel1 runtime tensor construction now enters through `keqing_core.build_xmodel1_runtime_tensors(...)`, and the native Rust bridge is the active owner when available; Python candidate-quality builders remain parity/fallback surfaces and should only run on missing capability.
- The active xmodel1 runtime/export path now contains only `candidate_* + response_* + history_summary`; do not reintroduce `special_*` tensors on the mainline path.
- New xmodel1 checkpoints must carry full metadata; runtime may infer old no-`cfg` checkpoints for read-only loading, but resume across the `v5` cutover is not allowed.
- Current xmodel1 preprocess internals use decision-level discard-analysis reuse on the response-only `v5` path; the latest pass also lowered discard-candidate semantics again by removing special-waits fallback, one-shanten draw metrics, and pinfu/iipeikou break tracking from the hot path. `legacy_kakan2` history drift is expected to be fixed at the normalized-event boundary, not tolerated by whitelist.
- Public legal actions, replay snapshots, and hora-truth/scoring are now Rust-first public surfaces; Python fallback is allowed only for missing capability, not for unexpected bridge drift.
- `keqingv4` runtime history is now explicit: `DefaultDecisionContextBuilder` injects Python-built `event_history(48, 5)` into both `runtime_snap` and `model_snap`, and parity/smoke paths build the same tensor from `sample.events` plus `sample.event_index`.
- `keqingv4` opportunity tags are now explicit `v4_opportunity[3]` cache rows. Do not recover opportunity semantics from summary magic indices such as `[..., 13]`.
- `keqingv4` checkpoints are now hard-cutover fail-closed: inference and resume both require metadata validation and `strict=True` state-dict loads; metadata-less checkpoints are intentionally rejected.
- `keqingv4` current placement-rank support is now landed through export/cache/model/trainer/review/runtime surfaces in `docs/keqingv4/keqingv4_model_design_v2.md`: the current cache contract version is `7`, cache targets stay raw (`final_rank_target` + `final_score_delta_points_target`), placement heads/losses are wired, review aux exports `rank_probs` / `final_score_delta` / derived `rank_pt_value`, runtime now has action-conditioned placement rerank plus per-candidate placement meta, `rank_pt` stays trainer/scoring-side only, and `rank_pt_lambda=0.0` remains the default.
- Shared replay samples now expose `score_before_action`, deterministic `final_rank_target`, and raw `final_score_delta_points_target`; shared preprocess now has `create_preprocess_adapter('xmodel2_aux')` for exporting that placement contract on the base cache schema.
- `.omx` is retired. Do not read from it, write to it, or treat it as memory.
- Legacy model families have been removed from active runtime/training surfaces. Do not restore legacy aliases or compatibility imports.
- Legacy runtime notes and deploy templates are archive-only surfaces now:
  - `docs/archive/legacy_runtime/`
  - `deploy/archive/legacy_models/`

## Coordination Rules
- When planning or sequencing work, treat `xmodel1`, `keqingv4`, and `keqingrl` as the same top priority. Only make one wait behind another when there is a direct contract or evidence dependency.
- When changing `src/inference`, training sample construction, or review/export behavior, check the other two surfaces for contract drift.
- For xmodel1 self-turn special sampling, do not allow `hora` / `reach` / self-turn `kan` rows with legal discards to zero out discard candidates; runtime compares those actions against `dahai` directly, so the training row must expose the same competition set.
- For `xmodel1_discard_v5`, do not reintroduce a parallel `special_*` path next to `response_*`; the mainline branch must have exactly one non-discard action surface.
- For `xmodel1_discard_v5`, do not compress multiple legal off-turn response actions into one family representative before training or runtime scoring; the branch must see the full legal response set.
- For `xmodel1_discard_v5`, keep `none` in the same direct comparison set as `hora / chi / pon / daiminkan`; do not route `none` through a separate heuristic shortcut.
- For `xmodel1_discard_v5`, keep post-response discard candidate sets explicit. Do not collapse them to a single summary scalar before the model/trainer boundary.
- For `xmodel1_discard_v5`, treat the schema/checkpoint change as a hard cutover and real re-export boundary; do not attempt a pseudo-upgrade from `v3` or a mixed `special_*`/`response_*` cache.
- Do not implement `keqingrl` inside the existing imitation trainers. `training.train_model(...)`, fixed 45-way CE logits, and supervised cache rows are not the `keqingrl` contract.
- For `keqingrl`, treat `legal_actions` ordering as part of the public learner contract: rollout sampling, log-prob replay, and env stepping must all consume the same ordered action list.
- For `keqingrl`, keep the exact per-turn `ActionSpec` instances stable through `legal_actions -> observe -> sample -> step`; equality alone is not sufficient once red-tile variants normalize to the same 34-tile id.
- For `keqingrl`, keep rollout storage separate from supervised caches. `old_log_prob`, `entropy`, `policy_version`, and rule-conditioned rewards are first-class rollout data, not metadata patches on imitation samples.
- For `keqingrl`, do not compute GAE on one globally interleaved step stream across all seats. Critic targets are actor-relative; returns and advantages must be backed up per actor trajectory before batching shared-policy updates.
- For `keqingrl`, learner-side batching must tolerate variable legal-action counts. Padding + legal mask is the batch contract; fixed-width stack assumptions are invalid on real Mahjong episodes.
- For `keqingrl`, when episodes mix learner and opponent policies, PPO batching must filter to learner-controlled seats only. Do not silently backprop on opponent actions just because they are present in the rollout record.
- For `keqingrl`, iteration metrics should describe the collected rollout distribution and PPO update stability, not imply strength. `mean_rank`, `reward`, `KL`, and `clip_fraction` are iteration diagnostics until longer-run evidence exists.
- For `keqingrl`, rollout review/export depends on storing the ordered `legal_actions` objects in each rollout step. Do not strip them from the review path just because learner replay only needs ids/features/masks.
- For `keqingrl`, mixed-policy rollout review must use the stored acting `policy_name` / `policy_version` metadata plus a resolver when an episode contains multiple acting policies. Do not treat `review_rollout_episode(single_policy, ...)` as semantically exact for mixed-policy episodes.
- For `keqingrl`, keep bare self-turn `"reach"` out of rollout/training/review data even after the composite contract lands. The learner surface should only see bound `REACH_DISCARD` actions tied to a concrete discard tile.
- For `keqingrl`, keep off-turn response windows on the same ordered legal-action contract as self-turn decisions. Do not fork a separate response-only replay/log-prob path.
- `keqingrl` Phase 0/1 should prioritize action-contract correctness, legality, and reviewability over full-action coverage or throughput.
- Do not put new placement-utility semantics into `xmodel1`; put them in `xmodel2` or shared neutral surfaces instead.
- For placement experiments, keep cache targets raw and neutral: `final_rank_target` + `final_score_delta_points_target` are allowed, league-specific `rank_pt` is not.
- Keep `pt_map` as a runtime/scoring utility boundary unless the status board and `docs/xmodel2/xmodel2_placement_design_v1.md` are updated together.
- Do not describe `xmodel2` as the mainline unless the status board explicitly moves the boundary; today it is a bounded experimental family only.
- Do not describe `keqingrl` as the backup line or as a drop-in `keqingv4` successor; it is a new family with a different training/runtime contract.
- Equal priority does not erase role boundaries:
  - `xmodel1` remains the supervised mainline
  - `keqingv4` remains the backup snapshot and Rust-ownership repair track
  - `keqingrl` remains the RL-native family
- Treat after-state scorer / reanalysis / self-play as the intended rule/style migration path. Do not substitute tactical heuristics for that missing data regime and call it equivalent.
- When changing public legal actions, runtime snapshots, or scoring/hora truth, preserve fail-closed behavior for unexpected Rust bridge drift; missing capability is the only valid fallback trigger.
- If a change alters `xmodel1` cache row semantics rather than only metadata, do not treat `scripts/repair_xmodel1_cache_schema.py` as sufficient evidence; document whether a real re-export is mandatory.
- For xmodel1 preprocess work, treat any active schema change as re-export-required; do not create a pseudo-upgrade or mixed `v3`/`v5` training path.
- For xmodel1 checkpoint work, distinguish runtime legacy-read compatibility from training resume compatibility; inference may infer old shapes, training must still fail closed without full `v5` metadata.
- For keqingv4 runtime/preprocess work, keep parity aligned with production: do not add a padding fallback for missing `event_history`, and do not accept partial cache rows that omit `v4_opportunity`.
- For keqingv4 trainer work, treat `v4_opportunity` as the only opportunity-label contract; if a summary field still carries an internal feature at index `13`, that is not a public trainer signal.
- For keqingv4 checkpoint work, keep the hard cutover intact: no metadata means no inference load and no resume load.
- For `keqingv4` current placement-rank support, keep it in-place on `keqingv4`; do not fork a separate family name for this slice.
- For `keqingv4` current placement-rank support, keep cache targets raw and neutral: `final_rank_target` + `final_score_delta_points_target` are allowed, cache-native `rank_pt` is not.
- For `keqingv4` current placement-rank support, do not add a standalone `rank_pt_value_head`; derive `rank_pt` from `rank_logits + final_score_delta` in trainer/scoring.
- For `keqingv4` current placement-rank support, treat the schema/checkpoint change as a hard cutover and real re-export boundary; do not repair old caches in place.
- The currently landed `keqingv4` placement boundary now includes runtime action-conditioned placement reranking, but only as a default-off surface. Do not describe it as a default-enabled policy change or as `rank-pt policy`.
- Keep `rank_pt_lambda=0.0` by default on `keqingv4`; any future runtime placement bonus must be action-conditioned, not a state-constant aux offset.
- The current `final_rank_target` tie-break for placement work is frozen to `game_start_oya` seat order until a platform-perfect clone is explicitly confirmed.
- When updating `docs/keqingv4/keqingv4_model_design_v2.md`, keep it aligned with the frozen backup snapshot and default-off placement boundary; do not mix old speculative v4 redesign text back into that document.
- The current xmodel1 full export root `processed_xmodel1_v1b` predates the repaired special-vs-discard row semantics and must not be reused for new training. Do not reopen any v2 path, and keep `XMODEL1_EXPORT_PROFILE=1` ready for the required re-export.
- The current `v1b_train_20260422` checkpoint is debugging evidence only; it is not an acceptance artifact for the new `v4` meld path.
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
