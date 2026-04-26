# Xmodel1 Call-Response Refactor Plan

Updated: 2026-04-21

## Objective

Refactor `xmodel1` in place so its weak meld/call path is replaced by an explicit call-response architecture. The user will only rerun preprocess and training after this refactor lands.

This plan exists because the current default `xmodel1` checkpoint fails a concrete call-vs-none review case:
- replay: `replay_719cee70_1776764202`
- response step: UI `step43`
- source event: `event_index=82`
- legal actions: `chi 7m(6m,8m)` vs `none`
- ground truth: `chi`
- current model choice: `none`

## Why This Must Be A Versioned Refactor

The user explicitly wants this work to stay inside `xmodel1`.

That means the safe boundary is:
- keep the namespace in `src/xmodel1/`
- treat `xmodel1_discard_v3` as the frozen old baseline
- land the refactor as an explicit new contract revision inside `xmodel1`

This plan assumes:
- old cache/checkpoint/runtime compatibility is not preserved across the refactor
- the new target should be named as a new `xmodel1` contract version rather than silently mutating `v3`

## Target Boundary

`xmodel1` should change the call problem from:
- one compressed `special` family shared by `reach / hora / chi / pon / kan / none`

to:
- a dedicated response-call decision surface
- explicit action-conditioned after-state comparison
- explicit post-call best-discard comparison

The minimal target is:
- separate response-call head
- separate self-turn declaration path
- no dependence on the current heuristic-only special-candidate scoring as the primary signal

## Non-Goals

- do not make `xmodel1` stronger via tactical patches
- do not mix placement utility into this refactor
- do not fold PPO / self-play into the first refactor implementation
- do not reopen `xmodel1_discard_v3`

## Proposed Public Direction

Planned namespace:
- `src/xmodel1/`

Planned design doc:
- `docs/xmodel1/xmodel1_call_model_design_v1.md`

Planned cache family:
- `xmodel1_discard_v4`

Planned runtime role:
- not the default runtime bot until smoke export/train/review gates are green

Locked v1 boundary:
- keep the refactor inside `xmodel1`
- freeze `xmodel1_discard_v3` as the old baseline
- first cut is not a standalone call-only model and not a full-action rewrite
- first cut is a dedicated off-turn response-action branch inside `xmodel1_discard_v4`
- this branch owns:
  - `hora`
  - `chi_low / chi_mid / chi_high`
  - `pon`
  - `daiminkan`
  - `none`
- self-turn `reach / hora / ankan / kakan / dahai` stay on the legacy `v3`-style path in the first cut
- post-call supervision keeps the full post-call discard candidate set, plus teacher best-discard targets and actual-next-discard targets when they can be aligned

## Dependency Graph

1. Step 1 must land before all other steps.
2. Step 2 must land before Steps 3, 4, and 6.
3. Step 3 must land before Steps 4 and 7.
4. Step 4 must land before Steps 5 and 7.
5. Step 5 and Step 6 can proceed in parallel once Steps 2 through 4 are stable.
6. Step 7 depends on Steps 3 through 6.

## Step 1: Freeze The Boundary

Context brief:
- `xmodel1_discard_v3` remains the frozen baseline and debugging reference.
- this refactor stays inside `xmodel1`
- `xmodel2` and `keqingrl` are out of scope for this plan

Tasks:
- write `docs/xmodel1/xmodel1_call_model_design_v1.md`
- define what belongs to the new `xmodel1` call-response revision vs what stays in the frozen `v3` baseline
- define the new public contract and naming
- update long-lived tracking surfaces only after the design wording is stable

Locked decisions:
- the new `xmodel1` revision is a full `xmodel1` contract update with a dedicated off-turn response-action branch, not a separate call-only family
- post-call discard supervision keeps the full discard candidate set after each legal call
- self-turn `reach / hora / self-turn kan` stay outside the first response-action revision

Verification:
- design doc written
- naming is stable across plan, design doc, and status wording
- `xmodel1_discard_v3` is still documented as frozen

Exit criteria:
- the project has a clear answer to “what is the next `xmodel1` contract?”

## Step 2: Rebuild Replay Sample Semantics For Calls

Context brief:
- the current call path learns from `label_action + legal_actions`, but call candidates are compressed into heuristic special slots.
- the next `xmodel1` revision needs explicit response-call records with action-conditioned after-state data.

Tasks:
- add a new replay sample adapter for off-turn response decisions
- export per-legal-call records for:
  - `hora`
  - `chi_low / chi_mid / chi_high`
  - `pon`
  - `daiminkan`
  - `none`
- for each legal call, simulate:
  - post-call hand
  - post-call meld state
  - post-call legal discards
  - post-call discard candidate set
  - teacher best post-call discard target
  - actual next discard target when replay alignment is available
- keep response-window actor/target/discard context explicit

Verification:
- focused unit tests on real chi/pon/hora windows
- parity tests that every GT call remains present in the exported legal set
- fixture covering the known failure case at `event_index=82`

Exit criteria:
- the new `xmodel1` call samples no longer depend on the old special-slot compression as the source of truth

## Step 3: Create A New Cache Schema And Loader

Context brief:
- `xmodel1_discard_v3` is frozen and should not absorb this refactor.

Tasks:
- define `xmodel1_discard_v4` cache schema
- add exporter path and strict loader
- include fields for:
  - response context
  - response action identity
  - per-action after-state features
  - nested post-response discard candidate sets
  - post-response discard supervision
  - auxiliary EV/risk targets if retained

Suggested structure:
- one row per decision window
- one candidate block per legal response action
- one nested candidate block per legal post-response discard set

Verification:
- schema contract tests
- cache shape tests
- fail-closed loader behavior on missing fields

Exit criteria:
- smoke export produces stable `xmodel1_discard_v4` cache files

## Step 4: Implement The New Model And Trainer

Context brief:
- the current `xmodel1` special path is too weak because it scores compressed special slots with heuristic-heavy features.

Tasks:
- refactor `src/xmodel1/model.py`
- refactor `src/xmodel1/trainer.py`
- build a dedicated off-turn response-action head
- incorporate explicit after-state embeddings
- incorporate post-response discard quality into the response score
- keep training loss centered on:
  - response action CE
  - post-response discard supervision
  - optional auxiliary EV/risk losses

Strong recommendation:
- do not reuse the current special-rank teacher as a primary supervisory signal
- if heuristic teacher losses are kept at all, gate them as weak regularizers only

Verification:
- model shape tests
- forward smoke tests
- CPU training smoke on tiny synthetic or cached slice

Exit criteria:
- the refactored `xmodel1` trains end-to-end on a tiny cache slice

## Step 5: Implement Runtime And Review Integration

Context brief:
- runtime must score the same response semantics that training saw.

Tasks:
- add the new `xmodel1` adapter/runtime path
- add runtime candidate construction for off-turn response decisions
- add scorer path that compares legal `hora / chi / pon / daiminkan / none` using the new `xmodel1` outputs
- add review export that makes:
  - legal response actions
  - chosen action
  - GT action
  - post-response preferred discard
  visible in one place

Verification:
- runtime adapter tests
- replay integration on known call windows
- review export includes per-call candidate diagnostics

Exit criteria:
- replay review can explain why the refactored `xmodel1` chose `chi` or `none`

## Step 6: Build A Dedicated Meld Slice Harness

Context brief:
- aggregate action accuracy is not an acceptance metric for this refactor.

Tasks:
- create a meld-focused slice harness covering:
  - `chi opportunity`
  - `pon opportunity`
  - `call vs none`
  - `ron vs call vs none`
  - post-call first discard quality
- include fixed review fixtures for:
  - the known `event_index=82` failure
  - at least one good `chi`
  - at least one correct `none`
  - at least one `hora`
  - at least one `pon`

Verification:
- deterministic fixture-based tests
- slice report script runs on exported review samples

Exit criteria:
- there is a stable acceptance surface for meld quality

## Step 7: Smoke Export, Smoke Train, And Acceptance Review

Context brief:
- the user will rerun full preprocess and real training only after the refactor is code-complete.

Tasks:
- run a small `xmodel1` smoke export on the new contract
- run a short smoke train
- run replay review on the meld slice harness
- compare against the frozen `xmodel1` baseline on:
  - known failure cases
  - call-vs-none slice metrics

Acceptance gate:
- the refactored `xmodel1` must beat the frozen `xmodel1_discard_v3` baseline on the meld slice before full export/train is requested from the user

Verification:
- syntax/compile checks
- focused xmodel1 tests
- smoke export
- smoke training
- replay review artifacts

Exit criteria:
- the repository is ready for the user to run full preprocess + short real train on the new `xmodel1` contract

## Parallelism Summary

Parallel work is allowed only after the schema direction is stable:
- Step 5 and Step 6 may proceed in parallel after Steps 2 through 4
- do not run runtime integration before the cache/model contract settles

## Anti-Patterns To Reject

- adding more heuristic coefficients to the old `xmodel1` special features
- sharing the same head for response calls and self-turn declarations again
- reusing `xmodel1_discard_v3` cache names for the new family
- measuring success with aggregate action accuracy only
- skipping post-call discard supervision
- making runtime score semantics diverge from training semantics

## Plan Mutation Protocol

Allowed mutations:
- split any step whose write scope becomes too large
- insert a step if schema or runtime integration exposes a hidden dependency
- abandon the plan only after writing down the blocking assumption that failed

If the first smoke export shows the schema is too heavy:
- keep `xmodel1_discard_v4` scoped to off-turn response windows only
- postpone full-action integration
- postpone richer auxiliary heads before cutting the after-state core

## Immediate Next Action

Start with Step 1:
- write `docs/xmodel1/xmodel1_call_model_design_v1.md`
- lock the `xmodel1_discard_v4` boundary as an off-turn response-action branch inside `xmodel1`
- sync long-lived status docs only after that boundary is stable
