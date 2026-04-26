# Rust Ownership Push Plan

Updated: 2026-04-19

## Objective

Push the repository from the current "Rust-first on selected paths" state toward a cleaner shared-semantic model where:

- replay sample construction is fully Rust-owned
- `keqingv4` runtime/preprocess semantic mirrors are fail-closed and then removable
- legal/state/scoring ownership shifts from Python twin logic into `rust/keqing_core`
- `xmodel1` runtime feature construction becomes eligible for Rust ownership after the current baseline freeze

This plan is for multi-session execution. It does not change the current global priority order:

1. `xmodel1` preprocess/train/review readiness
2. `xmodel1` architecture stability and evaluation quality
3. `keqingv4` preprocess/runtime Rust ownership
4. shared Rust semantic core consolidation
5. compatibility-surface cleanup around replay/runtime/gateway/UI

## Execution Mode

- Current branch reality: local work is on `keqingv4`
- Remote default branch: `main`
- GitHub CLI is unavailable in the current environment
- Use direct mode planning: focused reversible diffs, no GitHub workflow assumptions

## Repository Truth Snapshot

- `xmodel1` production preprocess is already Rust-owned; Python preprocess code is parity-only.
- `mahjong_env.replay.build_replay_samples_mc_return(...)` is already a Python shell over Rust replay decision records.
- `keqingv4` typed summaries and continuation scoring are already Rust-first with missing-capability-only fallback.
- hora truth is already Rust-first; Python scoring remains mainly as shell and emergency path.
- The largest remaining Python owner surfaces are:
  - `src/mahjong_env/legal_actions.py`
  - `src/mahjong_env/state.py`
  - `src/mahjong_env/scoring.py`
  - `src/training/state_features.py`
  - `src/xmodel1/features.py`
  - `src/xmodel1/candidate_quality.py`

## Invariants

- Do not move the `xmodel1` mainline / `keqingv4` backup boundary.
- Do not expand scope into old model lines, gateway/UI work, or training orchestration migration.
- Every Rust-ownership step must end with a narrower Python fallback surface than it started with.
- When a Python fallback remains, it must be missing-capability-only, not semantic-drift-tolerant.
- Contract tests must fail loudly on drift instead of silently mixing Python and Rust owners.

## Step Graph

```text
S0 boundary + guardrails
 -> S1 keqingv4 fail-closed cleanup
 -> S2 legal action public-owner consolidation
 -> S3 state core caller migration
 -> S4 scoring shell slimming
 -> S5 xmodel1 runtime feature Rust bridge
 -> S6 fallback deletion + hardening sweep
 -> S7 docs freeze + acceptance snapshot
```

Parallelism note:

- This migration is mostly serial because `legal_actions`, `state`, `scoring`, and runtime adapters share the same semantic boundary.
- The only safe parallel work is test-only coverage expansion or documentation prep that does not edit the same files.

## Step List

### S0. Boundary Freeze And Guardrails

Context brief:

- The repository has already removed most legacy active paths.
- Before shifting more ownership, the remaining Rust/Python seams must be made explicit and test-covered.

Primary write scope:

- `docs/project_progress.md`
- `docs/todo_2026_04_21.md`
- focused tests under `tests/`

Tasks:

1. Freeze the target Rustification boundary in docs using current code reality.
2. Add or tighten tests that assert "fallback only on missing capability" for:
   - `keqingv4` typed summaries
   - `keqingv4` continuation scenarios/scoring
   - hora truth/scoring
3. Record the current Python-owner inventory with file-level ownership notes.

Verification:

```bash
uv run pytest tests/test_keqingv4_inference_adapter.py tests/test_inference_contracts.py tests/test_scoring_backend_boundary.py -q
```

Exit criteria:

- The current owner map is documented.
- There is no ambiguity about which Python paths are parity-only, shell-only, or still semantic owners.

Rollback:

- Docs/tests only; revert the new guardrail notes and targeted tests.

### S1. keqingv4 Fail-Closed Cleanup

Status note (`2026-04-21`):

- This slice has landed on the current `keqingv4` backup line:
  - runtime `event_history` is explicit and fail-closed
  - `v4_opportunity` replaced summary magic-channel reads
  - keqingv4 checkpoints now require metadata validation and `strict=True` load for inference and resume
- Treat S1 as complete unless a new drift is discovered. The next remaining shared-core step is `S2`.

Context brief:

- `keqingv4` summary/future-truth/continuation paths are already Rust-first.
- The next move is not deeper structure expansion. It is deleting tolerance for silent mirror drift.

Primary write scope:

- `src/keqingv4/preprocess_features.py`
- `src/inference/scoring.py`
- `src/inference/keqing_adapter.py`
- `tests/test_keqingv4_inference_adapter.py`
- `tests/test_inference_contracts.py`
- `tests/test_keqingv4_preprocess.py`

Tasks:

1. Remove any fallback that still triggers for semantic drift rather than missing capability.
2. Narrow Python summary/continuation code into explicit emergency mirrors only.
3. Make contract validation run before fallback handoff wherever the bridge schema can drift.
4. Mark the remaining Python mirror helpers as removable once S6 lands.

Verification:

```bash
uv run pytest tests/test_keqingv4_preprocess.py tests/test_keqingv4_inference_adapter.py tests/test_inference_contracts.py tests/test_model_v4_shapes.py tests/test_training_v4_smoke.py -q
uv run python scripts/evaluate_keqingv4_special_calibration.py --help
uv run python scripts/evaluate_keqingv4_special_calibration_from_replays.py --help
```

Exit criteria:

- `keqingv4` runtime/preprocess path is Rust-first and fail-closed.
- Python mirror code remains only as explicit missing-capability emergency support.

Rollback:

- Re-enable the last known passing missing-capability fallback branches only.
- Do not reintroduce silent drift-tolerant mixed ownership.

### S2. Legal Action Public-Owner Consolidation

Context brief:

- Rust already owns structural legal enumeration and hora candidate generation.
- Python still assembles the public legal set and mixes Rust and Python rule branches.

Primary write scope:

- `rust/keqing_core/src/legal_actions.rs`
- `rust/keqing_core/src/py_module.rs`
- `src/keqing_core/__init__.py`
- `src/mahjong_env/legal_actions.py`
- `tests/test_rust_legal_actions_parity.py`
- `tests/test_legal_actions_rules.py`
- `tests/test_legal_actions_comprehensive.py`
- `tests/test_replay_strict_legal.py`

Tasks:

1. Extend Rust from structural enumeration toward the public legal action surface.
2. Move remaining reach-discard / ankan-after-reach / hora-detail filtering into Rust where feasible.
3. Reduce Python `legal_actions.py` to:
   - bridge wrapper
   - compatibility dataclass/object shaping
   - emergency missing-capability fallback only
4. Add parity cases that pin the exact public legal set, not just structural shape.

Verification:

```bash
uv run pytest tests/test_rust_legal_actions_parity.py tests/test_legal_actions_rules.py tests/test_legal_actions_comprehensive.py tests/test_replay_strict_legal.py -q
```

Exit criteria:

- Public legal-set ownership is Rust-first.
- Python no longer contains active rule logic except emergency fallback.

Rollback:

- Restore Python public legal assembly temporarily, but keep new Rust helper surfaces and parity tests.

### S3. State Core Caller Migration

Context brief:

- Rust replay/sample paths already use `GameStateCore`.
- Many runtime/replay callers still instantiate Python `GameState` and apply Python events.

Primary write scope:

- `rust/keqing_core/src/state_core.rs`
- `rust/keqing_core/src/event_apply.rs`
- `rust/keqing_core/src/snapshot.rs`
- `rust/keqing_core/src/py_module.rs`
- `src/keqing_core/__init__.py`
- `src/mahjong_env/state.py`
- `src/inference/default_context.py`
- `src/inference/runtime_bot.py`
- `src/inference/rulebase_bot.py`
- `src/replay/server.py`
- parity tests around state reconstruction

Tasks:

1. Expose a cleaner Rust-backed snapshot/apply interface for Python callers.
2. Migrate high-value callers from Python state mutation to Rust-backed state mutation or snapshot reconstruction.
3. Downgrade `mahjong_env.state.GameState` from semantic owner to compatibility shell.
4. Keep the Python dataclass only where object ergonomics are still needed.

Verification:

```bash
uv run pytest tests/test_rust_state_core_parity.py tests/test_replay_state_reconstruction.py tests/test_replay_hora_reconstruction.py tests/test_replay_tsumogiri_reconstruction.py tests/test_tenhou_public_state_sync.py -q
```

Exit criteria:

- Main replay/runtime callers no longer depend on Python state mutation as the semantic truth source.
- Python state code can be described as adapter-shell, not owner.

Rollback:

- Repoint selected callers back to Python state application while retaining the new Rust snapshot bridge and tests.

### S4. Scoring Shell Slimming

Context brief:

- Hora truth already resolves through Rust.
- `mahjong_env/scoring.py` still carries heavy state reconstruction and fallback orchestration.

Primary write scope:

- `rust/keqing_core/src/hora_truth.rs`
- `rust/keqing_core/src/native_scoring.rs`
- `rust/keqing_core/src/py_module.rs`
- `src/mahjong_env/scoring.py`
- `tests/test_scoring_backend_boundary.py`
- `tests/test_rust_hora_shape_parity.py`

Tasks:

1. Remove unnecessary Python hand-value estimation branches from default paths.
2. Move more prepared-payload evaluation and boundary validation into Rust.
3. Keep Python scoring focused on API shape and compatibility helpers.
4. Ensure fallback remains capability-driven only.

Verification:

```bash
uv run pytest tests/test_scoring_backend_boundary.py tests/test_rust_hora_shape_parity.py -q
```

Exit criteria:

- Default scoring path is Rust-only for truth and validation.
- Python scoring logic is materially smaller and no longer owns decision semantics.

Rollback:

- Re-enable the previous prepared-payload compatibility branch while keeping Rust truth as default.

### S5. xmodel1 Runtime Feature Rust Bridge

Context brief:

- `xmodel1` production preprocess is already Rust-owned.
- Runtime still constructs `state_tile_feat`, `candidate_feat`, `special_candidate_feat`, and `event_history` in Python.
- This is the biggest remaining train/runtime dual-owner seam, but it should wait until the current baseline is frozen enough to avoid destabilizing training-window work.

Primary write scope:

- `rust/keqing_core/src/xmodel1_export.rs`
- new shared Rust feature helpers if needed
- `src/xmodel1/features.py`
- `src/xmodel1/candidate_quality.py`
- `src/inference/keqing_adapter.py`
- `tests/test_xmodel1_rust_contract.py`
- `tests/test_xmodel1_runtime_event_history.py`
- `tests/test_xmodel1_inference_adapter.py`

Tasks:

1. Expose Rust runtime feature builders for:
   - state tile/scalar features
   - discard candidate features and flags
   - special candidate features
   - event history
2. Switch runtime adapters to prefer Rust-built arrays.
3. Keep Python feature code as parity oracle until S6.
4. Pin exact array parity on minimal and broader cases.

Verification:

```bash
uv run pytest tests/test_xmodel1_rust_contract.py tests/test_xmodel1_runtime_event_history.py tests/test_xmodel1_inference_adapter.py tests/test_xmodel1_cached_dataset.py -q
```

Exit criteria:

- `xmodel1` runtime and preprocess consume the same Rust-side feature semantics.
- Python feature code is no longer on the hot path.

Rollback:

- Restore Python runtime builders behind a single switch while keeping Rust parity tests and exported runtime bridge.

### S6. Fallback Deletion And Hardening Sweep

Context brief:

- After S1-S5, several Python mirrors should be dead weight.
- This step is where the repo stops pretending mixed ownership is a stable end state.

Primary write scope:

- `src/keqingv4/preprocess_features.py`
- `src/mahjong_env/legal_actions.py`
- `src/mahjong_env/state.py`
- `src/mahjong_env/scoring.py`
- `src/xmodel1/features.py`
- `src/xmodel1/preprocess.py`
- tests and docs

Tasks:

1. Delete parity code that is no longer needed on production/runtime paths.
2. Keep only:
   - explicit test oracle code
   - capability-missing emergency compatibility where still justified
3. Remove dead config flags and stale manifest wording that imply active Python semantics.
4. Strengthen regressions so fallback resurrection is detected quickly.

Verification:

```bash
uv run pytest tests/test_keqingv4_preprocess.py tests/test_keqingv4_inference_adapter.py tests/test_inference_contracts.py tests/test_xmodel1_rust_contract.py tests/test_scoring_backend_boundary.py tests/test_rust_legal_actions_parity.py tests/test_rust_state_core_parity.py -q
```

Exit criteria:

- Python fallback surface is intentionally tiny and documented.
- Production paths no longer advertise Python semantic ownership.

Rollback:

- Restore only the smallest emergency compatibility branch needed to recover a broken runtime or preprocess gate.

### S7. Docs Freeze And Acceptance Snapshot

Context brief:

- The migration is only complete when the ownership map is reflected in long-lived docs and the mainline/backup boundary remains consistent.

Primary write scope:

- `docs/project_progress.md`
- `docs/agent_sync.md`
- latest `docs/todo_YYYY_MM_DD.md`
- a new dated Rust snapshot if the phase meaning materially changed

Tasks:

1. Update long-lived status with the new ownership map.
2. Record which Python surfaces are parity-only, shell-only, or removed.
3. Confirm the priority order and mainline/backup wording did not drift.
4. Add a dated acceptance snapshot once S6 is complete.

Verification:

```bash
rg -n "xmodel1|keqingv4|Rust" docs/project_progress.md docs/agent_sync.md docs/todo_2026_04_21.md
```

Exit criteria:

- Docs match the code reality.
- No outdated runbook still implies old owner boundaries.

Rollback:

- Revert documentation-only changes; no code rollback required.

## Recommended Near-Term Slice

The next execution window should cover only:

1. `S0` boundary freeze and guardrails
2. `S1` keqingv4 fail-closed cleanup
3. `S2` legal action public-owner consolidation design/prototype

Reason:

- This keeps the work aligned with the current priority order.
- It pushes real Rust ownership now without destabilizing the `xmodel1` training window.
- It avoids starting the highest-risk `xmodel1` runtime feature migration too early.

## Anti-Patterns To Reject

- Reintroducing broad Python fallback "just in case"
- Mixing Rust structural logic with Python semantic filtering as a permanent architecture
- Starting `xmodel1` runtime feature migration before the current preprocess/train/review gate has stabilized
- Pulling gateway/UI/runtime bot ergonomics into the Rust-core scope
- Declaring completion after helper landing without parity and boundary tests
- Expanding recursion depth or summary richness in `keqingv4` before fail-closed cleanup is done

## Plan Mutation Protocol

If reality changes during execution:

- Split a step when a single PR would exceed one semantic boundary or one focused verification set.
- Insert a step only when a new blocker appears that cannot be absorbed into the current step.
- Reorder steps only if the dependency edge was wrong in code reality.
- Abandon a step only with a written note in `docs/project_progress.md` and the latest dated todo explaining why.

Required mutation record:

- what changed
- why the original ordering failed
- what new verification is required
- whether the mainline/backup boundary moved

## Review Checklist

Use this checklist before executing each step:

1. Does the step reduce Python semantic ownership instead of moving code around?
2. Is the write scope focused enough to verify in one pass?
3. Are the tests checking owner boundaries, not just happy-path behavior?
4. Would rollback preserve the newer Rust surface instead of reinstating the old architecture?
5. Does the step avoid stealing focus from `xmodel1` mainline validation?

## Success Definition

This plan succeeds when the repository can be described as:

- `xmodel1` mainline remains the training-window line
- `keqingv4` remains backup plus shared Rust repair track
- replay/legal/state/scoring semantics are Rust-owned by default
- Python is primarily trainer/orchestration/API shell, not semantic truth owner
