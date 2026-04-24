# KeqingRL Rust/Python Boundary

Updated: 2026-04-24

## Conclusion

KeqingRL should not validate everything in Python first and then rewrite in Rust.
The Rust/Python boundary must be fixed now:

- Rust owns deterministic mahjong semantics, hot paths, and cross-language single truth.
- Python owns learning algorithms, neural networks, experiment orchestration, review, and reports.
- Hybrid modules may start in Python, but their contracts must be written as if Rust will own the stable semantic core.

The repository is already partly aligned with this direction. `rust/keqing_core` contains Rust implementations for legal actions, event application, native scoring, rulebase scoring, shanten tables, state snapshots, and Python bindings. KeqingRL already calls Rust for rulebase action scoring and selected reach-discard/shanten paths.

This plan intentionally supports two tracks:

- Short term: continue controlled discard-only middle-game research while contracts are tightened.
- Medium term: safely unlock `REACH_DISCARD`, `PASS`, `PON`, `CHI`, and `KAN` only after action identity, legality, terminal, and feature contracts are Rust-owned or parity-gated.

## Current Layering

### Rust-Owned Or Rust-First

| Domain | Current code | Status | Required direction |
| --- | --- | --- | --- |
| State/event semantics | `rust/keqing_core/src/event_apply.rs`, `rust/keqing_core/src/state_core.rs`; Python twin in `src/mahjong_env/state.py` | Partially Rust-owned | Move game state advancement behind Rust wrappers for KeqingRL runtime; keep Python only as compatibility shell or tests. |
| Legal action enumeration | `rust/keqing_core/src/legal_actions.rs`; Python twin in `src/mahjong_env/legal_actions.py` | Partially Rust-owned | Rust must own ordering and legality before unlocking calls/kan. Python enumeration should become fallback/parity-only. |
| Shanten/tile tables | `rust/keqing_core/src/shanten.rs`, `rust/keqing_core/src/shanten_table.rs`; Python helper logic remains in replay/features | Mostly Rust-owned | Use Rust shanten/ukeire/waits for all runtime contracts; retire Python recursive/special-case sources where they affect policy/eval. |
| Rulebase scoring | `rust/keqing_core/src/rulebase.rs`, `rust/keqing_core/src/native_scoring.rs`; wrapper in `src/keqingrl/rule_score.py` | Rust-owned with Python wrapper | Keep Rust as scorer source of truth; Python only converts inputs, tensors, prior logits, and diagnostics. |
| Hora/terminal truth | `rust/keqing_core/src/hora_truth.rs`, legal terminal candidates; Python in `src/mahjong_env/scoring.py` | Rust-first, Python fallback remains | TSUMO/RON/RYUKYOKU arbitration must be Rust-owned or Rust-specified before terminal actions are policy-controlled. |
| Replay sample construction | `rust/keqing_core/src/replay_export_core.rs`, `src/mahjong_env/replay.py` shell | Rust-owned | Keep Rust as source; Python reads and packages records. |
| Python bindings | `rust/keqing_core/src/py_module.rs` | Boundary layer | Add all cross-language semantic contracts here, including schema gate, action identity, legal enumeration, terminal resolver, and typed hot-path APIs. |

### Python-Owned Or Python-First

| Domain | Current code | Status | Required direction |
| --- | --- | --- | --- |
| Policy/network architecture | `src/keqingrl/policy.py` | Python-owned | Keep in PyTorch; do not migrate PPO policy/value networks to Rust. |
| PPO/GAE/losses | `src/keqingrl/ppo.py`, `src/keqingrl/buffer.py` | Python-owned | Keep Python-first for research velocity. |
| Training orchestration | `src/keqingrl/training.py`, `src/keqingrl/selfplay.py`, `src/keqingrl/opponent_pool.py` | Python-owned | Keep Python orchestrating experiments, opponent pools, checkpoints, and smoke reports. |
| Reward/style experiments | `src/keqingrl/rewards.py`, `src/keqingrl/style.py` | Python-first | Keep mutable until spec stabilizes; do not freeze into Rust yet. |
| Review/reporting | `src/keqingrl/review.py`, JSONL export/report helpers | Python-owned | Keep Python; fields and summaries will change often. |
| Latency wrapper | `src/keqingrl/latency.py` | Python wrapper | Keep wrapper in Python, but measure and isolate Rust core latency separately. |

### Hybrid Contracts

| Domain | Current code | Current risk | Required direction |
| --- | --- | --- | --- |
| Native contract/schema/version gate | `py_module.rs` exports individual functions without one required schema gate | Python may run against an incompatible native module or silently use stale semantics | Add `native_schema_info_json_py` plus strict Python startup checks before any progressive unlock. |
| Action identity serialization | `src/keqingrl/actions.py` owns `ActionSpec.canonical_key` and `encode_action_id` | Highest drift risk once REACH/PASS/PON/CHI/KAN unlock | Add Rust `ActionIdentity` v1 covering type, actor, target, tiles, called tile/source, reach/kan/terminal metadata, canonical key, and action id; make Rust the source of truth before progressive unlock. |
| Observation builder | `src/keqingrl/env.py` calls `encode_state_features`; related builders in training/xmodel modules | Contract can drift from Rust state snapshots | Python may continue now; add stable schema/version tests, then migrate packed array generation to Rust typed API after contract freezes. |
| Action feature builder | `src/keqingrl/env.py` `_action_features` | Simple now, fragile with calls/kan/reach | Python can stay for DISCARD-only; Rust parity or Rust-owned features must be complete before opening non-discard actions. |
| Rollout serialization | `src/keqingrl/rollout.py`, `src/keqingrl/buffer.py` | Rebuilding action order/features would corrupt PPO | Keep Python JSONL/batches now; never rebuild old legal actions; consider Rust reader/writer after schema stabilizes. |
| Fixed-seed eval | `src/keqingrl/training.py` smoke eval | Scheduler is Python, deterministic game core must not be | Python orchestrates; Rust should own deterministic game progression and duplicate wall/deal contract in the medium term. |
| JSON bridge | current Python bindings often expose JSON convenience wrappers | Good for golden/debug, too loose for runtime hot paths | Keep JSON bridge for golden tests, debugging, and trace inspection; plan typed Rust/Python APIs for hot paths before scaling self-play. |

## Runtime Fallback Policy

- Runtime self-play, evaluation, and policy-controlled stepping must not silently fall back from Rust to Python semantics.
- If a Rust capability required by the active action scope is missing or schema-incompatible, fail closed with an explicit error and capability report.
- Python fallback may exist only in golden tests, offline parity/debug tooling, or legacy replay compatibility paths that are explicitly marked non-runtime.
- Unsupported actions, including `NUKI` until parity lands, must be represented as explicit unsupported capabilities rather than implicitly ignored or Python-handled.

## Rust Rewrite Backlog

### P0: Must Do Before Progressive Unlock

1. Add native contract/schema/version gate.
   - Target: `native_schema_info_json_py` in `rust/keqing_core/src/py_module.rs` backed by Rust constants for boundary schema, action identity version, legal enumeration version, terminal resolver version, and supported action capabilities.
   - Python affected: KeqingRL env/runtime initialization and tests that import native bindings.
   - Tests: strict Python check rejects missing, stale, or incompatible native schema info; capability report marks unsupported actions explicitly.
   - Reason: Python must never start a runtime with stale native semantics or silently downgrade to Python behavior.

2. Add Rust `ActionIdentity` v1 serializer.
   - Target: new Rust functions in `rust/keqing_core/src` exposed via `py_module.rs`.
   - Python affected: `src/keqingrl/actions.py`.
   - Scope: `ActionIdentity` includes action kind, actor, response target, tile payloads, called tile/source, kan/reach/terminal metadata, stable canonical key, and encoded action id; `canonical_key` becomes one field, not the whole contract.
   - Tests: parity for `DISCARD`, `REACH_DISCARD`, `TSUMO`, `RON`, `PASS`, `PON`, `CHI`, `DAIMINKAN`, `ANKAN`, `KAKAN`, `RYUKYOKU`, and `NUKI` as either parity-supported or explicitly unsupported.
   - Reason: rollout identity, action ids, review/debug identity, and future call semantics must share one full action contract; drift corrupts PPO training data.

3. Route KeqingRL env legal action enumeration through Rust source of truth.
   - Target: `rust/keqing_core/src/legal_actions.rs` + `py_module.rs` public wrapper.
   - Python affected: `src/keqingrl/env.py`, `src/mahjong_env/legal_actions.py`.
   - Contract: KeqingRL env consumes ordered legal actions from Rust for the active action scope; Python enumeration is parity/debug/legacy only, not a runtime source of truth.
   - Reason: legal action order is a rollout contract and cannot be Python/Rust dual-source.

4. Rust-own forced terminal arbitration via resolver API.
   - Target: resolver API in Rust that accepts state/context and returns one resolved terminal outcome or explicit no-terminal decision, not only a list of candidates.
   - Python affected: `src/keqingrl/env.py` autopilot methods and `src/mahjong_env/scoring.py` fallback paths.
   - Tests: autopilot trace regression for TSUMO/RON/RYUKYOKU arbitration and ordering-sensitive terminal windows.
   - Reason: TSUMO/RON/RYUKYOKU semantics must not depend on Python experiment code or candidate-list post-processing.

### P1: High Priority Runtime Ownership

5. Replace Python reach-discard fallback with fail-closed Rust capability.
   - Current Python fallback: `DiscardOnlyMahjongEnv._python_reach_discard_candidates`.
   - Target: `rust/keqing_core/src/legal_actions.rs` reach discard candidates.
   - Reason: reach discard legality depends on shanten/waits and is easy to drift.

6. Add Rust action-feature parity before non-discard unlock.
   - Current Python owner: `DiscardOnlyMahjongEnv._action_features`.
   - Target: Rust feature builder or parity oracle covering all policy-visible action kinds before `REACH_DISCARD`, `PASS`, `PON`, `CHI`, or `KAN` is enabled.
   - Reason: action features become semantic once calls/kan/reach are policy-visible and must not lag behind legal enumeration.

7. Move KeqingRL state stepping/event apply to Rust-owned API after action/legal/terminal stability.
   - Target: `rust/keqing_core/src/event_apply.rs`, `state_core.rs`, `snapshot.rs`.
   - Python affected: `src/keqingrl/env.py`, `src/mahjong_env/state.py`.
   - Ordering constraint: keep this P1, but do it only after `ActionIdentity`, Rust legal enumeration, and terminal resolver contracts are stable.
   - Reason: reset/step/autopilot/replay/eval must share one deterministic state machine.

8. Consolidate shanten/waits/ukeire through Rust.
   - Target: `shanten.rs`, `shanten_table.rs`, possible new ukeire/waits API.
   - Python affected: replay/feature helpers that still compute waits with local recursion.
   - Reason: hot path, fuzzable, and already table-backed in Rust.

### P1.5: Evaluation Gate Readiness

9. Add Rust deterministic duplicate evaluation core before it becomes a selection gate.
   - Current Python owner: fixed-seed evaluation smoke in `src/keqingrl/training.py`.
   - Target: Rust-owned duplicate walls, deals, seat rotations, terminal scoring, and result trace API; Python remains the scheduler/reporter.
   - Reason: once duplicate eval affects checkpoint selection, policy comparison needs reproducible game cores independent from training rollout state.

### P2: Medium Priority Contract Stabilization

10. Rust-generate observation packed arrays after schema stabilizes.
   - Current Python owner: `encode_state_features` path used by `DiscardOnlyMahjongEnv.observe`.
   - Reason: observation should be derived from Rust state snapshots once the feature contract stops changing.

11. Replace hot-path JSON bridge with typed native APIs.
   - Current bridge: JSON wrappers in Python bindings and debug/export helpers.
   - Keep: JSON for golden snapshots, debug trace comparison, and human-readable regression artifacts.
   - Add: typed APIs for runtime legal enumeration, action identity, terminal resolver, feature arrays, and eventually state stepping.
   - Reason: JSON is useful as a contract artifact but should not be the scaled self-play hot path.

### P3: Keep Python Unless Requirements Change

- Do not rewrite `src/keqingrl/policy.py` to Rust.
- Do not rewrite `src/keqingrl/ppo.py` or GAE/advantage normalization to Rust.
- Do not rewrite `src/keqingrl/training.py` orchestration to Rust.
- Do not rewrite review JSONL/reporting to Rust while fields are still evolving.
- Do not freeze reward shaping/style context into Rust until the reward spec is stable.

## Suggested Implementation Order

1. Land `native_schema_info_json_py` and Python strict schema/capability checks.
2. Land Rust `ActionIdentity` v1 and Python parity tests.
3. Make KeqingRL env use Rust legal enumeration for the current controlled scope.
4. Land Rust terminal resolver and autopilot trace regression.
5. Clean up runtime fallback policy so runtime fails closed instead of silently falling back to Python.
6. Land action-feature Rust parity before `REACH_DISCARD` or other non-discard actions are unlocked.
7. Replace KeqingRL state stepping with Rust-owned event apply/snapshot API after action/legal/terminal contracts are stable.
8. Move fixed-seed duplicate game core to Rust before evaluation becomes a real selection gate.

## Contract Tests To Add Or Extend

- `tests/test_native_schema_contract.py`: native schema/version/capability checks fail closed on incompatible bindings.
- `tests/test_keqingrl_actions.py`: Rust/Python `ActionIdentity` parity, including canonical key and action id fields.
- `tests/test_rust_legal_actions_parity.py`: KeqingRL ordered legal action parity, including response windows.
- `tests/test_keqingrl_env_contract.py`: env must not rebuild or reorder legal actions between observe/step.
- `tests/test_replay_action_semantics_matrix.py`: terminal and call action semantic matrix against Rust truth.
- `tests/test_keqingrl_terminal_resolver.py`: terminal resolver decisions and autopilot traces match golden Rust outcomes.
- `tests/test_keqingrl_training.py`: fixed-seed eval remains separate from training rollouts.

## Non-Goals

- No Rust PPO trainer.
- No Rust neural network runtime for research training.
- No Rust review/report schema while interpretability fields are volatile.
- No broad cleanup of frozen `xmodel1` or `keqingv4` assets unless it narrows a shared semantic fallback.
