# Keqing1 Agent Sync

Updated: 2026-05-10

## Short Handoff Summary

- Active project mainline is Mortal-based enhancement / fine-tuning / selfplay /
  evaluation / tooling.
- Core runtime direction is `Mortal + riichienv + libriichi`, with
  Mortal/libriichi v4 observation as the default model input contract.
- KeqingRL independent Mortal Action-Q imitation is archived experiment context,
  not the default strength route.
- `xmodel1`, `xmodel2`, `keqingv3`, `keqingv3.1`, and `keqingv4` are frozen or
  archive-only assets. Do not treat them as active teachers, baselines, or
  retrain candidates.
- Useful Mortal tools now live under `scripts/mortal/`; old script paths remain
  deprecated wrappers for compatibility.
- Strength work must start from Mortal/libriichi encoding and
  Mortal-compatible Brain/DQN checkpoints. Do not revive a custom observation
  stack unless the task explicitly changes the project direction.
- `docs/mortal/deep_research_sync_2026_05_10.md` is the concise sync of the two
  deep research reports and should be read before changing Mortal training,
  reward, or model-head design.

## Read Before Work

1. `docs/project_overview_current.md`
2. `docs/project_progress.md`
3. `docs/docs_index.md`
4. `docs/mortal/mainline_pivot_2026_05_09.md`
5. `docs/mortal/deep_research_sync_2026_05_10.md`
6. `docs/mortal/archive_decisions_2026_05.md`
7. `docs/mortal_action_contract.md`
8. `plans/mortal_training_runbook_2026_04_28.md`
9. `git status --short`

## Current Alignment

Default work should fit one of these surfaces:

- Mortal selfplay replay generation
- Mortal replay sidecar / Q-value / mask export
- Mortal checkpoint evaluation and comparison
- Mortal fine-tuning using Mortal's existing training stack
- Mortal-native continued/selfplay training from Mortal checkpoints
- Mortal reward-only experiments through GRP/rank-point scalarization
- future policy/value heads attached to Mortal-compatible `Brain(obs)`
- GUI replay/decision review tooling

## Deprecated But Still Present

The full KeqingRL/xmodel/keqingv experiment state is preserved by tag
`archive-keqingrl-mortal-imitation-202605`.

Do not extend them by default. If a task needs one of these routes, state why it
is deliberately leaving the Mortal mainline.

## Coordination Rules

- Do not delete archived routes during cleanup unless the task explicitly asks
  for code movement/removal and tests/imports are handled.
- Prefer adding archive markers before moving KeqingRL files; moving the package
  can break imports and tests.
- Keep old Mortal tool paths as thin wrappers until downstream commands are
  updated.
- Use `riichienv + Mortal/libriichi` as the default environment/legal runtime.
- Keep Mortal/libriichi observation encoding as the default model input contract.
- Keep `rust/keqing_core/` frozen as compatibility/research reference unless a
  specific tool still depends on it.
- Discard-only no-pass probes are diagnostic context only; they are not Mortal
  strength evidence.
- Paired eval should not become a blocker before Mortal workflow progress.

## Minimum Handoff Output

When finishing a work chunk, leave:

- files changed
- tests run
- tests blocked
- contract risks still open
- next concrete step
