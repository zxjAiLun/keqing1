# Keqing1 Docs Index

Updated: 2026-04-28

## Active Control Surfaces

- `docs/project_progress.md`: primary live status board
- `docs/agent_sync.md`: coordination and handoff rules
- `docs/todo_2026_04_24.md`: latest dated execution snapshot
- `docs/project_overview_current.md`: current summary

## Active Mainline

### KeqingRL-Lite

- `docs/keqingrl/keqingrl_model_design_v1.md`: active model design
- `docs/keqingrl/mortal_training_workflow.md`: Mortal data packaging, no-ds3 training workflow, and runtime promotion
- `docs/mortal_action_contract.md`: active Mortal Q/mask/action mapping and teacher-gate correction notes
- `plans/mortal_training_runbook_2026_04_28.md`: operational Mortal training and KeqingRL teacher probe runbook
- `plans/keqingrl_lite_mainline_2026_04_24.md`: active implementation plan

Deprecated Mortal context:

- `plans/mortal_teacher_contract_2026_04_28.md`: historical `mortal-discard-q`
  bridge notes only. It must not be used as the active teacher plan or as
  evidence that discard-only no-pass proves Mortal teacher weakness.

## Frozen Asset References

### xmodel1

These documents are historical/reference material for asset extraction and baseline comparison, not active growth-mainline runbooks:

- `docs/xmodel1/xmodel1_model_design_v1.md`
- `docs/xmodel1/xmodel1_call_model_design_v1.md`
- `docs/xmodel1/xmodel1_schema_spec.md`
- `docs/xmodel1/xmodel1_design.md`
- `docs/xmodel1/xmodel1_eval_prd.md`

### keqingv4

These documents describe the frozen backup/runtime/Rust asset line:

- `docs/keqingv4/keqingv4_model_design_v2.md`
- `docs/keqingv4/keqingv4_v2_workflow.md`

### xmodel2

Bounded offline experiment references:

- `docs/xmodel2/xmodel2_model_design_v1.md`
- `docs/xmodel2/xmodel2_placement_design_v1.md`

## Rust

- `docs/rust_refactor/rust_core_architecture.md`
- `docs/rust_refactor/rust_progress_2026_04_17.md`

## Archive

Archive and older dated docs are historical context only. They must not override the active KeqingRL-Lite status board.
