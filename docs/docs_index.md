# Keqing1 Docs Index

Updated: 2026-05-10

## Active Control Surfaces

- `docs/project_overview_current.md`: current project direction and boundaries
- `docs/project_progress.md`: live status board
- `docs/agent_sync.md`: coordination and handoff rules
- `docs/mortal/mainline_pivot_2026_05_09.md`: Mortal encoding mainline decision
- `docs/mortal/deep_research_sync_2026_05_10.md`: synchronized architecture,
  progress, and development plan from the two Mortal deep research reports
- `docs/mortal/archive_decisions_2026_05.md`: project contraction/archive decision

## Active Mainline

### Mortal Fine-Tuning / Selfplay / Evaluation

- `docs/mortal_action_contract.md`: Mortal Q/mask/action mapping and prior teacher-gate corrections
- `docs/mortal/deep_research_sync_2026_05_10.md`: native-first Mortal
  architecture understanding and implementation priority
- `plans/mortal_training_runbook_2026_04_28.md`: operational Mortal training runbook
- `scripts/mortal/generate_riichienv_selfplay_replays.py`: current 4-Mortal RiichiEnv selfplay replay generator
- `scripts/mortal/materialize_replay_sidecars.py`: current Mortal Q/mask decision sidecar exporter

Active model work should use Mortal/libriichi observation encoding, Mortal Brain,
and Mortal Dueling DQN checkpoints. Future policy/value/rank heads belong on
that compatible backbone unless a later control document says otherwise.

## Archived Experiments

### KeqingRL Mortal Action-Q Imitation

Archived experiment records are preserved by tag
`archive-keqingrl-mortal-imitation-202605`. They are no longer active files in
the working tree.

### Deprecated Mortal Context

- `plans/mortal_teacher_contract_2026_04_28.md`: historical `mortal-discard-q`
  bridge notes only. It must not be used as the active teacher plan or as
  evidence that discard-only no-pass proves Mortal teacher weakness.

## Archive-Only Asset References

### xmodel1

Historical/reference material only. Not baseline, teacher, or retrain candidate:

- `docs/xmodel1/xmodel1_model_design_v1.md`
- `docs/xmodel1/xmodel1_call_model_design_v1.md`
- `docs/xmodel1/xmodel1_schema_spec.md`
- `docs/xmodel1/xmodel1_design.md`
- `docs/xmodel1/xmodel1_eval_prd.md`

### keqingv4

Frozen backup/runtime/Rust asset line only:

- `docs/keqingv4/keqingv4_model_design_v2.md`
- `docs/keqingv4/keqingv4_v2_workflow.md`

### xmodel2

Historical bounded offline experiment references:

- `docs/xmodel2/xmodel2_model_design_v1.md`
- `docs/xmodel2/xmodel2_placement_design_v1.md`

## Rust

Frozen compatibility/research reference. Do not let it compete with
`riichienv + Mortal/libriichi` as the default main environment:

- `docs/rust_refactor/rust_core_architecture.md`
- `docs/rust_refactor/rust_progress_2026_04_17.md`

## Archive

Archive and older dated docs are historical context only. They must not override
the active Mortal-based mainline.
