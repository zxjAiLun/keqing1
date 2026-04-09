# Repository Guidelines

## Current Project Focus
This repository is now primarily a **model-building workspace**.
The main goal is to design and iterate toward a **lightweight but strong Mahjong model**.

Treat the current priority order as:
1. model architecture
2. training pipeline
3. evaluation/review of model quality
4. runtime/replay/bot integration only as compatibility surfaces

Do not optimize for old bugfix workflows unless the current model work is directly blocked by them.

## Operating Principles
- Prefer model-strength improvements over local patching.
- Prefer architecture clarity over heuristic accumulation.
- Keep the model line lightweight enough to iterate quickly.
- Use Rust when preprocessing/runtime cost matters.
- Do not add hand-written tactical rules just to hide model weakness.
- Version major model-family changes explicitly (`keqingv4`, `keqingv5`, etc.).
- Keep diffs focused and reversible.

## Current Truths
- `processed_v3/ds1` and `processed_v3/ds2` are already the **fixed** datasets.
- The repo is in a **post-v3 validation** phase: old replay/bot/selfplay cleanup notes are no longer the center of work.
- The project has largely narrowed scope to **model construction and evaluation**.
- `xmodel1` is a **benchmark target**, not the design template.
- Main external design references are currently:
  - Suphx
  - Mortal
  - LuckyJ
  - LsAc-MJ

## Relevant Structure
- `src/keqingv1/`, `src/keqingv2/`, `src/keqingv3/`, `src/keqingv4/`: versioned model families
- `src/training/`: shared training infrastructure
- `src/mahjong_env/`: shared Mahjong state / analysis / rules utilities
- `src/inference/`: runtime model adapters and shared bot entrypoints
- `tests/`: regression and shape tests
- `docs/`: model design notes and planning artifacts
- `rust/`: Rust acceleration / schema / export support

## Model-Line Rules
- Do not mutate old model families casually.
- Small compatible changes may stay inside the active version.
- Any change that breaks feature/schema/checkpoint continuity should become a new model family.
- New architecture ideas belong in their own namespace, not retrofitted into older version packages.
- Keep training/evaluation interfaces explicit so different model families can be compared fairly.

## Design Intent
The design target is **not** merely to imitate human labels.
The design target is a model that can learn:
- offensive value
- defensive risk
- round-income / EV tradeoffs
- robust action comparison

Avoid spending project energy on explicit `good_shape`-style handcrafted tactical knowledge unless there is a very strong measured reason.
Simple counts/statistics are acceptable; hand-authored Mahjong doctrine is not the default path.

## Verification
For model work, verify before claiming progress:
- syntax/compile checks for edited Python modules
- focused model tests
- training smoke tests when touching training code
- evaluation artifacts or review evidence when making behavioral claims

At minimum, add or update a focused regression when changing:
- model tensor contracts
- trainer outputs
- cache/schema assumptions
- inference adapter behavior

## Final Reporting
When finishing a task, report:
- changed files
- what architectural simplification or design shift was made
- what was verified
- remaining risks / unknowns
