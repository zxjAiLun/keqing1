# Training

This directory owns Mortal training and research work:

- `mortal/` — self-play, preparation, evaluation, and training utilities;
- the legacy training helpers at this directory root;
- `configs/`, `docs/`, `plans/`, and `reports/` that describe those runs.

Mutable datasets, checkpoints, run outputs, and logs belong below the
repository `data/` root, not beside this source code.  The shared Mortal
runtime remains in `src/` for this first migration step because Workbench
still imports it in-process.
