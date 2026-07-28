# Legal-Mean Value Objective A/B (2026-07)

## Status

This is a pre-registered objective experiment. The implementation, seed-aware
preflight, finite-diagnostic guards, and Windows pipeline are committed.
All three matched seeds have completed both objectives through step 72000 and
passed the archive/data-stream correctness gate. No checkpoint promotion has
been made.

The only planned variable is where the `final_rank_mc` target is applied to
the legal Q table:

- Control `C_behavior_action_mc`: `0.5 * (Q(s, a_behavior) - y)^2`.
- Variant `V_legal_mean_mc`: `0.5 * (mean_legal(Q(s, .)) - y)^2`.

The network, DQN output, legal-action mask, CQL preference loss, next-rank
auxiliary loss, reward, optimizer state, corpus, scheduler and evaluation
protocol remain unchanged. The variant adds no parameters and keeps the same
checkpoint/inference format.

## Fixed Contract

- Parent: `mortal_default_70k_promoted_candidate.pth` at step `70000`.
- Parent Adam state: preserved from the same checkpoint.
- Scheduler, scaler and data stream: fresh.
- Corpus: M0 mixed route, the existing 6,000-file index.
- Reward: `final_rank_mc`, rank points `[6, 4, 2, 0]`.
- CQL weight: `5.0`.
- Next-rank weight: `0.2`.
- Architecture and learning rate: unchanged.
- Matched model/data seeds: `20260803`, `20260804`, `20260805`.
- Target: `72000`.
- Archive steps: `70001`, `70010`, `70100`, `70500`, `71000`, `72000`.

The preparation entry point is
[`prepare_legal_mean_objective.py`](../../scripts/mortal/prepare_legal_mean_objective.py).
The Windows runner is
[`run_legal_mean_value_ab_2026_07.ps1`](../../scripts/mortal/run_legal_mean_value_ab_2026_07.ps1).
Run [`preflight_legal_mean_objective.py`](../../scripts/mortal/preflight_legal_mean_objective.py)
for each matched pair before training.

## Training Result

The three pair verifications are stored under the ignored local experiment
artifact directory:

`artifacts/experiments/model_pool_2026_07/legal_mean_value_ab_2026_07/preflight/verification_20260803.json`

and the corresponding `20260804` and `20260805` files. Each pair reached
`72000`, consumed `2000` batches / `1024000` samples per arm, and contains all
six required archive checkpoints. The first pair was trained at commit
`b1eb568`; the later two at `8c130c1`. The latter is a verifier-only fix and
does not change the training/objective implementation; the per-checkpoint
contract records this provenance explicitly.

## Required Checks

The objective helper tests must pass before generation:

```powershell
.venv-win\Scripts\python.exe -m pytest -q tests/test_mortal_objective.py
```

The preflight must confirm that the control and variant configs differ only
in `[objective].mode`, that both use `final_rank_mc`, and that the 70k parent
contains the Adam state. The runner records the objective contract in every
checkpoint and exposure report.

The loss helper also verifies:

- the control objective matches the legacy behavior-action MC, CQL and
  next-rank loss values;
- CQL is invariant to a per-state common Q offset;
- legal-mean value gradients are equal across legal actions for a state;
- CQL Q-output gradients sum to zero across legal actions.

## Evaluation

After all six runs reach 72k, evaluate each matched pair in the same native
four-model random-seat pool:

`70k / ext_mortal / C_behavior_action_mc / V_legal_mean_mc`

Report complete-hanchan paired differences for `V-C`, `V-70k`, and `C-70k`
with seed means/medians, positive seed counts, exact sign tests, and
hanchan-cluster plus equal-seed hierarchical bootstrap intervals. Also audit
the archived checkpoints for centered-advantage drift and margin expansion.

The variant can enter replication only if all of the following are observed
in the first three seeds: positive `V-C` in every seed, mean and median near
or above `+3 Pt` per hanchan, lower centered-advantage drift and margin
expansion than control, and no systematic degradation against 70k.

This first A/B does not promote a checkpoint. A recipe promotion requires a
later six-seed replication; only then may a separately trained candidate be
entered into the model-pool promotion gate.

Before the formal 3 x 1000-hanchan evaluation, a 25-hanchan CUDA smoke was
completed with the same lineup and rank points `[90,45,0,-135]`. It generated
25 native logs plus `metrics.json`, detailed stats, and platform-account
artifacts. Its observed throughput was about 25 hanchans per 8 minutes with
the current Python-engine four-player path, so the formal evaluation remains
pending a batch-throughput decision; the smoke result is not a strength
judgement.
