# Mortal Online Training Phase 2

## Why Phase 2

Phase 1 static selfplay fine-tune has been narrowed down enough to justify switching training mode.

Ruled out so far:

- `S1` static selfplay `+1000` is negative.
- `CQL0` is worse or not recovered, so CQL is not the main cause.
- target variance is only mildly higher on selfplay and is not enough to explain the failures.
- `75:25 original:selfplay` does not repair the issue.
- shorter `S1_plus250` and `S1_plus500` also do not recover neutral performance.

The remaining high-value variable is the replay generation loop itself:

`frozen static replay pool` vs `online continuously refreshed replay`.

## Core Question

Phase 2 should not initially be framed as "online must improve strength".

The first question is:

Can the upstream Mortal online actor-learner loop run from the 70k anchor and avoid the systematic negative drift seen in static replay-based selfplay fine-tune?

## O1: 70k Online Control

| Item | Setting |
| --- | --- |
| Experiment id | `O1_70k_online_control` |
| Parent checkpoint | `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth` |
| Rollout trainee | current online trainee |
| Rollout baseline | fixed 70k anchor |
| Training mode | upstream Mortal `online=true` |
| CQL | not applied in Mortal online branch |
| Batch norm | `freeze_bn.mortal=true` |
| DataLoader workers | `dataset.num_workers=0` |
| Reward | `mortal_default` `[6,4,2,0]` |
| Initial read points | `70400`, `70800`, `71200` |

Do not start from 80k, A1, or A2 in the first online pilot. O1 isolates the training-mode change from aggressive-lineage/style effects.

## Important Semantics

When an offline checkpoint is switched into `online=true`, upstream Mortal loads model weights but does not restore the old optimizer/scheduler state unless the checkpoint was already online.

So O1 is:

`70k weights + fresh online optimizer/scheduler`

not:

`continuation of the original 70k optimizer state`.

## Config Generation

Generate the isolated config:

```bash
uv run python scripts/mortal/prepare_online_pilot.py \
  --base-config artifacts/mortal_training/config.toml \
  --output-root artifacts/experiments/online_phase2_2026_05 \
  --experiment-id O1_70k_online_control \
  --anchor-checkpoint artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth \
  --copy-parent-checkpoint
```

Output:

`artifacts/experiments/online_phase2_2026_05/O1_70k_online_control`

The generated manifest records the exact server, trainer, and client commands.

The generated online config sets `dataset.num_workers=0`. A first plumbing attempt with multiprocessing workers reached server/client replay transfer, but trainer hit the same environment-level `OSError: [Errno 95] Operation not supported` seen in earlier offline runs. Keeping workers at `0` is part of the O1 environment contract.

## Checkpoint Archive

Online training saves every `400` steps to the same `mortal.pth` path. Archive each read point before it is overwritten:

```bash
uv run python scripts/mortal/archive_online_checkpoints.py \
  --state-file artifacts/experiments/online_phase2_2026_05/O1_70k_online_control/mortal.pth \
  --output-dir artifacts/experiments/online_phase2_2026_05/O1_70k_online_control/checkpoints \
  --read-points 70400,70800,71200 \
  --manifest artifacts/experiments/online_phase2_2026_05/O1_70k_online_control/checkpoint_archive_manifest.jsonl \
  --watch
```

Expected archived files:

- `checkpoints/mortal_online_70400.pth`
- `checkpoints/mortal_online_70800.pth`
- `checkpoints/mortal_online_71200.pth`

Start the archive watcher before or alongside the trainer.

## Run Order

Start three processes with the generated `MORTAL_CFG`.

Archive watcher:

```bash
uv run python scripts/mortal/archive_online_checkpoints.py \
  --state-file /abs/path/to/O1_70k_online_control/mortal.pth \
  --output-dir /abs/path/to/O1_70k_online_control/checkpoints \
  --read-points 70400,70800,71200 \
  --manifest /abs/path/to/O1_70k_online_control/checkpoint_archive_manifest.jsonl \
  --watch
```

Server:

```bash
env MORTAL_CFG=/abs/path/to/config.toml \
  uv run python third_party/Mortal/mortal/server.py
```

Trainer:

```bash
env MORTAL_CFG=/abs/path/to/config.toml \
  uv run python third_party/Mortal/mortal/train.py
```

Client:

```bash
env MORTAL_CFG=/abs/path/to/config.toml \
  uv run python third_party/Mortal/mortal/client.py
```

Start server first, trainer second, client third.
The archive watcher can be started before trainer; it will wait until each read point appears.

## Phase 2.0 Plumbing Smoke

Goal: verify the loop, not evaluate strength.

Required signals:

- server starts and listens
- trainer submits idle params
- client pulls params
- client generates trainee-vs-70k replay
- client submits replay
- trainer drains replay
- trainer performs training steps
- `mortal.pth` is saved at `70400`
- `checkpoints/mortal_online_70400.pth` is archived

Abort and fix plumbing before interpreting any model result if one of these fails.

## Phase 2.1 Short Online Pilot

First read points:

| Checkpoint | Purpose |
| --- | --- |
| `70k` | anchor |
| `70400` | first short online read |
| `70800` | second short online read |
| `71200` | third short online read |

At each read point:

1. Run 100h behavior readout versus the 70k anchor.
2. If behavior is not obviously broken, run 1000h bidirectional screening against 70k.
3. Do not run final 5000h A/B until a checkpoint passes 1000h screening.

## Success Signal

A useful O1 checkpoint should show:

- challenger direction near zero or positive
- reverse direction not clearly exploitable by 70k
- no obvious behavior collapse in fuuro/riichi/agari/houjuu

This would justify a longer online run.

## Failure Signal

If `+400`, `+800`, and `+1200` all look like static selfplay drift:

- online replay refresh alone is not enough
- inspect online reward, baseline setup, exploration, trainer/client sampling, and evaluation protocol before scaling

## Current Status

Phase 2.0 plumbing smoke has passed.

Observed on `2026-05-17`:

- server listened on `127.0.0.1:5000`
- trainer loaded the 70k checkpoint on `cuda:0`
- trainer submitted idle params
- client pulled params and generated trainee-vs-70k replay batches
- client submitted replay logs
- server transferred logs to trainer
- trainer drained replay batches and performed online train steps
- archive watcher preserved all read points

Archived checkpoints:

| Step | Path |
| ---: | --- |
| `70400` | `artifacts/experiments/online_phase2_2026_05/O1_70k_online_control/checkpoints/mortal_online_70400.pth` |
| `70800` | `artifacts/experiments/online_phase2_2026_05/O1_70k_online_control/checkpoints/mortal_online_70800.pth` |
| `71200` | `artifacts/experiments/online_phase2_2026_05/O1_70k_online_control/checkpoints/mortal_online_71200.pth` |

Operational notes:

- A first smoke attempt reached client replay submission and trainer drain, then failed in trainer DataLoader multiprocessing with `OSError: [Errno 95] Operation not supported`.
- The config generator was updated to set `dataset.num_workers=0`; the second smoke passed with that setting.
- Each `800` hanchan client batch produced roughly `245-250` train batches, so reaching `70400/70800/71200` required multiple replay batches.
- Trainer and client sharing the same GPU caused visible throughput contention. This is acceptable for plumbing smoke, but future longer online runs should consider client/trainer device scheduling or multiple clients only after measuring the bottleneck.
- Client-side trainee-vs-70k rankings during smoke are diagnostic logs only; they are not strength conclusions.

## Phase 2.2 Online Recipe Findings

The initial online runs narrowed the failure mode from "online replay refresh is enough" to "online continuation is still sensitive to optimizer state, replay freshness, and update size."

| Experiment | Main change | Readout summary | Interpretation |
| --- | --- | --- | --- |
| `O1_70k_online_control` | Fresh online optimizer/scheduler from 70k weights | `70400` failed | Resetting optimizer state was not a neutral continuation recipe. |
| `O2_70k_online_keep_optimizer` | Keep optimizer state | `70400` was the best verified point; `70800` turned negative | Keeping optimizer repaired the first short update but did not create a stable growth line. |
| `O3_70k_online_keep_optimizer_cql5` | Keep optimizer + CQL5 + `8000` games/iteration | Later checkpoints degraded; `80000` was much worse | Large rollout batches likely made actor-learner lag / stale replay a first-order problem. |
| `O4_70k_online_keep_optimizer_cql5_smallbatch` | O3 with `800` games/iteration | `70800` was near neutral; `71200` failed | Smaller batches delayed but did not eliminate drift. |

Do not continue extending O3/O4 with the same recipe as the main line. The 400-step read points remain useful as a safety valve for recipe changes, not as a high-throughput growth loop by themselves.

## Phase 2.3 O5 Targeted Sweep

O5 tests a small, hypothesis-driven matrix instead of broad parameter search. Keep O4's fixed choices unless listed below: 70k anchor, kept optimizer state, CQL5, `freeze_bn.mortal=true`, `dataset.num_workers=0`, `save_every=400`, `submit_every=400`, and `online.server.capacity=1600`.

| Experiment | `train_play.default.games` | Scheduler `peak/final` | Question |
| --- | ---: | ---: | --- |
| `O5a_800g_lr5e5` | `800` | `5e-5` | Does reducing LR alone prevent the O4 `71200` failure? |
| `O5b_400g_lr1e4` | `400` | `1e-4` | Does fresher rollout alone extend the stable window? |
| `O5c_400g_lr5e5` | `400` | `5e-5` | Does lower LR plus fresher rollout work better than either single change? |

Generate each config with `scripts/mortal/prepare_online_pilot.py`; use `--train-play-games` for freshness and `--peak-lr/--final-lr` for the scheduler override. Archive `70400`, `70800`, and `71200`, but judge O5 primarily from `70800` and `71200`.

Suggested first run:

```bash
uv run python scripts/mortal/prepare_online_pilot.py \
  --base-config artifacts/experiments/online_phase2_2026_05/O4_70k_online_keep_optimizer_cql5_smallbatch/config.toml \
  --output-root artifacts/experiments/online_phase2_2026_05 \
  --experiment-id O5a_800g_lr5e5 \
  --anchor-checkpoint artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth \
  --train-play-games 800 \
  --peak-lr 5e-5 \
  --final-lr 5e-5 \
  --copy-parent-checkpoint
```

Evaluation policy:

1. Run short behavior readout before any large gate.
2. Track win rate, deal-in rate, `fuuro_rate`, `avg_fuuro_num`, riichi rate, and Tenhou rank pt.
3. Only promote a checkpoint to 1000h bidirectional screening if the short readout is near neutral or better and does not show behavior collapse.

### O5 Results (2026-05-22)

All three O5 variants ran. O5a and O5b were gated at 70800. O5c was not run because both single-variable changes degraded independently.

| Recipe | games/iter | LR | @70800 challenger | @70800 reverse | vs O4 |
| --- | ---: | ---: | ---: | ---: | --- |
| **O4** | 800 | 1e-4 | **-0.17** | +2.06 | baseline |
| O5a | 800 | 5e-5 | -5.04 | +9.18 | worse |
| O5b | 400 | 1e-4 | -3.69 | +3.96 | worse |

O5a@71200 partially recovered to -1.85/+1.94 but still worse than O4@70800.

Key findings:
- Lowering LR to 5e-5 degrades performance.
- Reducing rollout batch to 400 games/iter also degrades performance.
- The 800g/1e-4 combination is a local sweet spot: fresh enough to avoid the stale-replay collapse seen at 8000g, but not so small that training signal becomes noisy or unstable.

### O5c Decision

O5c (400g + 5e-5) was skipped. Both 400g and 5e-5 independently degrade @70800. Combining them has low priority and low expected value.

## O-Series Conclusion

The best online continuation recipe found so far is **O4@70800**: keep optimizer/scheduler + CQL5 + 800 games/iter + LR=1e-4.

O4@70800 is near-neutral in challenger direction but still exploitable in reverse direction. It does not outperform the 70k anchor and does not approach model_v4 strength.

The scalar sweep (O5) confirmed that further tuning LR or rollout batch within the current recipe space does not yield additional improvement. O4@70800 is the local optimum for the current trainee-vs-fixed-70k-baseline online setup.

## O-Series Checkpoint Inventory

Current as of 2026-05-23 local inspection. The common 70k parent/anchor is:

`artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth`

Per-experiment `mortal.pth` files are mutable final/latest files for that run. Use the archived `checkpoints/mortal_online_*.pth` files for exact readout points:

| Experiment | Latest `mortal.pth` step | Archived readout checkpoints |
| --- | ---: | --- |
| `O1_70k_online_control` | 71200 | `checkpoints/mortal_online_70400.pth`, `checkpoints/mortal_online_70800.pth`, `checkpoints/mortal_online_71200.pth` |
| `O2_70k_online_keep_optimizer` | 71200 | `checkpoints/mortal_online_70400.pth`, `checkpoints/mortal_online_70800.pth`, `checkpoints/mortal_online_71200.pth` |
| `O3_70k_online_keep_optimizer_cql5` | 80000 | `checkpoints/mortal_online_70400.pth`, `checkpoints/mortal_online_70800.pth`, `checkpoints/mortal_online_80000.pth` |
| `O4_70k_online_keep_optimizer_cql5_smallbatch` | 73200 | `checkpoints/mortal_online_70400.pth`, `checkpoints/mortal_online_70800.pth`, `checkpoints/mortal_online_71200.pth` |
| `O5a_800g_lr5e5` | 73600 | `checkpoints/mortal_online_70400.pth`, `checkpoints/mortal_online_70800.pth`, `checkpoints/mortal_online_71200.pth` |
| `O5b_400g_lr1e4` | 73200 | `checkpoints/mortal_online_70400.pth`, `checkpoints/mortal_online_70800.pth`, `checkpoints/mortal_online_71200.pth` |
| `O5c_400g_lr5e5` | 70000 | No archived online readout checkpoints found; this remained at the 70k parent checkpoint during inspection. |

The best O-series read point found so far is `O4_70k_online_keep_optimizer_cql5_smallbatch/checkpoints/mortal_online_70800.pth`, not that experiment's later mutable `mortal.pth`.

## Next Directions

Do not continue O5/O6 as another scalar LR/batch sweep. The next useful changes are structural:

1. **Reviewer Teacher Probe** (recommended priority): Use public reviewer networks such as `4.1b` as black-box preference labelers on already generated logs. They are not local rollout generators because their weights are not available.

2. **model_v4 teacher replay transfer**: Use the local `model_v4` checkpoint as a local data generator, then test whether 70k can benefit from model_v4 demonstration logs. Treat the first pass as a feasibility test if it only uses the existing offline DQN/CQL loss.

3. **Opponent/Data Curriculum** (only after the teacher route is scoped): Replace the fixed 70k baseline rollout with a mixed opponent pool such as 70k + 80k_game + model_v4. This tests whether the fixed-baseline ecology limits the trainee's state distribution, but it is higher-risk than teacher replay transfer.

The recommended next training route is documented in `docs/mortal/reviewer_teacher_probe_2026_05.md` as T-series teacher transfer. The immediate practical priority is T1, not another O-series scalar sweep.
