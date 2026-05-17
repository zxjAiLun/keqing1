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

Next step: run 100h behavior readout for `70400`, `70800`, and `71200`, then select at most one checkpoint for 1000h bidirectional screening against the 70k anchor.
