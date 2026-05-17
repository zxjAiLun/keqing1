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

## Run Order

Start three processes with the generated `MORTAL_CFG`.

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
- checkpoint is saved

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

Phase 2 has been prepared at the config/runbook level. The next operational step is Phase 2.0 plumbing smoke.
