# Mortal Selfplay Fine-tune Phase 1

## Framing

70k and 80k are now treated as behavior anchors, not as a final winner/loser pair.

| Anchor | Role | Checkpoint |
| --- | --- | --- |
| 70k | Standard reference / balanced anchor | `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth` |
| 80k | Aggressive / fuuro-heavy behavior anchor | `artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth` |

The prior 1000h bidirectional gate blocks automatic 80k promotion, but it is only a screening result over 250 seed blocks with seat rotations. The 80k checkpoint remains useful as an aggressive selfplay behavior source.

## Phase 1 Matrix

| Exp | Parent | Training data | Question |
| --- | --- | --- | --- |
| `S1_standard_selfplay` | 70k | 70k selfplay logs | Does standard selfplay fine-tune stay stable? |
| `A1_aggressive_data_transfer` | 70k | 80k selfplay logs | Can aggressive data pull the standard model toward fuuro/attack behavior? |
| `A2_aggressive_lineage_continuation` | 80k | 80k selfplay logs | Can the 80k aggressive lineage continue as a stable branch? |
| `M1_mixed_selfplay` | 70k | 70k + 80k selfplay logs | Does mixed behavior data produce a steadier middle branch? |

All experiments keep Mortal/libriichi encoding, Brain, DQN action space, GRP structure, and model heads unchanged. Reward stays `mortal_default` `[6,4,2,0]`; Tenhou `[90,45,0,-135]` is only the reporting reference.

## Data

Default data inputs:

- 70k selfplay: `artifacts/experiments/grp_audit/selfplay_70k_base_500h_arena/logs/**/*.json.gz`
- 80k selfplay: `artifacts/experiments/grp_audit/selfplay_80k_base_500h_arena/logs/**/*.json.gz`

These are all-same arena logs generated for GRP audit and behavior diagnostics. They are enough for the first small policy-shaping pass; later larger pools should be generated if a branch shows useful direction.

## Config Generation

Generate isolated configs without overwriting main training:

```bash
uv run python scripts/mortal/prepare_selfplay_finetune_experiments.py \
  --base-config artifacts/mortal_training/config.toml \
  --output-root artifacts/experiments/selfplay_finetune_2026_05 \
  --train-steps 1000 \
  --copy-parent-checkpoint
```

The script writes one config per experiment, with independent `state_file`, `tensorboard_dir`, `file_index`, and output directories. Default target steps are `parent_steps + train_steps`, so 70k-parent experiments target 71000 and the 80k-parent experiment targets 81000.

Dry run:

```bash
uv run python scripts/mortal/prepare_selfplay_finetune_experiments.py \
  --base-config artifacts/mortal_training/config.toml \
  --output-root artifacts/experiments/selfplay_finetune_2026_05 \
  --train-steps 1000 \
  --dry-run
```

## Training Commands

The generated manifests contain exact `training_command` arrays. Expected commands:

```bash
uv run python scripts/run_mortal_dqn_offline.py \
  --config artifacts/experiments/selfplay_finetune_2026_05/S1_standard_selfplay/config.toml \
  --target-steps 71000 \
  --num-workers 0

uv run python scripts/run_mortal_dqn_offline.py \
  --config artifacts/experiments/selfplay_finetune_2026_05/A1_aggressive_data_transfer/config.toml \
  --target-steps 71000 \
  --num-workers 0

uv run python scripts/run_mortal_dqn_offline.py \
  --config artifacts/experiments/selfplay_finetune_2026_05/A2_aggressive_lineage_continuation/config.toml \
  --target-steps 81000 \
  --num-workers 0

uv run python scripts/run_mortal_dqn_offline.py \
  --config artifacts/experiments/selfplay_finetune_2026_05/M1_mixed_selfplay/config.toml \
  --target-steps 71000 \
  --num-workers 0
```

## First-Pass Readout

Do not treat this first round as a champion tournament. First ask whether the branches produce distinct, stable behavior.

Runtime smoke and capacity probe, run on 2026-05-15:

| Exp | Smoke target | 1k target attempt | Result | Last saved checkpoint |
| --- | ---: | ---: | --- | ---: |
| `S1_standard_selfplay` | 70100 | 71000 | single 500h 70k pool exhausted at 70714 | 70400 |
| `A1_aggressive_data_transfer` | 70100 | 71000 | single 500h 80k pool exhausted at 70691 | 70400 |
| `A2_aggressive_lineage_continuation` | 80100 | 81000 | single 500h 80k pool exhausted at 80691 | 80400 |
| `M1_mixed_selfplay` | 70100 | 71000 | mixed 1000h pool reached target | 71000 |

All four 100-step smoke runs succeeded: CUDA, dataloader, independent `file_index`, and checkpoint saving all worked. The single-pool exhaustion is a data-capacity finding, not an optimization failure. Do not force extra epochs without explicitly deciding to allow replay reuse.

The current reusable checkpoints are:

- `S1_standard_selfplay`: `artifacts/experiments/selfplay_finetune_2026_05/S1_standard_selfplay/mortal.pth@70400`
- `A1_aggressive_data_transfer`: `artifacts/experiments/selfplay_finetune_2026_05/A1_aggressive_data_transfer/mortal.pth@70400`
- `A2_aggressive_lineage_continuation`: `artifacts/experiments/selfplay_finetune_2026_05/A2_aggressive_lineage_continuation/mortal.pth@80400`
- `M1_mixed_selfplay`: `artifacts/experiments/selfplay_finetune_2026_05/M1_mixed_selfplay/mortal.pth@71000`

## Phase 1.1 Behavior Readout

Low-cost 100h same-model arena samples were generated with the same seed set:

- seed start: `310000`
- seed key: `8192`
- seed count: `25` (`100` half-games)
- output root: `artifacts/experiments/selfplay_finetune_2026_05/readout_100h`

This is a behavior readout only. It is not a strength gate, and the samples are too small for promotion decisions.

L2 all-seat behavior deltas versus 70k anchor:

| Model | Rounds | Agari d | Houjuu d | Fuuro d | Riichi d | Post-fuuro houjuu d | Dealer houjuu d | Rank-2 fuuro d | Rank-2 houjuu d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anchor_80k` | 4416 | +0.86pp | +1.67pp | +1.50pp | +1.45pp | -0.78pp | +0.36pp | -2.93pp | +1.32pp |
| `S1@70400` | 4448 | -0.56pp | -0.77pp | -3.04pp | +0.94pp | +0.54pp | +2.06pp | -8.56pp | +3.75pp |
| `A1@70400` | 4136 | +0.95pp | +1.57pp | +0.19pp | +2.15pp | -1.46pp | +3.69pp | -2.10pp | -0.99pp |
| `A2@80400` | 3816 | +1.87pp | +2.10pp | +1.66pp | -0.36pp | +0.91pp | +3.31pp | +1.72pp | +5.87pp |
| `M1@71000` | 3968 | +0.83pp | +0.35pp | -2.52pp | -0.82pp | -3.05pp | +1.88pp | -2.52pp | +1.01pp |

L3 chosen-call deltas versus 70k anchor:

| Model | Chosen-call count d | Avg margin d | Chosen-call houjuu d | Dealer call count d | Rank-2 call count d |
| --- | ---: | ---: | ---: | ---: | ---: |
| `anchor_80k` | +392 | +0.3142 | -0.17pp | +120 | +32 |
| `S1@70400` | +12 | +4.3366 | -0.36pp | -40 | -72 |
| `A1@70400` | +189 | +3.3515 | -3.53pp | +15 | +22 |
| `A2@80400` | +177 | +4.2441 | -0.72pp | +12 | +58 |
| `M1@71000` | -132 | +3.5346 | -4.17pp | -60 | -4 |

Paired divergence export counts versus 70k anchor:

| Model | Paired case counts |
| --- | --- |
| `S1@70400` | `paired_discard_divergence=10`, `paired_first_divergence=4` |
| `A1@70400` | `paired_call_divergence=4`, `paired_dealer_call_divergence=4`, `paired_discard_divergence=10`, `paired_first_divergence=4` |
| `A2@80400` | `paired_call_divergence=4`, `paired_discard_divergence=10` |
| `M1@71000` | `paired_call_divergence=4`, `paired_discard_divergence=10`, `paired_first_divergence=4` |

Readout interpretation:

- `A1` already shows the intended data-transfer signal: chosen-call count rises substantially (`+189`) while all-seat fuuro rate is nearly flat (`+0.19pp`) and riichi rises (`+2.15pp`). This is a useful early style-shaping signal, not a strength claim.
- `A2` is the clearest aggressive-lineage signal: fuuro rises (`+1.66pp`), agari rises (`+1.87pp`), but houjuu also rises (`+2.10pp`), with rank-2 houjuu especially high (`+5.87pp`). This branch needs a larger fair run before any gate.
- `M1` completed the full +1000 updates and looks more conservative in this 100h readout: fuuro drops (`-2.52pp`), riichi drops (`-0.82pp`), and post-fuuro houjuu improves (`-3.05pp`). Because it has +1000 updates while S1/A1/A2 have +400 saved updates, do not compare it as a fair recipe result yet.
- `S1` did not simply stay identical at +400; fuuro drops and dealer/rank-2 houjuu pockets move. This supports doing a larger fair single-domain run before interpreting standard selfplay behavior.
- Fine-tuned branches show much larger `Q(call)-Q(pass)` margin shifts than the 80k anchor. Treat Q-margin magnitude as a scale-sensitive diagnostic here; use action counts and outcome deltas as the primary first-pass readout.

Current Phase 1.1 conclusion: behavior differences are already visible enough to justify building fair 1000h single-domain pools for S1/A1/A2. Do not run 1000h strength gates yet.

Primary style metrics:

- fuuro rate
- riichi rate
- agari rate
- houjuu rate
- post-fuuro agari / houjuu
- chosen-call count and `Q(call)-Q(pass)` margin
- paired divergence distribution versus the 70k anchor

Strength screening:

- 1000h gate versus the 70k anchor
- only promote to larger 5000h bidirectional A/B if behavior direction is coherent and the 1000h screen is not obviously bad

## Decision Rules

- `S1` should not drift heavily; it is the standard selfplay control.
- `A1` is useful if 80k data changes 70k behavior toward higher fuuro/attack without immediate collapse.
- `A2` is useful if the 80k lineage can continue as a coherent aggressive branch, regardless of whether it becomes the standard mainline.
- `M1` is useful if it lands between S1 and A1/A2 or improves stability.
- GRP remains unchanged unless a future selfplay audit shows both calibration issues and expected-PT/reward-delta degradation.
