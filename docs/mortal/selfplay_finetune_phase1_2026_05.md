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
