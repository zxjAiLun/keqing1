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

## Phase 1.2 Fair Single-Domain Run

Additional same-model arena selfplay pools were generated on 2026-05-16:

| Pool | Checkpoint | Seed start | Seed key | Seed count | Hanchans | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 70k extra | `mortal_default_70k_promoted_candidate.pth` | `300125` | `8192` | `125` | `500` | `artifacts/experiments/selfplay_finetune_2026_05/pools/selfplay_70k_base_extra500_arena/logs/**/*.json.gz` |
| 80k extra | `mortal_default_80k_rejected_gate.pth` | `300125` | `8192` | `125` | `500` | `artifacts/experiments/selfplay_finetune_2026_05/pools/selfplay_80k_base_extra500_arena/logs/**/*.json.gz` |

The fair single-domain configs were generated under:

`artifacts/experiments/selfplay_finetune_2026_05/phase1_2_fair_1000`

They use:

- 70k data: original 500h 70k pool + extra 500h 70k pool
- 80k data: original 500h 80k pool + extra 500h 80k pool
- same `mortal_default` reward `[6,4,2,0]`
- no GRP/model/head/action-space changes

Training completion:

| Exp | Parent | Data | Target | Result | Checkpoint |
| --- | --- | --- | ---: | --- | --- |
| `S1_standard_selfplay` | 70k | 70k 1000h | 71000 | reached target | `phase1_2_fair_1000/S1_standard_selfplay/mortal.pth@71000` |
| `A1_aggressive_data_transfer` | 70k | 80k 1000h | 71000 | reached target | `phase1_2_fair_1000/A1_aggressive_data_transfer/mortal.pth@71000` |
| `A2_aggressive_lineage_continuation` | 80k | 80k 1000h | 81000 | reached target | `phase1_2_fair_1000/A2_aggressive_lineage_continuation/mortal.pth@81000` |

All three runs crossed the old single-pool exhaustion points and saved final checkpoints at the intended targets.

### Phase 1.2 Behavior Readout

Low-cost 100h same-model arena samples were generated with the same seed set as Phase 1.1:

- seed start: `310000`
- seed key: `8192`
- seed count: `25` (`100` half-games)
- output root: `artifacts/experiments/selfplay_finetune_2026_05/phase1_2_fair_1000/readout_100h`

This is still a behavior readout, not a strength gate.

L2 all-seat behavior deltas versus 70k anchor:

| Model | Rounds | Agari d | Houjuu d | Fuuro d | Riichi d | Post-fuuro agari d | Post-fuuro houjuu d | Dealer houjuu d | Rank-2 fuuro d | Rank-2 houjuu d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S1@71000` | 4368 | +0.55pp | -0.55pp | -1.73pp | -0.43pp | +3.38pp | -3.51pp | +1.61pp | -9.17pp | +3.64pp |
| `A1@71000` | 4080 | +1.70pp | +1.27pp | +2.12pp | -1.13pp | -1.47pp | +0.41pp | -0.89pp | -0.67pp | +5.02pp |
| `A2@81000` | 3908 | +0.99pp | +0.68pp | +1.09pp | -0.17pp | +2.12pp | -1.67pp | +2.51pp | -3.76pp | +3.34pp |

L3 chosen-call deltas versus 70k anchor:

| Model | Chosen-call count d | Avg margin d | Chosen-call agari d | Chosen-call houjuu d | Dealer call count d | Dealer call houjuu d | Rank-2 call count d | Rank-2 call houjuu d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S1@71000` | +36 | +4.3504 | +6.25pp | -6.13pp | -4 | -1.72pp | -88 | +6.43pp |
| `A1@71000` | +244 | +4.3175 | +1.27pp | +0.51pp | +68 | -8.54pp | +68 | +13.89pp |
| `A2@81000` | +136 | +4.5688 | +0.97pp | -3.46pp | +61 | -1.95pp | -26 | +7.82pp |

Paired divergence export counts versus 70k anchor:

| Model | Paired case counts |
| --- | --- |
| `S1@71000` | `paired_call_divergence=4`, `paired_dealer_call_divergence=4`, `paired_discard_divergence=10`, `paired_first_divergence=8`, `paired_reach_divergence=8` |
| `A1@71000` | `paired_call_divergence=8`, `paired_dealer_call_divergence=10`, `paired_discard_divergence=10` |
| `A2@81000` | `paired_call_divergence=4`, `paired_dealer_call_divergence=4`, `paired_discard_divergence=10` |

Readout interpretation:

- `S1@71000` is a viable standard-selfplay control: it does not become more aggressive overall, with lower fuuro (`-1.73pp`) and lower houjuu (`-0.55pp`) in this readout. It still has local pockets, especially start-rank-2 houjuu, so it should not be treated as behavior-identical to the 70k anchor.
- `A1@71000` is the clearest data-transfer style shift: fuuro rises (`+2.12pp`) and chosen calls rise strongly (`+244`). The risk is concentrated in rank-2 chosen calls (`+13.89pp` houjuu), while dealer calls look healthier (`-8.54pp` houjuu). This branch is useful but needs gate screening before any larger claim.
- `A2@81000` keeps an aggressive lineage signal with more fuuro (`+1.09pp`) and more chosen calls (`+136`) while overall chosen-call houjuu improves (`-3.46pp`). It still shows local risk in dealer/start-rank slices, so the branch should be screened rather than promoted.
- Q-margin magnitude remains scale-sensitive after fine-tuning. Use chosen-call counts and downstream outcome deltas as the primary readout, and use margin primarily as a direction/diagnostic signal.

Current Phase 1.2 conclusion: the fair +1000 single-domain runs completed successfully and produced coherent behavior separation. `A1` and `A2` are both worth 1000h screening versus the 70k anchor; `S1` can be screened as the standard-selfplay control if budget allows. Do not run a final 5000h bidirectional A/B until a branch passes 1000h screening.

### Phase 1.2 1000h Screening

`A1`, `A2`, and the existing `M1` mixed branch were screened against the 70k anchor with the same bidirectional seed set:

- seed start: `320000`
- seed key: `8192`
- seed count: `250` (`1000` half-games per direction)
- output root: `artifacts/experiments/selfplay_finetune_2026_05/phase1_2_fair_1000/gate_1000h`

Screening results:

| Direction | Challenger ranks | Avg rank | Tenhou pt | Gate read |
| --- | --- | ---: | ---: | --- |
| `A1@71000` vs `3x70k` | `[252,229,239,280]` | 2.547 | -4.815 | negative |
| `70k` vs `3xA1@71000` | `[251,266,269,214]` | 2.446 | +5.670 | negative for A1 |
| `A2@81000` vs `3x70k` | `[234,257,252,257]` | 2.532 | -2.070 | negative |
| `70k` vs `3xA2@81000` | `[260,247,263,230]` | 2.463 | +3.465 | negative for A2 |
| `M1@71000` vs `3x70k` | `[229,239,267,265]` | 2.568 | -4.410 | negative |
| `70k` vs `3xM1@71000` | `[267,240,254,239]` | 2.465 | +2.565 | negative for M1 |

Screening interpretation:

- `A1` has the clearest aggressive data-transfer behavior signal, but it fails the 1000h bidirectional screen. Do not advance it to final A/B in its current +1000 form.
- `A2` is behaviorally cleaner than `A1` in the 100h readout, but its 1000h screen is still negative in both directions. Do not advance it to final A/B in its current +1000 form.
- `M1` is the mixed-data branch and looked more conservative in the behavior readout, but it also fails the 1000h bidirectional screen. Do not use this exact mixed recipe as a promotion candidate.
- `S1` was not screened in this pass. It remains a standard-selfplay control/checkpoint-selection question, not a style candidate.

Current Phase 1.2 gate conclusion: selfplay fine-tune can produce measurable behavior/style drift, but none of the screened +1000 branches clears the 1000h screening gate against the 70k anchor. The next useful step is not final A/B; it is to analyze why selfplay fine-tune drifts lose strength, then try a more conservative recipe such as shorter updates, mixed-original data, or curriculum weighting before re-screening.

## Phase 1.3 Static Selfplay Diagnostics

Phase 1.3 tested whether the Phase 1.2 failures were caused by obvious offline-training mechanics rather than the static selfplay replay recipe itself.

### Target Variance Diagnostic

Script:

`scripts/mortal/diagnose_target_variance.py`

Artifacts:

- `artifacts/experiments/selfplay_finetune_2026_05/phase1_3/diagnostics/target_variance_10g.json`
- `artifacts/experiments/selfplay_finetune_2026_05/phase1_3/diagnostics/target_variance_200g.json`

The 200-game diagnostic compared original training logs with the 70k selfplay pool:

| Domain | Games | Steps | q_target std | q_target var | q_target p99 | steps/game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 200 | 34601 | 0.5047 | 0.2547 | 1.5290 | 173 |
| 70k selfplay | 200 | 31012 | 0.5297 | 0.2806 | 2.0093 | 155 |

Interpretation: selfplay target variance is slightly higher, but not enough to explain the consistent gate failures by itself. Target variance is not the primary blocker.

### CQL Ablation

Experiment:

- parent: 70k
- data: 70k selfplay 1000h
- update: +1000
- change: `cql.min_q_weight = 0`
- artifact root: `artifacts/experiments/selfplay_finetune_2026_05/phase1_3/cql_ablation`

1000h bidirectional screen:

| Direction | Challenger ranks | Avg rank | Tenhou pt | Read |
| --- | --- | ---: | ---: | --- |
| `S1_CQL0@71000` vs `3x70k` | `[236,246,247,271]` | 2.553 | -4.275 | negative |
| `70k` vs `3xS1_CQL0@71000` | `[252,248,245,255]` | 2.503 | -0.585 | near-neutral for 70k, still not positive for S1 |

Interpretation: removing CQL does not recover the selfplay fine-tune. CQL is not the main cause.

### Mixed Original/Selfplay 75:25

Experiment:

- parent: 70k
- data: original logs + 70k selfplay 1000h
- intended read: simple distribution repair with a majority original-data anchor
- artifact root: `artifacts/experiments/selfplay_finetune_2026_05/phase1_3/mixed_original`

1000h bidirectional screen:

| Direction | Challenger ranks | Avg rank | Tenhou pt | Read |
| --- | --- | ---: | ---: | --- |
| `S1_O75_S25@71000` vs `3x70k` | `[236,242,263,259]` | 2.545 | -2.835 | negative |
| `70k` vs `3xS1_O75_S25@71000` | `[260,259,257,224]` | 2.445 | +4.815 | negative for mixed branch |

Interpretation: a simple original/selfplay mix does not repair the issue. Do not keep sweeping mix ratios before answering the shorter-update question.

### S1 Short Update Control

This test asks whether static selfplay fine-tune is inherently harmful, or whether +1000 updates overshoots a mature 70k checkpoint.

Experiments:

| Exp | Parent | Data | Target | Update | Artifact |
| --- | --- | --- | ---: | ---: | --- |
| `S1_plus250` | 70k | 70k selfplay 1000h | 70250 | +250 | `artifacts/experiments/selfplay_finetune_2026_05/phase1_3/short_update/S1_plus250` |
| `S1_plus500` | 70k | 70k selfplay 1000h | 70500 | +500 | `artifacts/experiments/selfplay_finetune_2026_05/phase1_3/short_update/S1_plus500` |

Both checkpoints trained successfully with the default config, including `cql.min_q_weight = 5`.

1000h bidirectional screen, same protocol as Phase 1.2:

- seed start: `320000`
- seed key: `8192`
- seed count: `250`

| Direction | Challenger ranks | Avg rank | Tenhou pt | Read |
| --- | --- | ---: | ---: | --- |
| `S1_plus250@70250` vs `3x70k` | `[239,256,244,261]` | 2.527 | -2.205 | negative |
| `70k` vs `3xS1_plus250@70250` | `[249,265,242,244]` | 2.481 | +1.395 | negative for +250 |
| `S1_plus500@70500` vs `3x70k` | `[237,240,257,266]` | 2.552 | -3.780 | negative |
| `70k` vs `3xS1_plus500@70500` | `[257,255,258,230]` | 2.461 | +3.555 | negative for +500 |

Interpretation:

- Shortening to +250 does not recover a neutral/positive signal.
- +500 is more negative than +250 in this seed set.
- Together with the CQL ablation, target variance diagnostic, and 75:25 mixed-original result, this argues against further static batch selfplay fine-tune expansion as the next step.

Current Phase 1.3 conclusion: stop expanding static replay-based selfplay fine-tune recipes for now. The next experiment should be an online pilot, because the unresolved variable is now the data-generation loop itself: frozen batch pool versus continuously refreshed replay.

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
