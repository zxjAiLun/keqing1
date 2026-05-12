# GRP Audit Report

Status: scaffolded, audit not yet run.

## Goal

Audit the existing GRP before any retraining. Changing the PT table only changes rank-probability scalarization, so new GRP training is gated on calibration evidence, not assumed.

## Inputs

- Old GRP: `artifacts/mortal_training/grp.pth`
- Original logs: `artifacts/mortal_mjai_gz/train/**/*.json.gz`
- Style selfplay logs: `artifacts/eval/style_selfplay_alpha_grid_semantic_20260512_140408/**/*.json.gz`

## Commands

Original-log smoke:

```bash
uv run python scripts/mortal/audit_grp_on_logs.py \
  --grp-checkpoint artifacts/mortal_training/grp.pth \
  --logs 'artifacts/mortal_mjai_gz/train/**/*.json.gz' \
  --limit-games 2 \
  --max-prefixes-per-game 4 \
  --output artifacts/experiments/grp_audit/original_smoke_metrics.json
```

Style-log smoke:

```bash
uv run python scripts/mortal/audit_grp_on_logs.py \
  --grp-checkpoint artifacts/mortal_training/grp.pth \
  --logs 'artifacts/eval/style_selfplay_alpha_grid_semantic_20260512_140408/**/*.json.gz' \
  --limit-games 2 \
  --max-prefixes-per-game 4 \
  --output artifacts/experiments/grp_audit/style_smoke_metrics.json
```

## Decision Gate

Consider GRP retraining only if selfplay-domain calibration is clearly worse than original logs and the reward-delta variance/error under the active PT profiles is large enough to make DQN targets noisy.

## Results

Fill after audit runs complete.

| Domain | games | samples | rank CE | top-1 acc | base MAE | avoid4 MAE | top1 MAE | zero-sum MAE | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| original | | | | | | | | | |
| style selfplay | | | | | | | | | |
