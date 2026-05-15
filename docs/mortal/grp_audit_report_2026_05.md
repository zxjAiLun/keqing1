# GRP Audit Report

Status: first four-domain audit complete, old GRP not promoted for retraining.

## Goal

Audit the existing GRP before any retraining. Changing the PT table only changes rank-probability scalarization, so new GRP training is gated on calibration evidence, not assumed.

## Inputs

- Old GRP: `artifacts/mortal_training/grp.pth`
- Original logs: `artifacts/mortal_mjai_gz/train/**/*.json.gz`
- 70k default selfplay: `artifacts/experiments/grp_audit/selfplay_70k_base_500h_arena/logs/**/*.json.gz`
- 80k default selfplay: `artifacts/experiments/grp_audit/selfplay_80k_base_500h_arena/logs/**/*.json.gz`
- Style selfplay stress logs: `artifacts/eval/style_selfplay_alpha_500h_20260512_153647/**/*.json.gz`

The 70k and 80k selfplay domains were generated with arena all-same selfplay, 500 hanchan each. Tenhou `[90,45,0,-135]` is used as the reporting reference; Mortal default `[6,4,2,0]` is still the active GRP reward-scalarization profile for this audit.

## Commands

Original deterministic sample:

```bash
uv run python scripts/mortal/audit_grp_on_logs.py \
  --grp-checkpoint artifacts/mortal_training/grp.pth \
  --logs 'artifacts/mortal_mjai_gz/train/**/*.json.gz' \
  --sample-log-files 5000 \
  --sample-seed 20260514 \
  --device cuda \
  --output artifacts/experiments/grp_audit/audit_original_sample5000_metrics.json
```

70k default selfplay:

```bash
uv run python scripts/mortal/audit_grp_on_logs.py \
  --grp-checkpoint artifacts/mortal_training/grp.pth \
  --logs 'artifacts/experiments/grp_audit/selfplay_70k_base_500h_arena/logs/**/*.json.gz' \
  --device cuda \
  --output artifacts/experiments/grp_audit/audit_70k_base_500h_arena_metrics.json
```

80k default selfplay:

```bash
uv run python scripts/mortal/audit_grp_on_logs.py \
  --grp-checkpoint artifacts/mortal_training/grp.pth \
  --logs 'artifacts/experiments/grp_audit/selfplay_80k_base_500h_arena/logs/**/*.json.gz' \
  --device cuda \
  --output artifacts/experiments/grp_audit/audit_80k_base_500h_arena_metrics.json
```

Style selfplay stress:

```bash
uv run python scripts/mortal/audit_grp_on_logs.py \
  --grp-checkpoint artifacts/mortal_training/grp.pth \
  --logs 'artifacts/eval/style_selfplay_alpha_500h_20260512_153647/**/*.json.gz' \
  --device cuda \
  --output artifacts/experiments/grp_audit/audit_style_alpha500_500h_metrics.json
```

## Results

### Core Metrics

| Domain | games | samples | rank CE | top-1 acc | ECE | MCE | Mortal MAE | Mortal RMSE | Reward-delta var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original sample5000 | 5000 | 52906 | 2.6148 | 0.2026 | 0.0546 | 0.3389 | 1.5499 | 1.8628 | 0.2855 |
| 70k default selfplay | 500 | 5256 | 2.6099 | 0.1971 | 0.0542 | 0.3073 | 1.5625 | 1.8703 | 0.2934 |
| 80k default selfplay | 500 | 5183 | 2.5705 | 0.2562 | 0.1114 | 0.6803 | 1.5158 | 1.8320 | 0.2760 |
| style selfplay stress | 500 | 5345 | 2.6164 | 0.2135 | 0.0708 | 0.5422 | 1.5474 | 1.8623 | 0.2731 |

### Tenhou Reference Scale

These values are the same expected-PT errors measured under the Tenhou reference table. They are useful for report readability, but not the active Mortal training reward scale.

| Domain | Tenhou MAE | Tenhou RMSE | Tenhou reward-delta var |
| --- | ---: | ---: | ---: |
| original sample5000 | 57.0011 | 71.4009 | 411.0276 |
| 70k default selfplay | 57.0818 | 71.3611 | 420.4185 |
| 80k default selfplay | 55.5525 | 70.1953 | 399.2858 |
| style selfplay stress | 56.6271 | 70.9104 | 387.5446 |

## Interpretation

The old GRP does not show a clear expected-PT or reward-delta degradation on 80k selfplay relative to 70k selfplay. Under the active `mortal_default` profile, 80k has lower MAE, lower RMSE, and lower reward-delta variance than 70k in this audit sample.

The notable 80k difference is calibration: ECE rises from `0.0542` at 70k to `0.1114` at 80k, and MCE rises from `0.3073` to `0.6803`. That is worth monitoring, but by itself it does not explain the 80k gate failure as a GRP reward-scale or reward-variance problem.

Style selfplay is not worse than original or 70k on expected-PT error, but its MCE is higher than original/70k. Treat it as an out-of-domain stress signal rather than evidence that GRP retraining is currently mandatory.

## Decision

Do not retrain or adapt GRP yet.

Current evidence supports:

- Use 70k default as the standard reference / balanced anchor.
- Use 80k default as an aggressive / fuuro-heavy behavior anchor. It did not pass automatic promotion in the 1000h screening gate, but that gate is not a final strength verdict and does not remove its value as a style source.
- Do not attribute 80k non-promotion primarily to GRP expected-PT error or reward-delta variance.
- Continue tracking calibration if future selfplay or style runs move farther from the original distribution.
- Reopen mixed-domain GRP adaptation only if future audits show both calibration degradation and expected-PT or reward-delta degradation on selfplay domains.

Audit profiles are `mortal_default`, `avoid4_norm`, `top1_norm`, plus optional raw variants and Tenhou reference. `zero_sum_balanced` is not reported separately because it is a constant-shift equivalent of Mortal default.
