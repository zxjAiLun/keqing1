# Reward Semantics A/B

## Purpose

The next experiment isolates reward semantics before changing CQL, learning rate, auxiliary loss, or network architecture.

The two primary groups are:

| Group | Reward | Initialization |
| --- | --- | --- |
| F | `final_rank_mc` | 70k weights, fresh Adam, fresh data stream |
| G | `mortal_grp_delta_pt` | 70k weights, fresh Adam, fresh data stream |

Both groups must use the same 6,000-hanchan file index, `num_epochs=2`, matched data/model seeds, `gamma=1.0`, CQL weight `5.0`, next-rank weight `0.2`, and constant learning rate `1e-4`. The first comparison stops at step 72,000. Three matched seed pairs are required before extending any run to 74,000.

## Implementation Status

- `scripts/mortal/mainline_dataloader.py` now supports `mortal_grp_delta_pt` by reusing Mortal's `GRP` and `RewardCalculator`.
- `scripts/mortal/test_reward_adapter.py` verifies upstream/project parity, telescoping, and data identity.
- `scripts/mortal/preflight_reward_distribution.py` reports reward mean, standard deviation, quantiles, nonzero rate, absolute delta-Pt quantiles, and per-hanchan absolute movement before training.
- `scripts/mortal/prepare_reward_ab.py` prepares the six matched-seed configs and a shared file index.
- `scripts/mortal/prepare_grp_v1.py` creates the independent GRP train/validation/holdout split, and `scripts/mortal/run_grp_training.py` trains the project-owned GRP checkpoint.
- `scripts/mortal/evaluate_grp_checkpoint.py` evaluates validation and holdout without changing checkpoint selection.
- `scripts/run_mortal_dqn_offline.py` supports `--initialize-optimizer-from`, which preserves only Adam moments while keeping scheduler and data stream fresh.
- Checkpoints now record reward, GRP hash, file-index hash, dataset manifest hash, initialization mode, project commit, Mortal revision, and libriichi revision.

## GRP Checkpoint

The workspace did not contain an upstream GRP checkpoint, so the project-owned `keqing_grp_v1` was trained from an independent 2,000-hanchan corpus. Its validation/holdout results and reward preflight are recorded in [`grp_v1_2026_07.md`](grp_v1_2026_07.md). All G runs use the same frozen checkpoint SHA256 recorded in `manifest.json`. A first attempt with `num_epochs=1` ended normally at step 71,832 and is retained separately as a configuration audit; the formal retry is `reward_ab_2026_07_epoch2`.

## Commands

Adapter correctness test:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"
uv run --no-sync python scripts/mortal/test_reward_adapter.py `
  --output artifacts/experiments/model_pool_2026_07/reward_adapter_test.json
```

Prepare or refresh the matched A/B matrix with the frozen project GRP checkpoint:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"
uv run --no-sync python scripts/mortal/prepare_reward_ab.py `
  --grp-checkpoint artifacts/experiments/model_pool_2026_07/keqing_grp_v1/keqing_grp_v1_best.pth
```

The command only writes configs and manifests. It does not start training.

## 72000 Checkpoint Audit

The formal retry completed all six checkpoints in `reward_ab_2026_07_epoch2`:

- F: `final_rank_mc`, seeds `20260718/19/20`.
- G: `mortal_grp_delta_pt`, seeds `20260718/19/20`.
- Every run reached step `72000`, consumed the same 6,000-file index for two epochs, and used the same 70k parent.
- Every G run uses the frozen `keqing_grp_v1` SHA256 recorded in the audit.
- The contract audit remains a local-only artifact at `artifacts/experiments/model_pool_2026_07/reward_ab_2026_07_epoch2/reward_ab_audit.json`; raw audit artifacts are intentionally not uploaded.

## 1000-Hanchan Matched Evaluation

The three matched-seed native random-seat evaluations were extended from 250 to a fixed 1,000 hanchans each, keeping the lineup `70k + ext_mortal + F + G`. The existing 250 logs were resumed, not regenerated:

| Pair | F avg rank | G avg rank | F avg Pt | G avg Pt | G-F Pt |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20260718` | 2.551 | 2.460 | -2.745 | +2.880 | +5.625 |
| `20260719` | 2.535 | 2.533 | -1.665 | -1.845 | -0.180 |
| `20260720` | 2.546 | 2.547 | -3.600 | -3.465 | +0.135 |

The paired statistic is calculated per hanchan as `delta_pt = Pt(G) - Pt(F)`. Over 3,000 paired hanchans:

- Mean paired delta: `+1.86 Pt/局`.
- Hanchan-cluster bootstrap 95% CI: `[-3.17, +6.86] Pt/局`.
- G finished ahead in `50.67%` of hanchans.
- Training-seed means: `[+5.625, -0.180, +0.135]`; positive seeds `2/3`.
- One-sided three-seed exact sign-test p-value: `0.500` for `2/3` positive seeds.

Pooled behavior remains close: G agari `20.49%` vs F `20.40%`, houjuu `12.22%` vs `12.18%`, fuuro `25.53%` vs `24.93%`, and riichi `18.96%` vs `19.70%`. The earlier lower-houjuu signal did not remain stable after extension.

Conclusion: `mortal_grp_delta_pt` remains a valid implemented reward alternative, but it does **not** pass the current evidence threshold for promotion to the default research reward. The arena-level CI includes zero and the training-seed result is mixed. Do not start Adam-preserved, LR, CQL, or architecture variants yet. The next controlled step is to add 2-3 new matched F/G training seeds, or explicitly stop the reward hypothesis and return to data/optimizer diagnostics.

The complete paired report is [`reward_ab_eval_1000h_summary.md`](../../reports/mortal/reward_ab_2026_07_epoch2/reward_ab_eval_1000h_summary.md), with machine-readable output in the adjacent JSON file. The raw 250/1000-hanchan evaluation artifacts remain local-only under `artifacts/` for auditability.
