# KeqingRL Step 3.13B Teacher Confidence Gate Multi-Seed

scope: `diagnostic teacher confidence gate; not strength evidence`
source_config_id: `93`
support_policy_mode: `support-only-topk`
topK: `3`
teacher_source: `rule-component-v1`
teacher_ce_coef: `0.03`
rule_score_scale: `0.25`
seeds: `202604340000,202604350000,202604360000,202604370000`
run_count: `560 half-hands; 608.83s`

## Aggregate

- entropy<=0.8, margin>=0.2: qualified=2/4 train=2/4 fresh=2/4 top1_mean=0.191608 top1_max=0.265116 fresh_mean=0.165518 fresh_max=0.252137 changed_rank_mean=2.44006 rank_ge5_max=0 teacher_keep_mean=0.00403994 teacher_keep_min=0 teacher_prior_agree_mean=0.676845 eval_fourth_mean=0.40625 eval_deal_in_mean=0.15625
- entropy<=1, margin>=0.2: qualified=1/4 train=2/4 fresh=2/4 top1_mean=0.181758 top1_max=0.278302 fresh_mean=0.122799 fresh_max=0.280543 changed_rank_mean=2.54301 rank_ge5_max=0 teacher_keep_mean=0.285535 teacher_keep_min=0.236051 teacher_prior_agree_mean=0.681694 eval_fourth_mean=0.25 eval_deal_in_mean=0.0625

## Conclusion

- `support-only-topK=3` still prevents topK-outside flips: `rank_ge5_max=0` for both configs.
- Confidence gating does not produce a stable multi-seed recovery: best tested config is only `2/4` qualified.
- `entropy<=0.8, margin>=0.2` is too sparse: teacher keep rate is near zero, so it behaves like weak/no teacher control.
- `entropy<=1.0, margin>=0.2` keeps a meaningful teacher subset but still over-moves on most seeds.
- Do not proceed to paired strength-proxy eval from this gate; next useful step is per-iteration fresh early-stop or stronger teacher source audit.

## Artifacts

- `step313b_multiseed_summary.csv`
- `step313b_multiseed_summary.json`
- `seed_*/summary.csv`
- `seed_*/batch_steps.csv` with per-state teacher audit fields
