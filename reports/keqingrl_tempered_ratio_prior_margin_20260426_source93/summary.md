# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
source_config_ids: `93`
episodes: `16`
iterations: `3`
rule_score_scales: `1.0,0.5,0.25,0.1`
temperatures: `1.25`
lrs: `0.003`
update_epochs_values: `8`
clip_eps_values: `0.2`
rule_kl_coef: `0.001`
entropy_coef: `0.005`
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`

## Results

- cfg=3 source=93 scale=0.1 temp=1.25 lr=0.003 epochs=8 clip=0.2 non_top1=196 non_top1_pos=80 top1_changed=0.687783 effective_margin=0.070362 scaled_prior_margin=0.300013 t_kl=0.0157779 t_clip=0.221719 u_kl=0.0307776 u_clip=0.357466 delta_max=1.75305 eval_fourth=0.4375 deal_in=0.0625
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 non_top1=182 non_top1_pos=155 top1_changed=0.105023 effective_margin=0.518475 scaled_prior_margin=0.75004 t_kl=0.0113391 t_clip=0.200913 u_kl=0.015967 u_clip=0.26484 delta_max=1.71831 eval_fourth=0.4375 deal_in=0.125
- cfg=1 source=93 scale=0.5 temp=1.25 lr=0.003 epochs=8 clip=0.2 non_top1=183 non_top1_pos=129 top1_changed=0.0042735 effective_margin=1.11852 scaled_prior_margin=1.50009 t_kl=0.00857129 t_clip=0.0940171 u_kl=0.0214501 u_clip=0.34188 delta_max=1.93039 eval_fourth=0.375 deal_in=0.125
- cfg=0 source=93 scale=1 temp=1.25 lr=0.003 epochs=8 clip=0.2 non_top1=106 non_top1_pos=36 top1_changed=0 effective_margin=2.67924 scaled_prior_margin=3.00016 t_kl=0.0148902 t_clip=0.19802 u_kl=0.0862569 u_clip=0.816832 delta_max=1.60806 eval_fourth=0.375 deal_in=0.125

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
