# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
source_config_ids: `93`
episodes: `16`
iterations: `3`
rule_score_scales: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
temperatures: `1.25`
lrs: `0.003`
update_epochs_values: `4`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.005,0.01`
delta_l2_coef_values: `0.0,0.001`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
eval_seed_registry_id: `base=202604300000:stride=1:count=16`
eval_seed_hash: `b9e4928c53bdcc35`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.005 delta_l2=0 delta_clip=0/0 pass=False non_top1=171 non_top1_pos=76 top1_changed=0 effective_margin=0.720149 scaled_prior_margin=0.750025 t_kl=0.000888637 t_clip=0 u_kl=0.00255756 u_clip=0.0152284 delta_max=0.150024 eval_fourth=0.375 deal_in=0.125
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.005 delta_l2=0.001 delta_clip=0/0 pass=True non_top1=193 non_top1_pos=99 top1_changed=0.151111 effective_margin=0.388152 scaled_prior_margin=0.750047 t_kl=0.00441275 t_clip=0.0577778 u_kl=0.00981794 u_clip=0.16 delta_max=0.574112 eval_fourth=0.375 deal_in=0.125
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.01 delta_l2=0 delta_clip=0/0 pass=True non_top1=188 non_top1_pos=167 top1_changed=0.204651 effective_margin=0.347985 scaled_prior_margin=0.750045 t_kl=0.0223218 t_clip=0.246512 u_kl=0.0397087 u_clip=0.344186 delta_max=2.70342 eval_fourth=0.4375 deal_in=0.1875
- cfg=3 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.01 delta_l2=0.001 delta_clip=0/0 pass=False non_top1=188 non_top1_pos=43 top1_changed=0.337963 effective_margin=0.289144 scaled_prior_margin=0.750022 t_kl=0.0146384 t_clip=0.217593 u_kl=0.0352105 u_clip=0.430556 delta_max=1.16868 eval_fourth=0.4375 deal_in=0.125

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
