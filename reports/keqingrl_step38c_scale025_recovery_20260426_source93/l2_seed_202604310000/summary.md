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
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.005 delta_l2=0 delta_clip=0/0 pass=True non_top1=172 non_top1_pos=112 top1_changed=0.172727 effective_margin=0.365321 scaled_prior_margin=0.75005 t_kl=0.00545407 t_clip=0.0636364 u_kl=0.0131797 u_clip=0.177273 delta_max=0.968941 eval_fourth=0.375 deal_in=0.0625
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.005 delta_l2=0.001 delta_clip=0/0 pass=True non_top1=187 non_top1_pos=107 top1_changed=0.159817 effective_margin=0.394336 scaled_prior_margin=0.750042 t_kl=0.00883514 t_clip=0.114155 u_kl=0.0176827 u_clip=0.319635 delta_max=0.940407 eval_fourth=0.375 deal_in=0.125
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.01 delta_l2=0 delta_clip=0/0 pass=False non_top1=197 non_top1_pos=162 top1_changed=0.254386 effective_margin=0.426968 scaled_prior_margin=0.750054 t_kl=0.0237412 t_clip=0.302632 u_kl=0.042525 u_clip=0.486842 delta_max=1.05158 eval_fourth=0.4375 deal_in=0.125
- cfg=3 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.01 delta_l2=0.001 delta_clip=0/0 pass=True non_top1=190 non_top1_pos=170 top1_changed=0.0853081 effective_margin=0.439284 scaled_prior_margin=0.75005 t_kl=0.00512128 t_clip=0.0663507 u_kl=0.00679877 u_clip=0.0805687 delta_max=0.681741 eval_fourth=0.375 deal_in=0.125

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
