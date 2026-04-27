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
update_epochs_values: `8`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.001`
delta_l2_coef_values: `0.0`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.0`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.0`
delta_support_projection: `{'modes': ('all', 'topk'), 'topk_values': (3, 5), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'}, {'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'}, {'mode': 'topk', 'topk': 5, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'}), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('all',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 1.0}`
fresh_validation: `{'episodes': 16, 'seed_base': 202604610000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy'}`
eval_seed_registry_id: `base=202604600000:stride=1:count=16`
eval_seed_hash: `77b5e7c562366c07`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 delta_support=all/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=190 non_top1_pos=154 actor_kept=1 dropped_pos=0 kept_non_top1_pos=154 top1_changed=0.269058 train_quality=False changed_rank=5.51667 rank_ge5=0.45 margin_p50=3.003 low_rank_pen=3.24886 weak_margin_pen=0.847344 fresh_top1=0.2625 fresh_quality=False qualified_eval=False effective_margin=0.330752 scaled_prior_margin=0.750046 t_kl=0.0121543 t_clip=0.188341 u_kl=0.0204527 u_clip=0.300448 delta_max=3.91406 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=220 non_top1_pos=175 actor_kept=1 dropped_pos=0 kept_non_top1_pos=175 top1_changed=0.171875 train_quality=False changed_rank=2.75 rank_ge5=0 margin_p50=3.001 low_rank_pen=2.70207 weak_margin_pen=0.784051 fresh_top1=0.10917 fresh_quality=False qualified_eval=False effective_margin=0.59515 scaled_prior_margin=0.750031 t_kl=0.0167841 t_clip=0.285156 u_kl=0.0420117 u_clip=0.507812 delta_max=1.94678 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 delta_support=topk/5/0.75/zero actor_support=all/3/0.75 pass=False non_top1=218 non_top1_pos=131 actor_kept=1 dropped_pos=0 kept_non_top1_pos=131 top1_changed=0.164659 train_quality=False changed_rank=4.68293 rank_ge5=0.609756 margin_p50=3.004 low_rank_pen=3.25072 weak_margin_pen=0.844929 fresh_top1=0.195745 fresh_quality=False qualified_eval=False effective_margin=0.403149 scaled_prior_margin=0.750054 t_kl=0.0231079 t_clip=0.313253 u_kl=0.0441429 u_clip=0.389558 delta_max=2.2227 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
