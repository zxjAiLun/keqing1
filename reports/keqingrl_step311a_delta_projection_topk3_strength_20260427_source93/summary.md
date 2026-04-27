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
lrs: `0.001,0.003`
update_epochs_values: `4,8`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.001`
delta_l2_coef_values: `0.0`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.0`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.0`
delta_support_projection: `{'modes': ('topk',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('all',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 3.1}`
fresh_validation: `{'episodes': 16, 'seed_base': 202604640000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy'}`
eval_seed_registry_id: `base=202604630000:stride=1:count=16`
eval_seed_hash: `db75cb7bf9db60db`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=4 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=206 non_top1_pos=139 actor_kept=1 dropped_pos=0 kept_non_top1_pos=139 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=3.34319 weak_margin_pen=0.852586 fresh_top1=0 fresh_quality=False qualified_eval=False effective_margin=0.748001 scaled_prior_margin=0.75003 t_kl=5.9284e-05 t_clip=0 u_kl=0.00177316 u_clip=0 delta_max=0.0113039 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=219 non_top1_pos=161 actor_kept=1 dropped_pos=0 kept_non_top1_pos=161 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=2.98079 weak_margin_pen=0.810743 fresh_top1=0 fresh_quality=False qualified_eval=False effective_margin=0.748396 scaled_prior_margin=0.750048 t_kl=0.00897905 t_clip=0.168582 u_kl=0.0239894 u_clip=0.321839 delta_max=0.695586 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=185 non_top1_pos=133 actor_kept=1 dropped_pos=0 kept_non_top1_pos=133 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=3.44576 weak_margin_pen=0.856247 fresh_top1=0 fresh_quality=False qualified_eval=False effective_margin=0.732707 scaled_prior_margin=0.750021 t_kl=0.000153918 t_clip=0 u_kl=0.00194987 u_clip=0 delta_max=0.0349286 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=3 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=222 non_top1_pos=139 actor_kept=1 dropped_pos=0 kept_non_top1_pos=139 top1_changed=0.144531 train_quality=False changed_rank=4 rank_ge5=0 margin_p50=3.002 low_rank_pen=2.95293 weak_margin_pen=0.816776 fresh_top1=0.133047 fresh_quality=False qualified_eval=False effective_margin=0.544885 scaled_prior_margin=0.750025 t_kl=0.0535324 t_clip=0.484375 u_kl=0.0944804 u_clip=0.617188 delta_max=3.03565 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
