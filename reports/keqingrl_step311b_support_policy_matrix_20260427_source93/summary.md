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
support_policy: `{'modes': ('unrestricted', 'delta-topk-zero', 'support-only-topk'), 'expanded_configs': ({'support_policy_mode': 'unrestricted', 'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'}, {'support_policy_mode': 'delta-topk-zero', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'}, {'support_policy_mode': 'delta-topk-zero', 'mode': 'topk', 'topk': 5, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'}, {'support_policy_mode': 'support-only-topk', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'}, {'support_policy_mode': 'support-only-topk', 'mode': 'topk', 'topk': 5, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'}), 'legacy_delta_support_args_used': False, 'scope': 'policy_forward_rollout_loss_fresh_eval'}`
delta_support_projection: `{'modes': ('all',), 'topk_values': (3, 5), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'support_policy_mode': 'unrestricted', 'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('all',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 3.1}`
fresh_validation: `{'episodes': 16, 'seed_base': 202604700000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy'}`
eval_seed_registry_id: `base=202604690000:stride=1:count=16`
eval_seed_hash: `bbd4391d444c3989`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 support_policy=delta-topk-zero delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=230 non_top1_pos=175 actor_kept=1 dropped_pos=0 kept_non_top1_pos=175 top1_changed=0.133588 train_quality=False changed_rank=3.65714 rank_ge5=0 margin_p50=3.002 low_rank_pen=3.58467 weak_margin_pen=0.864246 fresh_top1=0.0917874 fresh_quality=False qualified_eval=False effective_margin=0.504564 scaled_prior_margin=0.750054 t_kl=0.00526377 t_clip=0.0687023 u_kl=0.00885016 u_clip=0.114504 delta_max=1.23649 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 support_policy=delta-topk-zero delta_support=topk/5/0.75/zero actor_support=all/3/0.75 pass=False non_top1=216 non_top1_pos=94 actor_kept=1 dropped_pos=0 kept_non_top1_pos=94 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=3.85617 weak_margin_pen=0.8788 fresh_top1=0 fresh_quality=False qualified_eval=False effective_margin=0.337824 scaled_prior_margin=0.75004 t_kl=0.00499442 t_clip=0.00406504 u_kl=0.0091541 u_clip=0.109756 delta_max=0.813174 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=3 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=103 non_top1_pos=55 actor_kept=1 dropped_pos=0 kept_non_top1_pos=55 top1_changed=0.0883721 train_quality=True changed_rank=2.47368 rank_ge5=0 margin_p50=3.001 low_rank_pen=0 weak_margin_pen=0.495105 fresh_top1=0.154867 fresh_quality=False qualified_eval=False effective_margin=0.671866 scaled_prior_margin=0.750058 t_kl=0.0140856 t_clip=0.218605 u_kl=0.0282349 u_clip=0.390698 delta_max=2.35389 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=4 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/5/0.75/zero actor_support=all/3/0.75 pass=False non_top1=156 non_top1_pos=92 actor_kept=1 dropped_pos=0 kept_non_top1_pos=92 top1_changed=0.183406 train_quality=False changed_rank=4.11905 rank_ge5=0.5 margin_p50=3.003 low_rank_pen=0.546053 weak_margin_pen=0.688002 fresh_top1=0.112195 fresh_quality=False qualified_eval=False effective_margin=0.422123 scaled_prior_margin=0.750039 t_kl=0.0126857 t_clip=0.213974 u_kl=0.0229492 u_clip=0.358079 delta_max=2.12161 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 low_rank=3/0 weak_margin=0.75/0 support_policy=unrestricted delta_support=all/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=215 non_top1_pos=190 actor_kept=1 dropped_pos=0 kept_non_top1_pos=190 top1_changed=0.387755 train_quality=False changed_rank=8.01053 rank_ge5=0.905263 margin_p50=3.007 low_rank_pen=3.50574 weak_margin_pen=0.859411 fresh_top1=0.350962 fresh_quality=False qualified_eval=False effective_margin=0.280932 scaled_prior_margin=0.750057 t_kl=0.00572144 t_clip=0.0653061 u_kl=0.0113856 u_clip=0.167347 delta_max=3.34327 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
