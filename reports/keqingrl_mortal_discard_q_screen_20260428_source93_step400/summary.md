# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_step39cd_penalty_full_seed_202604340000_source93/k5_coef001_diagnostic_no_quality_gate/summary.csv`
source_config_ids: `93`
episodes: `4`
iterations: `2`
rule_score_scales: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
temperatures: `1.25`
lrs: `0.0003`
update_epochs_values: `1`
clip_eps_values: `0.1`
rule_kl_coef_values: `0.001`
delta_l2_coef_values: `0.0`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
topk_ranking_aux: `{'modes': ('teacher-ce',), 'coef_values': (0.2, 0.5, 1.0), 'topk': 3, 'teacher_sources': ('mortal-discard-q',), 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max_values': (1000000000.0,), 'teacher_margin_min_values': (0.0,), 'teacher_prior_agree_min_values': (0.0,), 'expanded_configs': ({'mode': 'teacher-ce', 'coef': 0.2, 'topk': 3, 'teacher_source': 'mortal-discard-q', 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max': 1000000000.0, 'teacher_margin_min': 0.0, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'mortal_discard_teacher_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'}, {'mode': 'teacher-ce', 'coef': 0.5, 'topk': 3, 'teacher_source': 'mortal-discard-q', 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max': 1000000000.0, 'teacher_margin_min': 0.0, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'mortal_discard_teacher_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'}, {'mode': 'teacher-ce', 'coef': 1.0, 'topk': 3, 'teacher_source': 'mortal-discard-q', 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max': 1000000000.0, 'teacher_margin_min': 0.0, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'mortal_discard_teacher_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'}), 'implemented_modes': ('none', 'teacher-ce'), 'teacher_contract_version': 'keqingrl_topk_teacher_v1', 'teacher_source_note': 'mortal-discard-q is the only allowed strength teacher source; rule-prior-topk is a diagnostic negative control; rule-components is a legacy alias; rule-component-v1 is an action-feature diagnostic reranker; mortal-discard-q requires Mortal q/mask tensors in policy_input.obs.extras'}`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.0`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.0`
support_policy: `{'modes': ('support-only-topk',), 'expanded_configs': ({'support_policy_mode': 'support-only-topk', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'legacy_delta_support_args_used': False, 'scope': 'policy_forward_rollout_loss_fresh_eval'}`
delta_support_projection: `{'modes': ('all',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'support_policy_mode': 'unrestricted', 'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('topk',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.2, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 1.0}`
fresh_validation: `{'episodes': 2, 'seed_base': 202604320000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.0, 'max_top1_changed': 0.2, 'policy_mode': 'greedy', 'per_iteration_early_stop_enabled': False, 'selection': 'best train+fresh gate pass; otherwise lowest fresh violation, then train/stability violation'}`
eval_seed_registry_id: `base=202604310000:stride=1:count=4`
eval_seed_hash: `b22f97dd19fd1449`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.0003 epochs=1 clip=0.1 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-discard-q/0.2/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter1 pass=False non_top1=48 non_top1_pos=48 actor_kept=1 dropped_pos=0 kept_non_top1_pos=48 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0 weak_margin_pen=0.523237 rank_aux=1.01222 teacher_kl=0.173328 teacher_agree=0.708333 teacher_prior_agree=0.708333 teacher_rule_top1_rank=1.36111 teacher_margin=1.02163 teacher_entropy=0.838894 teacher_conf_keep=1 fresh_top1=0 fresh_quality=True qualified_eval=False effective_margin=0.749995 scaled_prior_margin=0.750028 t_kl=8.12888e-10 t_clip=0 u_kl=0.0027969 u_clip=0 delta_max=0.000703646 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.0003 epochs=1 clip=0.1 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-discard-q/0.5/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter1 pass=False non_top1=20 non_top1_pos=6 actor_kept=1 dropped_pos=0 kept_non_top1_pos=6 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0 weak_margin_pen=0.523236 rank_aux=0.948273 teacher_kl=0.235076 teacher_agree=0.809524 teacher_prior_agree=0.809524 teacher_rule_top1_rank=1.21429 teacher_margin=1.1746 teacher_entropy=0.713197 teacher_conf_keep=1 fresh_top1=0 fresh_quality=True qualified_eval=False effective_margin=0.750033 scaled_prior_margin=0.750048 t_kl=5.15671e-10 t_clip=0 u_kl=0.00281666 u_clip=0 delta_max=0.000741274 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.0003 epochs=1 clip=0.1 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-discard-q/1/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter1 pass=False non_top1=33 non_top1_pos=0 actor_kept=1 dropped_pos=0 kept_non_top1_pos=0 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0 weak_margin_pen=0.523219 rank_aux=0.950682 teacher_kl=0.223621 teacher_agree=0.746032 teacher_prior_agree=0.746032 teacher_rule_top1_rank=1.30159 teacher_margin=1.35124 teacher_entropy=0.727061 teacher_conf_keep=1 fresh_top1=0 fresh_quality=True qualified_eval=False effective_margin=0.750077 scaled_prior_margin=0.750079 t_kl=2.53395e-09 t_clip=0 u_kl=0.00281312 u_clip=0 delta_max=0.00102356 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
