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
topk_ranking_aux: `{'modes': ('none', 'teacher-ce'), 'coef_values': (0.01, 0.05, 0.1), 'topk': 3, 'teacher_source': 'rule-components', 'teacher_temperature': 1.0, 'expanded_configs': ({'mode': 'none', 'coef': 0.0, 'topk': 3, 'teacher_source': 'rule-components', 'teacher_temperature': 1.0}, {'mode': 'teacher-ce', 'coef': 0.01, 'topk': 3, 'teacher_source': 'rule-components', 'teacher_temperature': 1.0}, {'mode': 'teacher-ce', 'coef': 0.05, 'topk': 3, 'teacher_source': 'rule-components', 'teacher_temperature': 1.0}, {'mode': 'teacher-ce', 'coef': 0.1, 'topk': 3, 'teacher_source': 'rule-components', 'teacher_temperature': 1.0}), 'implemented_modes': ('none', 'teacher-ce'), 'teacher_source_note': 'rule-components currently maps to the existing rule-prior topK distribution'}`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.0`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.0`
support_policy: `{'modes': ('support-only-topk',), 'expanded_configs': ({'support_policy_mode': 'support-only-topk', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'legacy_delta_support_args_used': False, 'scope': 'policy_forward_rollout_loss_fresh_eval'}`
delta_support_projection: `{'modes': ('all',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'support_policy_mode': 'unrestricted', 'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('all',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 3.1}`
fresh_validation: `{'episodes': 16, 'seed_base': 202604820000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy'}`
eval_seed_registry_id: `base=202604810000:stride=1:count=16`
eval_seed_hash: `80ba1b5e636cc52a`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=none/0/k3 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=123 non_top1_pos=100 actor_kept=1 dropped_pos=0 kept_non_top1_pos=100 top1_changed=0.39823 train_quality=False changed_rank=2.45556 rank_ge5=0 margin_p50=3.001 low_rank_pen=0 weak_margin_pen=0.574181 rank_aux=0 teacher_kl=0 teacher_agree=0 fresh_top1=0.204255 fresh_quality=False qualified_eval=False effective_margin=0.401086 scaled_prior_margin=0.750049 t_kl=0.0194908 t_clip=0.292035 u_kl=0.0354221 u_clip=0.420354 delta_max=2.28136 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/0.01/k3 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=119 non_top1_pos=67 actor_kept=1 dropped_pos=0 kept_non_top1_pos=67 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0 weak_margin_pen=0.516148 rank_aux=0.727508 teacher_kl=0.361108 teacher_agree=1 fresh_top1=0 fresh_quality=False qualified_eval=False effective_margin=0.690482 scaled_prior_margin=0.750059 t_kl=0.00758303 t_clip=0.115556 u_kl=0.00871862 u_clip=0.124444 delta_max=0.824395 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/0.05/k3 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=111 non_top1_pos=83 actor_kept=1 dropped_pos=0 kept_non_top1_pos=83 top1_changed=0.133333 train_quality=True changed_rank=2.5 rank_ge5=0 margin_p50=3.001 low_rank_pen=0 weak_margin_pen=0.470408 rank_aux=0.691332 teacher_kl=0.324914 teacher_agree=0.866667 fresh_top1=0.140187 fresh_quality=False qualified_eval=False effective_margin=0.794967 scaled_prior_margin=0.750044 t_kl=0.00653421 t_clip=0.106667 u_kl=0.0218794 u_clip=0.284444 delta_max=1.4648 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=3 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/0.1/k3 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=110 non_top1_pos=94 actor_kept=1 dropped_pos=0 kept_non_top1_pos=94 top1_changed=0.251064 train_quality=False changed_rank=2.44068 rank_ge5=0 margin_p50=3 low_rank_pen=0 weak_margin_pen=0.516895 rank_aux=0.788912 teacher_kl=0.422504 teacher_agree=0.748936 fresh_top1=0.122172 fresh_quality=False qualified_eval=False effective_margin=0.611626 scaled_prior_margin=0.750055 t_kl=0.0163622 t_clip=0.27234 u_kl=0.0275931 u_clip=0.421277 delta_max=1.63808 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
