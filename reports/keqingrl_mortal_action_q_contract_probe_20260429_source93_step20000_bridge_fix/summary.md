# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_step39cd_penalty_full_seed_202604340000_source93/k5_coef001_diagnostic_no_quality_gate/summary.csv`
source_config_ids: `93`
episodes: `32`
iterations: `1`
rule_score_scales: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
temperatures: `1.25`
lrs: `0.0001,0.0003`
update_epochs_values: `1`
clip_eps_values: `0.1,0.2`
rule_kl_coef_values: `0.001`
delta_l2_coef_values: `0.0`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
topk_ranking_aux: `{'modes': ('teacher-ce',), 'coef_values': (1.0,), 'topk': 3, 'teacher_sources': ('mortal-action-q',), 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max_values': (1000000000.0,), 'teacher_margin_min_values': (0.0,), 'teacher_prior_agree_min_values': (0.0,), 'expanded_configs': ({'mode': 'teacher-ce', 'coef': 1.0, 'topk': 3, 'teacher_source': 'mortal-action-q', 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max': 1000000000.0, 'teacher_margin_min': 0.0, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'mortal_action_teacher_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'},), 'implemented_modes': ('none', 'teacher-ce'), 'teacher_contract_version': 'keqingrl_topk_teacher_v1', 'teacher_source_note': 'mortal-discard-q and mortal-action-q are the only allowed Mortal teacher sources; rule-prior-topk is a diagnostic negative control; rule-components is a legacy alias; rule-component-v1 is an action-feature diagnostic reranker; Mortal teacher sources require Mortal q/mask tensors in policy_input.obs.extras'}`
action_scope: `{'self_turn_action_types': ('DISCARD', 'REACH_DISCARD', 'TSUMO', 'RYUKYOKU'), 'response_action_types': ('PASS', 'RON', 'PON', 'CHI'), 'forced_autopilot_action_types': ('TSUMO', 'RON', 'RYUKYOKU')}`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.0`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.0`
support_policy: `{'modes': ('support-only-topk',), 'expanded_configs': ({'support_policy_mode': 'support-only-topk', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'legacy_delta_support_args_used': False, 'scope': 'policy_forward_rollout_loss_fresh_eval'}`
delta_support_projection: `{'modes': ('topk',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'support_policy_mode': 'delta-topk-zero', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('topk',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': False, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 1.0}`
terminal_coverage_gate: `{'enabled': True, 'outcome_gate_enabled': False, 'role': 'qualification_gate_for_terminal_decision_opportunity_coverage', 'min_legal_terminal_rows': 1, 'min_score_changed_episode_rate': 0.0, 'min_legal_agari_rows': 1, 'min_prepared_legal_terminal_rows': 0, 'min_prepared_legal_agari_rows': 0, 'min_selected_agari_count': 0, 'min_selected_agari_episode_rate': 0.0, 'note': 'qualification is opportunity-based by default; score_changed/selected_agari thresholds are outcome diagnostics and only gate when outcome_gate_enabled=true'}`
fresh_validation: `{'episodes': 0, 'seed_base': 202604320000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy', 'per_iteration_early_stop_enabled': False, 'selection': 'best train+fresh gate pass; otherwise lowest fresh violation, then train/stability violation'}`
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.0001 epochs=1 clip=0.1 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-action-q/1/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter0 pass=False non_top1=279 non_top1_pos=245 actor_kept=1 dropped_pos=0 kept_non_top1_pos=245 top1_changed=0 train_quality=True changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0 weak_margin_pen=0.478834 rank_aux=0.912777 teacher_kl=0.587108 teacher_agree=0.699203 teacher_prior_agree=0.699203 teacher_rule_top1_rank=1.38845 teacher_margin=3.74702 teacher_entropy=0.325669 teacher_conf_keep=0.832504 reach_n=0 reach_policy=0 reach_teacher=0 non_discard_n=107 call_n=107 agari_n=0 score_changed_ep=30/32 legal_agari_rows=16 selected_agari=16 terminal_quality=True fresh_top1=0 fresh_quality=True qualified_eval=True effective_margin=0.843751 scaled_prior_margin=0.843746 t_kl=2.893e-10 t_clip=0 u_kl=0.00346153 u_clip=0.0597015 delta_max=0.000394713 recovery_extra=0 recovery_stop=disabled eval_fourth=0.125 deal_in=0
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.0001 epochs=1 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-action-q/1/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter0 pass=False non_top1=256 non_top1_pos=193 actor_kept=0.996289 dropped_pos=1 kept_non_top1_pos=192 top1_changed=0 train_quality=True changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0.00194145 weak_margin_pen=0.480435 rank_aux=0.907745 teacher_kl=0.600462 teacher_agree=0.692478 teacher_prior_agree=0.692478 teacher_rule_top1_rank=1.38496 teacher_margin=3.86057 teacher_entropy=0.307282 teacher_conf_keep=0.83859 reach_n=2 reach_policy=1 reach_teacher=1 non_discard_n=97 call_n=95 agari_n=0 score_changed_ep=32/32 legal_agari_rows=21 selected_agari=21 terminal_quality=True fresh_top1=0 fresh_quality=True qualified_eval=True effective_margin=0.84002 scaled_prior_margin=0.839101 t_kl=1.18972e-10 t_clip=0 u_kl=0.00359157 u_clip=0 delta_max=0.000665312 recovery_extra=0 recovery_stop=disabled eval_fourth=0.125 deal_in=0
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.0003 epochs=1 clip=0.1 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-action-q/1/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter0 pass=False non_top1=276 non_top1_pos=234 actor_kept=0.994872 dropped_pos=2 kept_non_top1_pos=232 top1_changed=0 train_quality=True changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0.00670784 weak_margin_pen=0.480415 rank_aux=0.906869 teacher_kl=0.587559 teacher_agree=0.705285 teacher_prior_agree=0.705285 teacher_rule_top1_rank=1.36789 teacher_margin=3.64795 teacher_entropy=0.31931 teacher_conf_keep=0.841026 reach_n=4 reach_policy=1 reach_teacher=0.25 non_discard_n=108 call_n=104 agari_n=0 score_changed_ep=32/32 legal_agari_rows=16 selected_agari=16 terminal_quality=True fresh_top1=0 fresh_quality=True qualified_eval=True effective_margin=0.842399 scaled_prior_margin=0.840671 t_kl=2.75483e-09 t_clip=0 u_kl=0.0034149 u_clip=0.0598291 delta_max=0.000826867 recovery_extra=0 recovery_stop=disabled eval_fourth=0.125 deal_in=0
- cfg=3 source=93 scale=0.25 temp=1.25 lr=0.0003 epochs=1 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-action-q/1/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter0 pass=False non_top1=252 non_top1_pos=226 actor_kept=0.998347 dropped_pos=0 kept_non_top1_pos=226 top1_changed=0 train_quality=True changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0.00259459 weak_margin_pen=0.470482 rank_aux=0.932541 teacher_kl=0.589979 teacher_agree=0.651822 teacher_prior_agree=0.651822 teacher_rule_top1_rank=1.4332 teacher_margin=3.57042 teacher_entropy=0.342562 teacher_conf_keep=0.816529 reach_n=2 reach_policy=1 reach_teacher=1 non_discard_n=110 call_n=108 agari_n=0 score_changed_ep=29/32 legal_agari_rows=17 selected_agari=17 terminal_quality=True fresh_top1=0 fresh_quality=True qualified_eval=True effective_margin=0.830215 scaled_prior_margin=0.8294 t_kl=2.40383e-09 t_clip=0 u_kl=0.00343551 u_clip=0.00495868 delta_max=0.000851691 recovery_extra=0 recovery_stop=disabled eval_fourth=0.125 deal_in=0

## Contract Scoreboard

- diagnostic_only_outcome_note: `outcome counters do not gate unless --terminal-coverage-outcome-gate is enabled`
- cfg=0 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=603/603 opportunity_gate=True outcome_gate=False score_changed=30 selected_agari=16 qualified=True reason=qualified_for_eval notes=qualified for paired eval
- cfg=1 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=539/539 opportunity_gate=True outcome_gate=False score_changed=32 selected_agari=21 qualified=True reason=qualified_for_eval notes=qualified for paired eval
- cfg=2 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=585/585 opportunity_gate=True outcome_gate=False score_changed=32 selected_agari=16 qualified=True reason=qualified_for_eval notes=qualified for paired eval
- cfg=3 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=605/605 opportunity_gate=True outcome_gate=False score_changed=29 selected_agari=17 qualified=True reason=qualified_for_eval notes=qualified for paired eval

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
