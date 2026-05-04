# KeqingRL Paired Candidate Evaluation

mode: `summary`
source_type: `checkpoint`
candidate_summary: `reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_summary.csv`
checkpoint_path: `None`
config_path: `None`
source_config_ids: `[93]`
rerun_config_ids: `None`
eval_seed_registry_id: `base=202604310000:stride=1:count=32`
eval_seed_hash: `5e8b45c2f77adc12`
eval_seed_count: `32`
diagnostic_episode_count: `4`
seat_rotation: `0,1,2,3`
action_scope: `self=DISCARD,REACH_DISCARD,TSUMO,RYUKYOKU; response=PASS,RON,PON,CHI; forced=TSUMO,RON,RYUKYOKU`
policy_mode: `greedy`
training_rollout_reuse: `false`
strength_proxy_note: `paired seat-rotation sanity/strength-proxy only; this is not proof of model strength.`
movement_quality_note: `neural_delta_abs_* is pure neural_delta; scaled_prior_delta_abs_* isolates scale contribution; final_minus_unscaled_prior_abs_* is compatibility/diagnostic only.`

## Required Fields

- checkpoint_path / checkpoint_sha256
- source_type / source_config_id / config_key
- eval_seed_registry_id / eval_seed_hash
- learner_deal_in_rate
- rulebase_fallback_count
- forced_terminal_preempt_count / autopilot_terminal_count
- illegal_action_rate_fail_closed / fallback_rate_fail_closed / forced_terminal_missed_fail_closed

## Candidate Results

- opponent=rule_prior_greedy candidate_id=source_93_rerun_0_e297fc66 source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.132812 rule_agree=1 top1_changed=0 neural_delta_mean=6.03602e-05 neural_delta_max=0.000403387 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_0_3f8b27da source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.132812 rule_agree=1 top1_changed=0 neural_delta_mean=8.33239e-05 neural_delta_max=0.000574464 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_config_001/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_0_dfe90648 source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.132812 rule_agree=1 top1_changed=0 neural_delta_mean=6.55912e-05 neural_delta_max=0.000711031 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_config_003/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_0_5a25ad3f source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.132812 rule_agree=1 top1_changed=0 neural_delta_mean=7.89941e-05 neural_delta_max=0.000815773 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_config_002/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_0_e297fc66 source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.132812 rule_agree=1 top1_changed=0 neural_delta_mean=6.03602e-05 neural_delta_max=0.000403387 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_0_3f8b27da source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.132812 rule_agree=1 top1_changed=0 neural_delta_mean=8.33239e-05 neural_delta_max=0.000574464 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_config_001/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_0_dfe90648 source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.132812 rule_agree=1 top1_changed=0 neural_delta_mean=6.55912e-05 neural_delta_max=0.000711031 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_config_003/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_0_5a25ad3f source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.132812 rule_agree=1 top1_changed=0 neural_delta_mean=7.89941e-05 neural_delta_max=0.000815773 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt/checkpoint_config_002/policy_final.pt

## Decision

Current checkpoints remain rule-prior equivalent on top-1 decisions; prioritize learning-signal research before more eval scaling.
