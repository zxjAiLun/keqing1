# KeqingRL Paired Candidate Evaluation

mode: `summary`
source_type: `checkpoint`
candidate_summary: `reports/keqingrl_mortal_action_q_candidate_cfg3_fresh_seedshift_20260429_source93_step20000/checkpoint_summary.csv`
checkpoint_path: `None`
config_path: `None`
source_config_ids: `[93]`
rerun_config_ids: `None`
eval_seed_registry_id: `base=202604250000:stride=1:count=32`
eval_seed_hash: `d35c12b75e341bf5`
eval_seed_count: `32`
diagnostic_episode_count: `32`
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

- opponent=rule_prior_greedy candidate_id=93 source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0.046875 delta_vs_untrained=0.046875 delta_vs_source=0.046875 rank_pt=0.046875 mean_rank=2.4375 fourth=0.234375 learner_deal_in=0.0703125 rule_agree=0.981608 top1_changed=0.0183918 neural_delta_mean=0.0835015 neural_delta_max=0.756963 changed_rank_mean=2.27907 rank_ge5=0 changed_margin_p50=3 checkpoint=reports/keqingrl_mortal_action_q_candidate_cfg3_fresh_seedshift_20260429_source93_step20000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=93 source_type=checkpoint source_id=93 support_policy=support-only-topk delta_vs_zero=0.046875 delta_vs_untrained=0.046875 delta_vs_source=0.046875 rank_pt=0.046875 mean_rank=2.4375 fourth=0.234375 learner_deal_in=0.0703125 rule_agree=0.981608 top1_changed=0.0183918 neural_delta_mean=0.0835015 neural_delta_max=0.756963 changed_rank_mean=2.27907 rank_ge5=0 changed_margin_p50=3 checkpoint=reports/keqingrl_mortal_action_q_candidate_cfg3_fresh_seedshift_20260429_source93_step20000/checkpoint_config_000/policy_final.pt

## Decision

At least one checkpoint changes top-1 or has non-trivial delta; continue paired strength/stability evaluation before changing learning signal.
