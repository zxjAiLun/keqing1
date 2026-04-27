# KeqingRL Paired Candidate Evaluation

mode: `summary`
source_type: `checkpoint`
candidate_summary: `reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/checkpoint_summary.csv`
checkpoint_path: `None`
config_path: `None`
source_config_ids: `None`
rerun_config_ids: `None`
eval_seed_registry_id: `base=202604410000:stride=1:count=32`
eval_seed_hash: `4dbacb65778dfba6`
eval_seed_count: `32`
diagnostic_episode_count: `8`
seat_rotation: `0,1,2,3`
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

- opponent=rule_prior_greedy candidate_id=source_93_seed_202604370000 source_type=checkpoint source_id=93 delta_vs_zero=0.0195312 delta_vs_untrained=0.0195312 delta_vs_source=0.0195312 rank_pt=0.0195312 mean_rank=2.49219 fourth=0.234375 learner_deal_in=0.109375 rule_agree=0.940397 top1_changed=0.0596026 neural_delta_mean=0.251304 neural_delta_max=0.95983 changed_rank_mean=5.88889 rank_ge5=0.555556 changed_margin_p50=3.003 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604370000/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_seed_202604360000 source_type=checkpoint source_id=93 delta_vs_zero=-0.0507812 delta_vs_untrained=-0.0507812 delta_vs_source=-0.0507812 rank_pt=-0.0507812 mean_rank=2.55469 fourth=0.273438 learner_deal_in=0.109375 rule_agree=0.886918 top1_changed=0.113082 neural_delta_mean=0.229307 neural_delta_max=0.963136 changed_rank_mean=5.68627 rank_ge5=0.666667 changed_margin_p50=3.003 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604360000/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_seed_202604340000 source_type=checkpoint source_id=93 delta_vs_zero=-0.101562 delta_vs_untrained=-0.101562 delta_vs_source=-0.101562 rank_pt=-0.101562 mean_rank=2.625 fourth=0.289062 learner_deal_in=0.132812 rule_agree=0.775281 top1_changed=0.224719 neural_delta_mean=0.287936 neural_delta_max=1.39159 changed_rank_mean=7.24 rank_ge5=0.72 changed_margin_p50=3.007 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604340000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_seed_202604370000 source_type=checkpoint source_id=93 delta_vs_zero=0.0195312 delta_vs_untrained=0.0195312 delta_vs_source=0.0195312 rank_pt=0.0195312 mean_rank=2.49219 fourth=0.234375 learner_deal_in=0.109375 rule_agree=0.940397 top1_changed=0.0596026 neural_delta_mean=0.251304 neural_delta_max=0.95983 changed_rank_mean=5.88889 rank_ge5=0.555556 changed_margin_p50=3.003 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604370000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_seed_202604360000 source_type=checkpoint source_id=93 delta_vs_zero=-0.0507812 delta_vs_untrained=-0.0507812 delta_vs_source=-0.0507812 rank_pt=-0.0507812 mean_rank=2.55469 fourth=0.273438 learner_deal_in=0.109375 rule_agree=0.886918 top1_changed=0.113082 neural_delta_mean=0.229307 neural_delta_max=0.963136 changed_rank_mean=5.68627 rank_ge5=0.666667 changed_margin_p50=3.003 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604360000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_seed_202604340000 source_type=checkpoint source_id=93 delta_vs_zero=-0.101562 delta_vs_untrained=-0.101562 delta_vs_source=-0.101562 rank_pt=-0.101562 mean_rank=2.625 fourth=0.289062 learner_deal_in=0.132812 rule_agree=0.775281 top1_changed=0.224719 neural_delta_mean=0.287936 neural_delta_max=1.39159 changed_rank_mean=7.24 rank_ge5=0.72 changed_margin_p50=3.007 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604340000/checkpoint_config_000/policy_final.pt

## Decision

At least one checkpoint changes top-1 or has non-trivial delta; continue paired strength/stability evaluation before changing learning signal.
