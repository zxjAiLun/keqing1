# KeqingRL Learning-Signal Diagnostics

source_config_id: `57`
checkpoint_path: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_001_source_57/policy_final.pt`
checkpoint_sha256: `2d5c4cc89f71d05a950c07e153da02ed7943eea80161e9f54d06530ec3c96994`
config_key: `{"entropy_coef": 0.01, "lr": 3e-05, "opponent_mode": "rule_prior_greedy", "rule_kl_coef": 0.05}`
diagnostic_seed_registry_id: `learning_signal_20260425_step35_v1`
diagnostic_seed_hash: `0f42628c88353290`
primary_blocker: `BATCH_OVERFITS_BUT_ROLLOUT_FAILS`

## Batch Signal

- batch_size: `771`
- advantage_std: `0.72662`
- selected_non_top1_positive_advantage_count: `44`
- mean_delta_needed_to_flip_top1: `3.00067`
- learner_deal_in_rate: `0.0625`

## Stability

- rulebase_fallback_count: `0`
- illegal_action_rate_fail_closed: `0`
- fallback_rate_fail_closed: `0`
- forced_terminal_missed_fail_closed: `0`

## Fixed-Batch Overfit

- best_top1_action_changed_rate: `0.0324254`
- best_neural_delta_abs_max: `4.47776`
- pass_actor_can_move: `True`
- gradient_dead: `False`
- prior_margin_too_large: `False`
