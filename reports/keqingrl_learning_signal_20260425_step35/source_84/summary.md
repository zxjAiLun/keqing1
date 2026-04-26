# KeqingRL Learning-Signal Diagnostics

source_config_id: `84`
checkpoint_path: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_002_source_84/policy_final.pt`
checkpoint_sha256: `d19031dc8d7ffaf649d811961698102b8bb6b05c06b68f09cb090f65341923fb`
config_key: `{"entropy_coef": 0.0, "lr": 3e-05, "opponent_mode": "rulebase", "rule_kl_coef": 0.01}`
diagnostic_seed_registry_id: `learning_signal_20260425_step35_v1`
diagnostic_seed_hash: `0f42628c88353290`
primary_blocker: `BATCH_OVERFITS_BUT_ROLLOUT_FAILS`

## Batch Signal

- batch_size: `795`
- advantage_std: `0.690577`
- selected_non_top1_positive_advantage_count: `21`
- mean_delta_needed_to_flip_top1: `3.00048`
- learner_deal_in_rate: `0.25`

## Stability

- rulebase_fallback_count: `0`
- illegal_action_rate_fail_closed: `0`
- fallback_rate_fail_closed: `0`
- forced_terminal_missed_fail_closed: `0`

## Fixed-Batch Overfit

- best_top1_action_changed_rate: `0.0201258`
- best_neural_delta_abs_max: `5.22682`
- pass_actor_can_move: `True`
- gradient_dead: `False`
- prior_margin_too_large: `False`
