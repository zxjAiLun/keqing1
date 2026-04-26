# KeqingRL Learning-Signal Diagnostics

source_config_id: `8`
checkpoint_path: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_003_source_8/policy_final.pt`
checkpoint_sha256: `09f2aa9a65639583a79695c5eca26f8361c4d60ca447cb1e037aae0a0360268a`
config_key: `{"entropy_coef": 0.005, "lr": 0.0003, "opponent_mode": "rule_prior_greedy", "rule_kl_coef": 0.0}`
diagnostic_seed_registry_id: `learning_signal_20260425_step35_v1`
diagnostic_seed_hash: `0f42628c88353290`
primary_blocker: `BATCH_OVERFITS_BUT_ROLLOUT_FAILS`

## Batch Signal

- batch_size: `749`
- advantage_std: `0.732937`
- selected_non_top1_positive_advantage_count: `40`
- mean_delta_needed_to_flip_top1: `3.00054`
- learner_deal_in_rate: `0.1875`

## Stability

- rulebase_fallback_count: `0`
- illegal_action_rate_fail_closed: `0`
- fallback_rate_fail_closed: `0`
- forced_terminal_missed_fail_closed: `0`

## Fixed-Batch Overfit

- best_top1_action_changed_rate: `0.0400534`
- best_neural_delta_abs_max: `4.79099`
- pass_actor_can_move: `True`
- gradient_dead: `False`
- prior_margin_too_large: `False`
