# KeqingRL Controlled Discard-Only Sweep

seed: `20260425`
configs: `48`
ablation_profile: `actor-only-small`
profile_config_count: `48`
iterations/config: `3`
rollout_episodes/config/iter: `16`
eval_episodes/config: `16`
eval_seed_registry_id: `base=202604250000:stride=1:count=16`
eval_seed_count: `16`
eval_seed_hash: `0f42628c88353290`
shared_eval_seeds: `true`
eval_seed_policy: `forced_shared_across_configs`
source_type: `retrained_config`
comparison_note: `config repeat only; use checkpoint eval for candidate comparison`
eval_scope: `single learner seat 0; use paired checkpoint eval for seat rotation validation`

## Scope

- style_context: neutral
- action_scope: DISCARD only
- reward_spec: fixed default
- opponents: rule_prior_greedy, rulebase
- no teacher imitation, no full action RL, no style variants

## Stability Checks

- max illegal_action_rate_fail_closed: 0
- max fallback_rate_fail_closed: 0
- max forced_terminal_missed_fail_closed: 0
- max rulebase_fallback_count: 0
- max rulebase_chosen_missing_count: 0
- max rulebase_chosen_not_found_count: 0
- max rulebase_batch_unsupported_count: 0
- fail-closed note: these are hard raise gates, not observed recoverable event rates
- trace counters observed: forced_terminal_preempt_count, autopilot_terminal_count
- trace counters pending: illegal_attempt_count, fallback_policy_count, forced_terminal_missed_count
- caveat: this is a tiny smoke-scale matrix, not strength evidence

## Top Configs

- id=0 opponent=rulebase rule_kl=0 entropy=0 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=0.061525 clip=0.743017 rule_kl_obs=5.33555e-06 rule_agree=1 delta_mean=0.00428477 delta_max=0.0247154 top1_changed=0
- id=18 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=2.35721e-06 clip=0 rule_kl_obs=1.36088e-05 rule_agree=1 delta_mean=0.0103169 delta_max=0.026796 top1_changed=0
- id=34 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=1.87834e-06 clip=0 rule_kl_obs=7.88588e-06 rule_agree=1 delta_mean=0.00992958 delta_max=0.034057 top1_changed=0
- id=24 opponent=rule_prior_greedy rule_kl=0 entropy=0 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=7.83079e-07 clip=0 rule_kl_obs=4.31109e-06 rule_agree=1 delta_mean=0.00729523 delta_max=0.0352838 top1_changed=0
- id=40 opponent=rule_prior_greedy rule_kl=0 entropy=0 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=3.68061e-06 clip=0 rule_kl_obs=1.53392e-05 rule_agree=1 delta_mean=0.00702409 delta_max=0.0364432 top1_changed=0
- id=32 opponent=rule_prior_greedy rule_kl=0 entropy=0 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=5.68157e-06 clip=0 rule_kl_obs=2.67417e-05 rule_agree=1 delta_mean=0.0109697 delta_max=0.0392342 top1_changed=0
- id=42 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=3.41066e-06 clip=0 rule_kl_obs=1.37825e-05 rule_agree=1 delta_mean=0.00744821 delta_max=0.0416264 top1_changed=0
- id=2 opponent=rulebase rule_kl=0 entropy=0.005 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=0.0653242 clip=0.743229 rule_kl_obs=1.63004e-05 rule_agree=1 delta_mean=0.0141678 delta_max=0.0524914 top1_changed=0
- id=16 opponent=rule_prior_greedy rule_kl=0 entropy=0 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=1.38597e-05 clip=0 rule_kl_obs=5.77201e-05 rule_agree=1 delta_mean=0.0181053 delta_max=0.0580735 top1_changed=0
- id=10 opponent=rulebase rule_kl=0 entropy=0.005 lr=0.001 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=0.0629306 clip=0.742574 rule_kl_obs=3.1907e-05 rule_agree=1 delta_mean=0.0143521 delta_max=0.0629301 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
- `configs/<rerun_config_id>_*/config.json`
- `configs/<rerun_config_id>_*/policy_final.pt`
- `configs/<rerun_config_id>_*/optimizer_final.pt`
