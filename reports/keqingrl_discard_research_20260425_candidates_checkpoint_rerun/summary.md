# KeqingRL Controlled Discard-Only Sweep

seed: `20260425`
configs: `4`
iterations/config: `5`
rollout_episodes/config/iter: `16`
eval_episodes/config: `64`
eval_seed_registry_id: `base=202604250000:stride=1:count=64`
eval_seed_count: `64`
eval_seed_hash: `809ce4e45eb9ebe3`

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
- trace counters pending: illegal_attempt_count, fallback_policy_count, forced_terminal_preempt_count, forced_terminal_missed_count, autopilot_terminal_count
- caveat: this is a tiny smoke-scale matrix, not strength evidence

## Top Configs

- id=1 opponent=rule_prior_greedy rule_kl=0.05 entropy=0.01 lr=3e-05 eval_rank_pt=-0.140625 fourth=0.40625 learner_deal_in=0.09375 approx_kl=1.7395e-13 clip=0 rule_kl_obs=-2.77217e-09 rule_agree=1 delta_mean=0.000150272 delta_max=0.000207492 top1_changed=0
- id=0 opponent=rulebase rule_kl=0.01 entropy=0.01 lr=3e-05 eval_rank_pt=-0.140625 fourth=0.40625 learner_deal_in=0.09375 approx_kl=7.51057e-12 clip=0 rule_kl_obs=-1.62036e-08 rule_agree=1 delta_mean=0.000119777 delta_max=0.000216918 top1_changed=0
- id=2 opponent=rulebase rule_kl=0.01 entropy=0 lr=3e-05 eval_rank_pt=-0.140625 fourth=0.40625 learner_deal_in=0.09375 approx_kl=1.58838e-11 clip=0 rule_kl_obs=6.98615e-09 rule_agree=1 delta_mean=0.000131888 delta_max=0.000266433 top1_changed=0
- id=3 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.0003 eval_rank_pt=-0.140625 fourth=0.40625 learner_deal_in=0.09375 approx_kl=9.59406e-11 clip=0 rule_kl_obs=-1.35571e-08 rule_agree=1 delta_mean=0.000193201 delta_max=0.000897105 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
- `configs/<rerun_config_id>_*/config.json`
- `configs/<rerun_config_id>_*/policy_final.pt`
- `configs/<rerun_config_id>_*/optimizer_final.pt`
