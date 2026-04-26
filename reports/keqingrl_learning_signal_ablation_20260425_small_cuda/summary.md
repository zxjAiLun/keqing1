# KeqingRL Controlled Discard-Only Sweep

seed: `20260425`
configs: `48`
ablation_profile: `small`
profile_config_count: `48`
iterations/config: `5`
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

- id=32 opponent=rule_prior_greedy rule_kl=0 entropy=0 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=1.61304e-09 clip=0 rule_kl_obs=3.67714e-09 rule_agree=1 delta_mean=0.000301777 delta_max=0.00116449 top1_changed=0
- id=2 opponent=rulebase rule_kl=0 entropy=0.005 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=0.0612638 clip=0.743304 rule_kl_obs=4.75403e-09 rule_agree=1 delta_mean=0.000789749 delta_max=0.00186935 top1_changed=0
- id=18 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=3.86517e-09 clip=0 rule_kl_obs=1.35788e-08 rule_agree=1 delta_mean=0.000346229 delta_max=0.00187391 top1_changed=0
- id=16 opponent=rule_prior_greedy rule_kl=0 entropy=0 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=1.55499e-09 clip=0 rule_kl_obs=4.23167e-09 rule_agree=1 delta_mean=0.000798922 delta_max=0.0020961 top1_changed=0
- id=36 opponent=rule_prior_greedy rule_kl=0.001 entropy=0 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=1.68195e-09 clip=0 rule_kl_obs=3.41671e-09 rule_agree=1 delta_mean=0.00108316 delta_max=0.00222043 top1_changed=0
- id=6 opponent=rulebase rule_kl=0.001 entropy=0.005 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=0.0677635 clip=0.741752 rule_kl_obs=1.36413e-08 rule_agree=1 delta_mean=0.000631426 delta_max=0.00232449 top1_changed=0
- id=34 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=2.86805e-09 clip=0 rule_kl_obs=1.01353e-08 rule_agree=1 delta_mean=0.00103805 delta_max=0.00242251 top1_changed=0
- id=20 opponent=rule_prior_greedy rule_kl=0.001 entropy=0 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=3.80896e-09 clip=0 rule_kl_obs=9.91837e-09 rule_agree=1 delta_mean=0.00158086 delta_max=0.00256352 top1_changed=0
- id=22 opponent=rule_prior_greedy rule_kl=0.001 entropy=0.005 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=1.988e-09 clip=0 rule_kl_obs=6.09367e-09 rule_agree=1 delta_mean=0.00138361 delta_max=0.00257913 top1_changed=0
- id=0 opponent=rulebase rule_kl=0 entropy=0 lr=0.0003 eval_rank_pt=-0.3125 fourth=0.4375 learner_deal_in=0.1875 approx_kl=0.0636715 clip=0.743091 rule_kl_obs=2.74545e-09 rule_agree=1 delta_mean=0.00158436 delta_max=0.00301514 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
- `configs/<rerun_config_id>_*/config.json`
- `configs/<rerun_config_id>_*/policy_final.pt`
- `configs/<rerun_config_id>_*/optimizer_final.pt`
