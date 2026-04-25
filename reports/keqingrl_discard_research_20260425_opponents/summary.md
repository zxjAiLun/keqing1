# KeqingRL Controlled Discard-Only Sweep

seed: `20260425`
configs: `120`
iterations/config: `3`
rollout_episodes/config/iter: `2`
eval_episodes/config: `2`

## Scope

- style_context: neutral
- action_scope: DISCARD only
- reward_spec: fixed default
- opponents: rule_prior_greedy, rulebase
- no teacher imitation, no full action RL, no style variants

## Stability Checks

- max illegal_action_rate: 0
- max fallback_rate: 0
- max forced_terminal_missed: 0
- caveat: this is a tiny smoke-scale matrix, not strength evidence

## Top Configs

- id=57 opponent=rule_prior_greedy rule_kl=0.05 entropy=0.01 lr=3e-05 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=4.88443e-13 clip=0 rule_kl_obs=-4.63314e-09 rule_agree=1 delta_mean=2.52871e-05 delta_max=6.40683e-05 top1_changed=0
- id=45 opponent=rule_prior_greedy rule_kl=0.02 entropy=0.01 lr=3e-05 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=2.26526e-12 clip=0 rule_kl_obs=-1.63725e-09 rule_agree=1 delta_mean=2.93065e-05 delta_max=7.15811e-05 top1_changed=0
- id=105 opponent=rulebase rule_kl=0.02 entropy=0.01 lr=3e-05 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=1.87591e-13 clip=0 rule_kl_obs=-9.38466e-10 rule_agree=1 delta_mean=4.62248e-05 delta_max=9.45717e-05 top1_changed=0
- id=93 opponent=rulebase rule_kl=0.01 entropy=0.01 lr=3e-05 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=2.7298e-12 clip=0 rule_kl_obs=6.45451e-09 rule_agree=1 delta_mean=5.81406e-05 delta_max=9.71996e-05 top1_changed=0
- id=84 opponent=rulebase rule_kl=0.01 entropy=0 lr=3e-05 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=3.14769e-12 clip=0 rule_kl_obs=3.20874e-09 rule_agree=1 delta_mean=7.55016e-05 delta_max=0.000139698 top1_changed=0
- id=46 opponent=rule_prior_greedy rule_kl=0.02 entropy=0.01 lr=0.0001 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=6.39892e-12 clip=0 rule_kl_obs=-3.15322e-08 rule_agree=1 delta_mean=9.08365e-05 delta_max=0.000286933 top1_changed=0
- id=58 opponent=rule_prior_greedy rule_kl=0.05 entropy=0.01 lr=0.0001 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=2.19075e-11 clip=0 rule_kl_obs=-2.24896e-08 rule_agree=1 delta_mean=0.000253507 delta_max=0.000391186 top1_changed=0
- id=8 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.0003 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=3.45647e-11 clip=0 rule_kl_obs=-5.392e-09 rule_agree=1 delta_mean=0.000289516 delta_max=0.000785288 top1_changed=0
- id=64 opponent=rulebase rule_kl=0 entropy=0.001 lr=0.0001 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=7.64631e-12 clip=0 rule_kl_obs=2.0261e-09 rule_agree=1 delta_mean=0.000473885 delta_max=0.000790588 top1_changed=0
- id=59 opponent=rule_prior_greedy rule_kl=0.05 entropy=0.01 lr=0.0003 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=4.93456e-11 clip=0 rule_kl_obs=-8.17888e-10 rule_agree=1 delta_mean=0.000510586 delta_max=0.000928385 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
