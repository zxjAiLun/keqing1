# KeqingRL Controlled Discard-Only Sweep

seed: `20260425`
configs: `60`
iterations/config: `3`
rollout_episodes/config/iter: `2`
eval_episodes/config: `2`

## Scope

- style_context: neutral
- action_scope: DISCARD only
- reward_spec: fixed default
- opponents: rule_prior_greedy
- no teacher imitation, no full action RL, no style variants

## Top Configs

- id=57 rule_kl=0.05 entropy=0.01 lr=3e-05 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=4.88443e-13 clip=0 rule_kl_obs=-4.63314e-09 rule_agree=1 delta_mean=2.52871e-05 delta_max=6.40683e-05 top1_changed=0
- id=45 rule_kl=0.02 entropy=0.01 lr=3e-05 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=2.26526e-12 clip=0 rule_kl_obs=-1.63725e-09 rule_agree=1 delta_mean=2.93065e-05 delta_max=7.15811e-05 top1_changed=0
- id=46 rule_kl=0.02 entropy=0.01 lr=0.0001 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=6.39892e-12 clip=0 rule_kl_obs=-3.15322e-08 rule_agree=1 delta_mean=9.08365e-05 delta_max=0.000286933 top1_changed=0
- id=58 rule_kl=0.05 entropy=0.01 lr=0.0001 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=2.19075e-11 clip=0 rule_kl_obs=-2.24896e-08 rule_agree=1 delta_mean=0.000253507 delta_max=0.000391186 top1_changed=0
- id=8 rule_kl=0 entropy=0.005 lr=0.0003 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=3.45647e-11 clip=0 rule_kl_obs=-5.392e-09 rule_agree=1 delta_mean=0.000289516 delta_max=0.000785288 top1_changed=0
- id=59 rule_kl=0.05 entropy=0.01 lr=0.0003 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=4.93456e-11 clip=0 rule_kl_obs=-8.17888e-10 rule_agree=1 delta_mean=0.000510586 delta_max=0.000928385 top1_changed=0
- id=35 rule_kl=0.01 entropy=0.01 lr=0.0003 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=6.12012e-11 clip=0 rule_kl_obs=1.6125e-08 rule_agree=1 delta_mean=0.000880051 delta_max=0.00127947 top1_changed=0
- id=29 rule_kl=0.01 entropy=0.001 lr=0.0003 eval_rank_pt=1 fourth=0 deal_in=0 approx_kl=8.89826e-11 clip=0 rule_kl_obs=-6.86413e-10 rule_agree=1 delta_mean=0.000825828 delta_max=0.00148104 top1_changed=0
- id=30 rule_kl=0.01 entropy=0.005 lr=3e-05 eval_rank_pt=0.75 fourth=0 deal_in=0 approx_kl=4.12674e-13 clip=0 rule_kl_obs=-2.00885e-08 rule_agree=1 delta_mean=2.31705e-05 delta_max=7.01016e-05 top1_changed=0
- id=6 rule_kl=0 entropy=0.005 lr=3e-05 eval_rank_pt=0.75 fourth=0 deal_in=0 approx_kl=5.21822e-12 clip=0 rule_kl_obs=-1.08731e-08 rule_agree=1 delta_mean=2.53631e-05 delta_max=8.54777e-05 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
