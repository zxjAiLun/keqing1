# KeqingRL Online Stabilization Sweep

source_config_id: `93`
bridge_summary: `reports/keqingrl_fixed_online_bridge_20260426_step36d_learner_only/summary.csv`
grid_count: `162`
eval_count: `0`
online_seed_registry_id: `base=202604270000:stride=1:count=16`
stable_kl_threshold: `0.03`
stable_clip_threshold: `0.3`
status_counts: `{'STABLE_NO_MOVE': 162}`
max_top1_action_changed_rate: `0`
online_kl_range: `5.05526e-05..0.00516098`
online_clip_range: `0..0.177419`
max_neural_delta_abs_max: `2.42908`
conclusion: `no stable top-1 movement under this grid`

## Top Update Rows

- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.005 entropy=0.005 grad=0.25 top1=0 kl=5.05526e-05 clip=0 delta_mean=1.3425 delta_max=2.27829
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.005 entropy=0.005 grad=1 top1=0 kl=5.05531e-05 clip=0 delta_mean=1.3425 delta_max=2.27829
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.005 entropy=0.005 grad=0.5 top1=0 kl=5.05532e-05 clip=0 delta_mean=1.3425 delta_max=2.27829
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.005 entropy=0 grad=0.25 top1=0 kl=5.06383e-05 clip=0 delta_mean=1.34251 delta_max=2.27817
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.005 entropy=0 grad=0.5 top1=0 kl=5.06385e-05 clip=0 delta_mean=1.34251 delta_max=2.27817
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.005 entropy=0 grad=1 top1=0 kl=5.06386e-05 clip=0 delta_mean=1.34251 delta_max=2.27817
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.001 entropy=0 grad=0.25 top1=0 kl=5.09205e-05 clip=0 delta_mean=1.34244 delta_max=2.27844
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.001 entropy=0 grad=1 top1=0 kl=5.09208e-05 clip=0 delta_mean=1.34243 delta_max=2.27844
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.001 entropy=0 grad=0.5 top1=0 kl=5.09209e-05 clip=0 delta_mean=1.34243 delta_max=2.27844
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0 entropy=0.005 grad=0.25 top1=0 kl=5.10174e-05 clip=0 delta_mean=1.34242 delta_max=2.27862
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0 entropy=0.005 grad=0.5 top1=0 kl=5.10175e-05 clip=0 delta_mean=1.34242 delta_max=2.27861
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0 entropy=0.005 grad=1 top1=0 kl=5.10177e-05 clip=0 delta_mean=1.34242 delta_max=2.27861
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.001 entropy=0.005 grad=0.25 top1=0 kl=5.10353e-05 clip=0 delta_mean=1.34245 delta_max=2.27858
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0 entropy=0 grad=0.25 top1=0 kl=5.10355e-05 clip=0 delta_mean=1.34239 delta_max=2.2785
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0 entropy=0 grad=1 top1=0 kl=5.10356e-05 clip=0 delta_mean=1.34239 delta_max=2.2785
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0 entropy=0 grad=0.5 top1=0 kl=5.10357e-05 clip=0 delta_mean=1.34239 delta_max=2.2785
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.001 entropy=0.005 grad=0.5 top1=0 kl=5.10359e-05 clip=0 delta_mean=1.34244 delta_max=2.27858
- status=STABLE_NO_MOVE lr=0.0001 epochs=1 rule_kl=0.001 entropy=0.005 grad=1 top1=0 kl=5.1036e-05 clip=0 delta_mean=1.34244 delta_max=2.27858
- status=STABLE_NO_MOVE lr=0.0001 epochs=2 rule_kl=0.005 entropy=0 grad=0.25 top1=0 kl=0.000205216 clip=0.00806452 delta_mean=1.33117 delta_max=2.2731
- status=STABLE_NO_MOVE lr=0.0001 epochs=2 rule_kl=0.005 entropy=0.005 grad=0.25 top1=0 kl=0.00020523 clip=0.00806452 delta_mean=1.33111 delta_max=2.27322

## Eval Rows


## Artifacts

- `online_stabilization_sweep.json`
- `summary.csv`
- `eval.csv`
