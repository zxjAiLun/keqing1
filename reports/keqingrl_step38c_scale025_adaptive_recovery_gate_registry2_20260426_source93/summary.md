# KeqingRL Step 3.8C Adaptive Recovery Gate Registry 2

source_config_id: `93`
base_config: `cfg10: lr=0.003, update_epochs=4, clip_eps=0.2, behavior_temperature=1.25, rule_kl=0.005, delta_l2=0.001`
rule_score_scale: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
seed_registry: `202604340000,202604350000,202604360000,202604370000`
adaptive_recovery_gate: `enabled=True, max_extra_epochs=4, rollback_on_unstable_or_overmove=True`
games_count_note: `per seed = 3*16 rollout episodes + 16 eval games = 64; total = 256; max_kyokus=1`
elapsed: `parallel_wall_seconds=182.74, per_seed_mean=182.19, per_seed_max=182.74`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Summary

- pass=4/4 movement=4/4 stability=4/4 eval_sanity=4/4
- top1_mean=0.135341 top1_range=[0.0434783,0.236948] t_kl_mean=0.00986924 t_clip_mean=0.147089 u_clip_mean=0.284961
- games: rollout=192 eval=64 total=256
- recovery: extra=24 attempted=24 rejected_epochs=4

## Seed Rows

- seed=202604340000 pass=True top1=0.236948 t_kl=0.0116571 t_clip=0.156627 u_clip=0.373494 delta_max=1.01719 eval_fourth=0.4375 deal_in=0.125 games=64 elapsed=181.42s extra=4 attempted=4 rejected_epochs=4 stops=budget_exhausted,base_rejected_unstable_or_overmove,not_needed
- seed=202604350000 pass=True top1=0.138889 t_kl=0.0109624 t_clip=0.199074 u_clip=0.300926 delta_max=1.35986 eval_fourth=0.4375 deal_in=0.125 games=64 elapsed=181.93s extra=4 attempted=4 rejected_epochs=0 stops=budget_exhausted,not_needed,not_needed
- seed=202604360000 pass=True top1=0.0434783 t_kl=0.0072684 t_clip=0.0948617 u_clip=0.217391 delta_max=0.633711 eval_fourth=0.25 deal_in=0 games=64 elapsed=182.74s extra=12 attempted=12 rejected_epochs=0 stops=budget_exhausted,budget_exhausted,target_reached
- seed=202604370000 pass=True top1=0.122047 t_kl=0.00958907 t_clip=0.137795 u_clip=0.248031 delta_max=0.934528 eval_fourth=0.375 deal_in=0.125 games=64 elapsed=182.65s extra=4 attempted=4 rejected_epochs=0 stops=budget_exhausted,not_needed,not_needed

## Iteration Recovery Rows

- seed=202604340000 iter=0 top1=0.0130435 t_kl=0.0124055 t_clip=0.2 u_clip=0.352174 delta_max=0.99386 extra=4 attempted=4 rejected_epochs=0 stop=budget_exhausted pre_top1=0
- seed=202604340000 iter=1 top1=0.00497512 t_kl=6.96403e-15 t_clip=0 u_clip=0 delta_max=0.869017 extra=0 attempted=0 rejected_epochs=4 stop=base_rejected_unstable_or_overmove pre_top1=0.303483
- seed=202604340000 iter=2 top1=0.236948 t_kl=0.0116571 t_clip=0.156627 u_clip=0.373494 delta_max=1.01719 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.236948
- seed=202604350000 iter=0 top1=0 t_kl=0.0020135 t_clip=0.00421941 u_clip=0.0548523 delta_max=0.362973 extra=4 attempted=4 rejected_epochs=0 stop=budget_exhausted pre_top1=0
- seed=202604350000 iter=1 top1=0.0585774 t_kl=0.0134119 t_clip=0.209205 u_clip=0.322176 delta_max=0.83454 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.0585774
- seed=202604350000 iter=2 top1=0.138889 t_kl=0.0109624 t_clip=0.199074 u_clip=0.300926 delta_max=1.35986 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.138889
- seed=202604360000 iter=0 top1=0.00900901 t_kl=0.00800427 t_clip=0.121622 u_clip=0.202703 delta_max=0.825693 extra=4 attempted=4 rejected_epochs=0 stop=budget_exhausted pre_top1=0
- seed=202604360000 iter=1 top1=0 t_kl=0.0100337 t_clip=0.146465 u_clip=0.20202 delta_max=0.647459 extra=4 attempted=4 rejected_epochs=0 stop=budget_exhausted pre_top1=0
- seed=202604360000 iter=2 top1=0.0434783 t_kl=0.0072684 t_clip=0.0948617 u_clip=0.217391 delta_max=0.633711 extra=4 attempted=4 rejected_epochs=0 stop=target_reached pre_top1=0
- seed=202604370000 iter=0 top1=0 t_kl=0.00844179 t_clip=0.12037 u_clip=0.277778 delta_max=0.854332 extra=4 attempted=4 rejected_epochs=0 stop=budget_exhausted pre_top1=0
- seed=202604370000 iter=1 top1=0.0833333 t_kl=0.00834105 t_clip=0.111111 u_clip=0.329365 delta_max=1.24887 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.0833333
- seed=202604370000 iter=2 top1=0.122047 t_kl=0.00958907 t_clip=0.137795 u_clip=0.248031 delta_max=0.934528 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.122047

## Interpretation

- Registry 2 also passes 4/4 under the adaptive recovery gate.
- Several first-iteration recovery attempts exhausted the 4-extra-epoch budget, but later rollout/update iterations moved into the target window.
- The result supports the gate as a diagnostic stabilizer across two seed registries, but eval remains learner seat 0 smoke only.

## Artifacts

- `step38c_adaptive_recovery_gate_registry2_summary.json`
- `step38c_adaptive_recovery_gate_registry2_summary.csv`
- `step38c_adaptive_recovery_gate_registry2_rows.csv`
- `step38c_adaptive_recovery_gate_registry2_iterations.csv`
