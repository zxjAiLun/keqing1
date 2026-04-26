# KeqingRL Step 3.8C Scale=0.25 Adaptive Recovery Gate

source_config_id: `93`
base_config: `cfg10: lr=0.003, update_epochs=4, clip_eps=0.2, behavior_temperature=1.25, rule_kl=0.005, delta_l2=0.001`
rule_score_scale: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
seed_registry: `202604300000,202604310000,202604320000,202604330000`
episodes: `16`
adaptive_recovery_gate: `enabled=True, max_extra_epochs=4, rollback_on_unstable_or_overmove=True`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Config Summary

- `adaptive_gate_rulekl0.005_l20.001_extra4` pass=4/4 movement=4/4 stability=4/4 eval_sanity=4/4 top1_mean=0.12991 top1_range=[0.0322581,0.227053] t_kl_mean=0.00708983 t_clip_mean=0.118635 u_clip_mean=0.160332 delta_mean=0.228012 delta_max_max=1.95612 eval_fourth_mean=0.390625 deal_in_mean=0.09375 extra_epochs=15 attempted=16 rejected_epochs=17
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` pass=3/4 movement=3/4 stability=4/4 eval_sanity=4/4 top1_mean=0.102732 top1_range=[0,0.159817] t_kl_mean=0.00557695 t_clip_mean=0.0752749 u_clip_mean=0.17095 delta_mean=0.147512 delta_max_max=0.940407 eval_fourth_mean=0.328125 deal_in_mean=0.09375 extra_epochs=0 attempted=0 rejected_epochs=0

## Seed Rows

- `adaptive_gate_rulekl0.005_l20.001_extra4` seed=202604300000 pass=True top1=0.227053 eff_margin=0.42276 non_top1_pos=147 t_kl=0.0156815 t_clip=0.285024 u_clip=0.439614 delta_mean=0.326705 delta_max=1.95612 eval_fourth=0.4375 deal_in=0.1875 extra=3 attempted=4 rejected_epochs=1 stops=rejected_unstable_or_overmove,not_needed,not_needed
- `adaptive_gate_rulekl0.005_l20.001_extra4` seed=202604310000 pass=True top1=0.161157 eff_margin=0.386257 non_top1_pos=163 t_kl=1.07462e-14 t_clip=0 u_clip=0 delta_mean=0.241475 delta_max=0.995086 eval_fourth=0.4375 deal_in=0.0625 extra=4 attempted=4 rejected_epochs=8 stops=target_reached,base_rejected_unstable_or_overmove,base_rejected_unstable_or_overmove
- `adaptive_gate_rulekl0.005_l20.001_extra4` seed=202604320000 pass=True top1=0.0991736 eff_margin=0.380815 non_top1_pos=118 t_kl=7.0467e-15 t_clip=0 u_clip=0.00413223 delta_mean=0.146828 delta_max=0.713246 eval_fourth=0.375 deal_in=0.125 extra=4 attempted=4 rejected_epochs=8 stops=target_reached,base_rejected_unstable_or_overmove,base_rejected_unstable_or_overmove
- `adaptive_gate_rulekl0.005_l20.001_extra4` seed=202604330000 pass=True top1=0.0322581 eff_margin=0.473184 non_top1_pos=155 t_kl=0.0126778 t_clip=0.189516 u_clip=0.197581 delta_mean=0.197042 delta_max=0.563261 eval_fourth=0.3125 deal_in=0 extra=4 attempted=4 rejected_epochs=0 stops=target_reached,not_needed,not_needed
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604300000 pass=True top1=0.151111 eff_margin=0.388152 non_top1_pos=99 t_kl=0.00441275 t_clip=0.0577778 u_clip=0.16 delta_mean=0.170839 delta_max=0.574112 eval_fourth=0.375 deal_in=0.125 extra=0 attempted=0 rejected_epochs=0 stops=
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.159817 eff_margin=0.394336 non_top1_pos=107 t_kl=0.00883514 t_clip=0.114155 u_clip=0.319635 delta_mean=0.236269 delta_max=0.940407 eval_fourth=0.375 deal_in=0.125 extra=0 attempted=0 rejected_epochs=0 stops=
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0 eff_margin=0.745895 non_top1_pos=138 t_kl=0.000149284 t_clip=0 u_clip=0 delta_mean=0.0239678 delta_max=0.0411505 eval_fourth=0.25 deal_in=0.0625 extra=0 attempted=0 rejected_epochs=0 stops=
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.1 eff_margin=0.42251 non_top1_pos=141 t_kl=0.00891063 t_clip=0.129167 u_clip=0.204167 delta_mean=0.158974 delta_max=0.882218 eval_fourth=0.3125 deal_in=0.0625 extra=0 attempted=0 rejected_epochs=0 stops=

## Iteration Recovery Rows

- seed=202604300000 iter=0 top1=0.00900901 t_kl=0.00816853 t_clip=0.117117 u_clip=0.216216 delta_max=0.465754 extra=3 attempted=4 rejected_epochs=1 stop=rejected_unstable_or_overmove pre_top1=0
- seed=202604300000 iter=1 top1=0.118367 t_kl=0.0104667 t_clip=0.179592 u_clip=0.240816 delta_max=0.874979 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.118367
- seed=202604300000 iter=2 top1=0.227053 t_kl=0.0156815 t_clip=0.285024 u_clip=0.439614 delta_max=1.95612 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.227053
- seed=202604310000 iter=0 top1=0.056338 t_kl=0.012818 t_clip=0.169014 u_clip=0.323944 delta_max=0.914066 extra=4 attempted=4 rejected_epochs=0 stop=target_reached pre_top1=0
- seed=202604310000 iter=1 top1=0.131707 t_kl=1.03982e-14 t_clip=0 u_clip=0.00487805 delta_max=0.996866 extra=0 attempted=0 rejected_epochs=4 stop=base_rejected_unstable_or_overmove pre_top1=0.336585
- seed=202604310000 iter=2 top1=0.161157 t_kl=1.07462e-14 t_clip=0 u_clip=0 delta_max=0.995086 extra=0 attempted=0 rejected_epochs=4 stop=base_rejected_unstable_or_overmove pre_top1=0.252066
- seed=202604320000 iter=0 top1=0.0526316 t_kl=0.010752 t_clip=0.15311 u_clip=0.248804 delta_max=0.71678 extra=4 attempted=4 rejected_epochs=0 stop=target_reached pre_top1=0
- seed=202604320000 iter=1 top1=0.134715 t_kl=5.41191e-15 t_clip=0 u_clip=0.00518135 delta_max=0.72471 extra=0 attempted=0 rejected_epochs=4 stop=base_rejected_unstable_or_overmove pre_top1=0.331606
- seed=202604320000 iter=2 top1=0.0991736 t_kl=7.0467e-15 t_clip=0 u_clip=0.00413223 delta_max=0.713246 extra=0 attempted=0 rejected_epochs=4 stop=base_rejected_unstable_or_overmove pre_top1=0.293388
- seed=202604330000 iter=0 top1=0.0391304 t_kl=0.0142073 t_clip=0.204348 u_clip=0.295652 delta_max=0.990114 extra=4 attempted=4 rejected_epochs=0 stop=target_reached pre_top1=0
- seed=202604330000 iter=1 top1=0.240566 t_kl=0.0116719 t_clip=0.198113 u_clip=0.386792 delta_max=1.42748 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.240566
- seed=202604330000 iter=2 top1=0.0322581 t_kl=0.0126778 t_clip=0.189516 u_clip=0.197581 delta_max=0.563261 extra=0 attempted=0 rejected_epochs=0 stop=not_needed pre_top1=0.0322581

## Interpretation

- Adaptive recovery gate improves the cfg10 recovery candidate from 3/4 to 4/4 on the fixed seed registry.
- Seed 202604320000 is recovered from under-movement (`top1=0` baseline) to `top1=0.099174` without breaching KL/clip thresholds.
- The gate also rolls back unsafe base updates on seeds 202604310000 and 202604320000, keeping final movement inside the target window.
- This is still fixed-seed smoke with learner seat 0 only, so it is diagnostic recovery evidence, not strength evidence.
- Next step should be a repeat across a second seed registry or an adaptive-gate multi-seed registry expansion before paired seat-rotation eval.

## Artifacts

- `step38c_adaptive_recovery_gate_summary.json`
- `step38c_adaptive_recovery_gate_summary.csv`
- `step38c_adaptive_recovery_gate_iterations.csv`
