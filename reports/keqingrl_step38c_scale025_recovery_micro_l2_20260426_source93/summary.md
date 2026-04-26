# KeqingRL Step 3.8C Scale=0.25 Micro-L2 Recovery

source_config_id: `93`
base_config: `cfg10: lr=0.003, update_epochs=4, clip_eps=0.2, behavior_temperature=1.25, rule_kl=0.005`
rule_score_scale: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
seed_registry: `202604300000,202604310000,202604320000,202604330000`
episodes: `16`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Config Summary

- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` pass=3/4 movement=3/4 stability=4/4 eval_sanity=4/4 top1_mean=0.102732 top1_range=[0,0.159817] t_kl_mean=0.00557695 t_clip_mean=0.0752749 u_clip_mean=0.17095 delta_mean=0.147512 delta_max_max=0.940407 eval_fourth_mean=0.328125 deal_in_mean=0.09375
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` pass=3/4 movement=3/4 stability=4/4 eval_sanity=4/4 top1_mean=0.102732 top1_range=[0,0.159817] t_kl_mean=0.00564808 t_clip_mean=0.0752749 u_clip_mean=0.17095 delta_mean=0.149086 delta_max_max=0.956718 eval_fourth_mean=0.328125 deal_in_mean=0.09375
- `iter3_rulekl0.005_l20.0003_dclip0_dclipcoef0` pass=1/4 movement=2/4 stability=3/4 eval_sanity=4/4 top1_mean=0.0633952 top1_range=[0,0.168182] t_kl_mean=0.00703544 t_clip_mean=0.0981144 u_clip_mean=0.199561 delta_mean=0.181118 delta_max_max=1.11189 eval_fourth_mean=0.328125 deal_in_mean=0.0625

## Seed Rows

- `iter3_rulekl0.005_l20.0003_dclip0_dclipcoef0` seed=202604300000 pass=False top1=0 eff_margin=0.720312 non_top1_pos=76 t_kl=0.000887389 t_clip=0 u_clip=0.0152284 delta_mean=0.0446975 delta_max=0.108275 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.0003_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.168182 eff_margin=0.365631 non_top1_pos=112 t_kl=0.00523342 t_clip=0.0590909 u_clip=0.172727 delta_mean=0.23568 delta_max=1.0653 eval_fourth=0.375 deal_in=0.0625
- `iter3_rulekl0.005_l20.0003_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0.00816326 eff_margin=0.445391 non_top1_pos=123 t_kl=0.00286883 t_clip=0.00816326 u_clip=0.130612 delta_mean=0.162653 delta_max=0.774933 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.005_l20.0003_dclip0_dclipcoef0` seed=202604330000 pass=False top1=0.0772358 eff_margin=0.50804 non_top1_pos=135 t_kl=0.0191521 t_clip=0.325203 u_clip=0.479675 delta_mean=0.281442 delta_max=1.11189 eval_fourth=0.3125 deal_in=0
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` seed=202604300000 pass=True top1=0.151111 eff_margin=0.386309 non_top1_pos=99 t_kl=0.00448469 t_clip=0.0577778 u_clip=0.16 delta_mean=0.171128 delta_max=0.602804 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.159817 eff_margin=0.392745 non_top1_pos=107 t_kl=0.00900362 t_clip=0.114155 u_clip=0.319635 delta_mean=0.240548 delta_max=0.956718 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0 eff_margin=0.745902 non_top1_pos=138 t_kl=0.000149448 t_clip=0 u_clip=0 delta_mean=0.0239222 delta_max=0.0408541 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.1 eff_margin=0.421826 non_top1_pos=141 t_kl=0.00895457 t_clip=0.129167 u_clip=0.204167 delta_mean=0.160745 delta_max=0.891514 eval_fourth=0.3125 deal_in=0.0625
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604300000 pass=True top1=0.151111 eff_margin=0.388152 non_top1_pos=99 t_kl=0.00441275 t_clip=0.0577778 u_clip=0.16 delta_mean=0.170839 delta_max=0.574112 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.159817 eff_margin=0.394336 non_top1_pos=107 t_kl=0.00883514 t_clip=0.114155 u_clip=0.319635 delta_mean=0.236269 delta_max=0.940407 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0 eff_margin=0.745895 non_top1_pos=138 t_kl=0.000149284 t_clip=0 u_clip=0 delta_mean=0.0239678 delta_max=0.0411505 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.1 eff_margin=0.42251 non_top1_pos=141 t_kl=0.00891063 t_clip=0.129167 u_clip=0.204167 delta_mean=0.158974 delta_max=0.882218 eval_fourth=0.3125 deal_in=0.0625

## Interpretation

- `delta_l2=0.0005` matches the previous `0.001` baseline at pass 3/4, with similar movement and stability, but does not recover seed 202604320000.
- `delta_l2=0.0003` underperforms: pass 1/4, movement 2/4, and seed 202604330000 exceeds the tempered clip threshold.
- The persistent failure mode remains seed-specific under-movement, not eval sanity collapse.
- Do not lower `rule_score_scale`. The next useful branch is an adaptive recovery/early-stop gate or a targeted update-pressure adjustment for under-moving seeds.

## Artifacts

- `step38c_micro_l2_summary.json`
- `step38c_micro_l2_summary.csv`
