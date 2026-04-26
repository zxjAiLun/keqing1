# KeqingRL Step 3.8C Scale=0.25 Rule-KL Midpoint Recovery

source_config_id: `93`
base_config: `cfg10: lr=0.003, update_epochs=4, clip_eps=0.2, behavior_temperature=1.25`
rule_score_scale: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
seed_registry: `202604300000,202604310000,202604320000,202604330000`
episodes: `16`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Config Summary

- `iter3_rulekl0.0075_l20.001_dclip0_dclipcoef0` pass=3/4 movement=3/4 stability=4/4 eval_sanity=4/4 top1_mean=0.101621 top1_range=[0,0.159817] t_kl_mean=0.00565922 t_clip_mean=0.0730223 u_clip_mean=0.1698 delta_mean=0.147924 delta_max_max=0.933189 eval_fourth_mean=0.328125 deal_in_mean=0.09375
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` pass=3/4 movement=3/4 stability=4/4 eval_sanity=4/4 top1_mean=0.102732 top1_range=[0,0.159817] t_kl_mean=0.00557695 t_clip_mean=0.0752749 u_clip_mean=0.17095 delta_mean=0.147512 delta_max_max=0.940407 eval_fourth_mean=0.328125 deal_in_mean=0.09375
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` pass=3/4 movement=3/4 stability=4/4 eval_sanity=4/4 top1_mean=0.102732 top1_range=[0,0.159817] t_kl_mean=0.00564808 t_clip_mean=0.0752749 u_clip_mean=0.17095 delta_mean=0.149086 delta_max_max=0.956718 eval_fourth_mean=0.328125 deal_in_mean=0.09375
- `iter3_rulekl0.0075_l20.0005_dclip0_dclipcoef0` pass=2/4 movement=2/4 stability=4/4 eval_sanity=4/4 top1_mean=0.0597479 top1_range=[0,0.157658] t_kl_mean=0.00668741 t_clip_mean=0.0886152 u_clip_mean=0.197363 delta_mean=0.172269 delta_max_max=1.06407 eval_fourth_mean=0.34375 deal_in_mean=0.078125
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` pass=2/4 movement=2/4 stability=4/4 eval_sanity=4/4 top1_mean=0.118584 top1_range=[0,0.337963] t_kl_mean=0.00644736 t_clip_mean=0.081873 u_clip_mean=0.181734 delta_mean=0.19711 delta_max_max=1.16868 eval_fourth_mean=0.34375 deal_in_mean=0.09375

## Seed Rows

- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` seed=202604300000 pass=True top1=0.151111 eff_margin=0.386309 non_top1_pos=99 t_kl=0.00448469 t_clip=0.0577778 u_clip=0.16 delta_mean=0.171128 delta_max=0.602804 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.159817 eff_margin=0.392745 non_top1_pos=107 t_kl=0.00900362 t_clip=0.114155 u_clip=0.319635 delta_mean=0.240548 delta_max=0.956718 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0 eff_margin=0.745902 non_top1_pos=138 t_kl=0.000149448 t_clip=0 u_clip=0 delta_mean=0.0239222 delta_max=0.0408541 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.005_l20.0005_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.1 eff_margin=0.421826 non_top1_pos=141 t_kl=0.00895457 t_clip=0.129167 u_clip=0.204167 delta_mean=0.160745 delta_max=0.891514 eval_fourth=0.3125 deal_in=0.0625
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604300000 pass=True top1=0.151111 eff_margin=0.388152 non_top1_pos=99 t_kl=0.00441275 t_clip=0.0577778 u_clip=0.16 delta_mean=0.170839 delta_max=0.574112 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.159817 eff_margin=0.394336 non_top1_pos=107 t_kl=0.00883514 t_clip=0.114155 u_clip=0.319635 delta_mean=0.236269 delta_max=0.940407 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0 eff_margin=0.745895 non_top1_pos=138 t_kl=0.000149284 t_clip=0 u_clip=0 delta_mean=0.0239678 delta_max=0.0411505 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.1 eff_margin=0.42251 non_top1_pos=141 t_kl=0.00891063 t_clip=0.129167 u_clip=0.204167 delta_mean=0.158974 delta_max=0.882218 eval_fourth=0.3125 deal_in=0.0625
- `iter3_rulekl0.0075_l20.0005_dclip0_dclipcoef0` seed=202604300000 pass=False top1=0 eff_margin=0.720845 non_top1_pos=76 t_kl=0.000893731 t_clip=0 u_clip=0.0152284 delta_mean=0.0407755 delta_max=0.100683 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.0075_l20.0005_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.157658 eff_margin=0.352039 non_top1_pos=108 t_kl=0.00548799 t_clip=0.0495495 u_clip=0.18018 delta_mean=0.222662 delta_max=0.829815 eval_fourth=0.5 deal_in=0.125
- `iter3_rulekl0.0075_l20.0005_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0.00816326 eff_margin=0.444243 non_top1_pos=123 t_kl=0.00289374 t_clip=0.00816326 u_clip=0.134694 delta_mean=0.154364 delta_max=0.758956 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.0075_l20.0005_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.0731707 eff_margin=0.515612 non_top1_pos=133 t_kl=0.0174742 t_clip=0.296748 u_clip=0.45935 delta_mean=0.271274 delta_max=1.06407 eval_fourth=0.25 deal_in=0
- `iter3_rulekl0.0075_l20.001_dclip0_dclipcoef0` seed=202604300000 pass=True top1=0.146667 eff_margin=0.391706 non_top1_pos=99 t_kl=0.0043345 t_clip=0.0533333 u_clip=0.146667 delta_mean=0.169412 delta_max=0.571747 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.0075_l20.001_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.159817 eff_margin=0.394777 non_top1_pos=107 t_kl=0.00905384 t_clip=0.109589 u_clip=0.324201 delta_mean=0.237813 delta_max=0.933189 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.0075_l20.001_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0 eff_margin=0.745893 non_top1_pos=137 t_kl=0.000149601 t_clip=0 u_clip=0 delta_mean=0.0236172 delta_max=0.0395896 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.0075_l20.001_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.1 eff_margin=0.421716 non_top1_pos=141 t_kl=0.00909893 t_clip=0.129167 u_clip=0.208333 delta_mean=0.160852 delta_max=0.898365 eval_fourth=0.3125 deal_in=0.0625
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` seed=202604300000 pass=False top1=0.337963 eff_margin=0.289144 non_top1_pos=43 t_kl=0.0146384 t_clip=0.217593 u_clip=0.430556 delta_mean=0.325478 delta_max=1.16868 eval_fourth=0.4375 deal_in=0.125
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.0853081 eff_margin=0.439284 non_top1_pos=170 t_kl=0.00512128 t_clip=0.0663507 u_clip=0.0805687 delta_mean=0.175885 delta_max=0.681741 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` seed=202604320000 pass=True top1=0.0510638 eff_margin=0.474612 non_top1_pos=154 t_kl=0.00278182 t_clip=0.0297872 u_clip=0.114894 delta_mean=0.172287 delta_max=0.76543 eval_fourth=0.3125 deal_in=0.125
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` seed=202604330000 pass=False top1=0 eff_margin=0.535583 non_top1_pos=99 t_kl=0.00324794 t_clip=0.0137615 u_clip=0.100917 delta_mean=0.114789 delta_max=0.483864 eval_fourth=0.25 deal_in=0

## Interpretation

- `rule_kl=0.0075` does not recover seed 202604320000; both L2 settings still miss the movement window on that seed.
- `rule_kl=0.0075, delta_l2=0.001` remains a 3/4 candidate, but it is not better than `rule_kl=0.005, delta_l2=0.001`.
- `rule_kl=0.0075, delta_l2=0.0005` drops to 2/4 because seed 202604300000 under-moves.
- The narrow rule-KL midpoint does not remove the need for an adaptive recovery gate or targeted handling of under-moving seeds.

## Artifacts

- `step38c_rulekl_mid_summary.json`
- `step38c_rulekl_mid_summary.csv`
