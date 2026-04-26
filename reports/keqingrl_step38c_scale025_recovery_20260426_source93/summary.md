# KeqingRL Step 3.8C Scale=0.25 Recovery Regularization

source_config_id: `93`
base_config: `cfg10: lr=0.003, update_epochs=4, clip_eps=0.2, behavior_temperature=1.25`
rule_score_scale: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
seed_registry: `202604300000,202604310000,202604320000,202604330000`
episodes: `16`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Config Summary

- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` pass=3/4 movement=3/4 stability=4/4 eval_sanity=4/4 top1_mean=0.102732 top1_range=[0,0.159817] t_kl_mean=0.00557695 t_clip_mean=0.0752749 u_clip_mean=0.17095 delta_mean=0.147512 delta_max_mean=0.609472 delta_max_max=0.940407 eval_fourth_mean=0.328125 deal_in_mean=0.09375
- `iter3_rulekl0.01_l20_dclip1_dclipcoef0.001` pass=3/4 movement=3/4 stability=4/4 eval_sanity=4/4 top1_mean=0.101651 top1_range=[0,0.164384] t_kl_mean=0.00567505 t_clip_mean=0.0741638 u_clip_mean=0.169839 delta_mean=0.149742 delta_max_mean=0.643642 delta_max_max=1.02329 eval_fourth_mean=0.328125 deal_in_mean=0.09375
- `iter3_rulekl0.01_l20_dclip0_dclipcoef0` pass=2/4 movement=2/4 stability=3/4 eval_sanity=4/4 top1_mean=0.17412 top1_range=[0,0.254386] t_kl_mean=0.0138497 t_clip_mean=0.157752 u_clip_mean=0.303464 delta_mean=0.438818 delta_max_mean=1.38635 delta_max_max=2.70342 eval_fourth_mean=0.34375 deal_in_mean=0.09375
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` pass=2/4 movement=2/4 stability=4/4 eval_sanity=4/4 top1_mean=0.118584 top1_range=[0,0.337963] t_kl_mean=0.00644736 t_clip_mean=0.081873 u_clip_mean=0.181734 delta_mean=0.19711 delta_max_mean=0.774929 delta_max_max=1.16868 eval_fourth_mean=0.34375 deal_in_mean=0.09375
- `iter3_rulekl0.005_l20_dclip0_dclipcoef0` pass=1/4 movement=2/4 stability=3/4 eval_sanity=4/4 top1_mean=0.0645316 top1_range=[0,0.172727] t_kl_mean=0.00713766 t_clip_mean=0.0992507 u_clip_mean=0.201713 delta_mean=0.216569 delta_max_mean=0.778331 delta_max_max=1.16846 eval_fourth_mean=0.328125 deal_in_mean=0.0625
- `iter3_rulekl0.005_l20_dclip1_dclipcoef0.001` pass=1/4 movement=2/4 stability=3/4 eval_sanity=4/4 top1_mean=0.0645316 top1_range=[0,0.172727] t_kl_mean=0.00713435 t_clip_mean=0.0992507 u_clip_mean=0.201713 delta_mean=0.214839 delta_max_mean=0.77381 delta_max_max=1.16846 eval_fourth_mean=0.328125 deal_in_mean=0.0625
- `iter5_rulekl0.005_l20.001_dclip0_dclipcoef0` pass=1/4 movement=1/4 stability=4/4 eval_sanity=4/4 top1_mean=0.171195 top1_range=[0,0.473029] t_kl_mean=0.00568382 t_clip_mean=0.0823873 u_clip_mean=0.159162 delta_mean=0.19494 delta_max_mean=0.776448 delta_max_max=1.57008 eval_fourth_mean=0.359375 deal_in_mean=0.125

## Seed Rows

- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604300000 pass=True top1=0.151111 eff_margin=0.388152 non_top1_pos=99 t_kl=0.00441275 t_clip=0.0577778 u_clip=0.16 delta_mean=0.170839 delta_max=0.574112 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.159817 eff_margin=0.394336 non_top1_pos=107 t_kl=0.00883514 t_clip=0.114155 u_clip=0.319635 delta_mean=0.236269 delta_max=0.940407 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0 eff_margin=0.745895 non_top1_pos=138 t_kl=0.000149284 t_clip=0 u_clip=0 delta_mean=0.0239678 delta_max=0.0411505 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.1 eff_margin=0.42251 non_top1_pos=141 t_kl=0.00891063 t_clip=0.129167 u_clip=0.204167 delta_mean=0.158974 delta_max=0.882218 eval_fourth=0.3125 deal_in=0.0625
- `iter3_rulekl0.005_l20_dclip0_dclipcoef0` seed=202604300000 pass=False top1=0 eff_margin=0.720149 non_top1_pos=76 t_kl=0.000888637 t_clip=0 u_clip=0.0152284 delta_mean=0.0832341 delta_max=0.150024 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.172727 eff_margin=0.365321 non_top1_pos=112 t_kl=0.00545407 t_clip=0.0636364 u_clip=0.177273 delta_mean=0.300404 delta_max=0.968941 eval_fourth=0.375 deal_in=0.0625
- `iter3_rulekl0.005_l20_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0.00816326 eff_margin=0.445108 non_top1_pos=123 t_kl=0.00287593 t_clip=0.00816326 u_clip=0.130612 delta_mean=0.195306 delta_max=0.825898 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.005_l20_dclip0_dclipcoef0` seed=202604330000 pass=False top1=0.0772358 eff_margin=0.507486 non_top1_pos=134 t_kl=0.019332 t_clip=0.325203 u_clip=0.48374 delta_mean=0.287332 delta_max=1.16846 eval_fourth=0.3125 deal_in=0
- `iter3_rulekl0.005_l20_dclip1_dclipcoef0.001` seed=202604300000 pass=False top1=0 eff_margin=0.720149 non_top1_pos=76 t_kl=0.000888637 t_clip=0 u_clip=0.0152284 delta_mean=0.0832341 delta_max=0.150024 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.005_l20_dclip1_dclipcoef0.001` seed=202604310000 pass=True top1=0.172727 eff_margin=0.365452 non_top1_pos=112 t_kl=0.00544086 t_clip=0.0636364 u_clip=0.177273 delta_mean=0.293484 delta_max=0.950855 eval_fourth=0.375 deal_in=0.0625
- `iter3_rulekl0.005_l20_dclip1_dclipcoef0.001` seed=202604320000 pass=False top1=0.00816326 eff_margin=0.445108 non_top1_pos=123 t_kl=0.00287593 t_clip=0.00816326 u_clip=0.130612 delta_mean=0.195306 delta_max=0.825898 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.005_l20_dclip1_dclipcoef0.001` seed=202604330000 pass=False top1=0.0772358 eff_margin=0.507486 non_top1_pos=134 t_kl=0.019332 t_clip=0.325203 u_clip=0.48374 delta_mean=0.287332 delta_max=1.16846 eval_fourth=0.3125 deal_in=0
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` seed=202604300000 pass=False top1=0.337963 eff_margin=0.289144 non_top1_pos=43 t_kl=0.0146384 t_clip=0.217593 u_clip=0.430556 delta_mean=0.325478 delta_max=1.16868 eval_fourth=0.4375 deal_in=0.125
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` seed=202604310000 pass=True top1=0.0853081 eff_margin=0.439284 non_top1_pos=170 t_kl=0.00512128 t_clip=0.0663507 u_clip=0.0805687 delta_mean=0.175885 delta_max=0.681741 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` seed=202604320000 pass=True top1=0.0510638 eff_margin=0.474612 non_top1_pos=154 t_kl=0.00278182 t_clip=0.0297872 u_clip=0.114894 delta_mean=0.172287 delta_max=0.76543 eval_fourth=0.3125 deal_in=0.125
- `iter3_rulekl0.01_l20.001_dclip0_dclipcoef0` seed=202604330000 pass=False top1=0 eff_margin=0.535583 non_top1_pos=99 t_kl=0.00324794 t_clip=0.0137615 u_clip=0.100917 delta_mean=0.114789 delta_max=0.483864 eval_fourth=0.25 deal_in=0
- `iter3_rulekl0.01_l20_dclip0_dclipcoef0` seed=202604300000 pass=True top1=0.204651 eff_margin=0.347985 non_top1_pos=167 t_kl=0.0223218 t_clip=0.246512 u_clip=0.344186 delta_mean=0.892818 delta_max=2.70342 eval_fourth=0.4375 deal_in=0.1875
- `iter3_rulekl0.01_l20_dclip0_dclipcoef0` seed=202604310000 pass=False top1=0.254386 eff_margin=0.426968 non_top1_pos=162 t_kl=0.0237412 t_clip=0.302632 u_clip=0.486842 delta_mean=0.366912 delta_max=1.05158 eval_fourth=0.4375 deal_in=0.125
- `iter3_rulekl0.01_l20_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0 eff_margin=0.4723 non_top1_pos=126 t_kl=0.0029989 t_clip=0.00423729 u_clip=0.127119 delta_mean=0.158715 delta_max=0.500713 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.01_l20_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.237443 eff_margin=0.299082 non_top1_pos=96 t_kl=0.00633681 t_clip=0.0776256 u_clip=0.255708 delta_mean=0.336827 delta_max=1.28968 eval_fourth=0.25 deal_in=0
- `iter3_rulekl0.01_l20_dclip1_dclipcoef0.001` seed=202604300000 pass=True top1=0.142222 eff_margin=0.39169 non_top1_pos=99 t_kl=0.0043704 t_clip=0.0533333 u_clip=0.155556 delta_mean=0.172844 delta_max=0.644013 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.01_l20_dclip1_dclipcoef0.001` seed=202604310000 pass=True top1=0.164384 eff_margin=0.392392 non_top1_pos=108 t_kl=0.00900653 t_clip=0.114155 u_clip=0.319635 delta_mean=0.25647 delta_max=1.02329 eval_fourth=0.375 deal_in=0.125
- `iter3_rulekl0.01_l20_dclip1_dclipcoef0.001` seed=202604320000 pass=False top1=0 eff_margin=0.745958 non_top1_pos=137 t_kl=0.000150211 t_clip=0 u_clip=0 delta_mean=0.0115329 delta_max=0.0261229 eval_fourth=0.25 deal_in=0.0625
- `iter3_rulekl0.01_l20_dclip1_dclipcoef0.001` seed=202604330000 pass=True top1=0.1 eff_margin=0.42184 non_top1_pos=141 t_kl=0.00917306 t_clip=0.129167 u_clip=0.204167 delta_mean=0.15812 delta_max=0.881143 eval_fourth=0.3125 deal_in=0.0625
- `iter5_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604300000 pass=False top1=0 eff_margin=0.695653 non_top1_pos=115 t_kl=0.000259997 t_clip=0 u_clip=0 delta_mean=0.0302142 delta_max=0.115364 eval_fourth=0.375 deal_in=0.125
- `iter5_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604310000 pass=False top1=0.005 eff_margin=0.525831 non_top1_pos=112 t_kl=0.00601052 t_clip=0.05 u_clip=0.09 delta_mean=0.152971 delta_max=0.639669 eval_fourth=0.375 deal_in=0.125
- `iter5_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604320000 pass=False top1=0.473029 eff_margin=0.204465 non_top1_pos=171 t_kl=0.0109214 t_clip=0.186722 u_clip=0.360996 delta_mean=0.335989 delta_max=1.57008 eval_fourth=0.4375 deal_in=0.1875
- `iter5_rulekl0.005_l20.001_dclip0_dclipcoef0` seed=202604330000 pass=True top1=0.206751 eff_margin=0.356394 non_top1_pos=155 t_kl=0.00554331 t_clip=0.092827 u_clip=0.185654 delta_mean=0.260587 delta_max=0.78068 eval_fourth=0.25 deal_in=0.0625

## Interpretation

- Best tested 3-iteration recovery remains `iter3_rulekl0.005_l20.001_dclip0_dclipcoef0`: pass 3/4 with stability and eval sanity 4/4.
- Its only miss is seed 202604320000 under-moving; increasing to 5 iterations did not fix this cleanly and introduced over-move on the same seed (`top1=0.473029`).
- `delta_clip=1.0, coef=0.001` behaves similarly to no clip and does not resolve under-movement.
- Do not proceed to paired seat-rotation strength eval yet. Use the 3-iteration L2 recovery config as the current best diagnostic candidate, and next test a smaller L2 coefficient (`0.0003`) or a seed-adaptive early-stop criterion rather than more iterations.

## Artifacts

- `step38c_recovery_summary.json`
- `step38c_recovery_summary.csv`
