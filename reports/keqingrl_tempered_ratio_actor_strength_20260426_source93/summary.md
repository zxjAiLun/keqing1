# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
source_config_ids: `93`
episodes: `16`
iterations: `3`
rule_score_scales: `1.0`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
temperatures: `1.25`
lrs: `0.001,0.003`
update_epochs_values: `4,8`
clip_eps_values: `0.2,0.3`
rule_kl_coef: `0.001`
entropy_coef: `0.005`
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=1 temp=1.25 lr=0.001 epochs=4 clip=0.2 non_top1=103 non_top1_pos=41 top1_changed=0 t_kl=9.52121e-06 t_clip=0 u_kl=0.0462553 u_clip=1 delta_max=0.0389973 eval_fourth=0.375 deal_in=0.125
- cfg=1 source=93 scale=1 temp=1.25 lr=0.001 epochs=4 clip=0.3 non_top1=118 non_top1_pos=71 top1_changed=0 t_kl=9.22598e-06 t_clip=0 u_kl=0.0459873 u_clip=0.331967 delta_max=0.0588579 eval_fourth=0.375 deal_in=0.125
- cfg=2 source=93 scale=1 temp=1.25 lr=0.001 epochs=8 clip=0.2 non_top1=114 non_top1_pos=82 top1_changed=0 t_kl=0.0172974 t_clip=0.234234 u_kl=0.060774 u_clip=0.617117 delta_max=1.79803 eval_fourth=0.375 deal_in=0.125
- cfg=3 source=93 scale=1 temp=1.25 lr=0.001 epochs=8 clip=0.3 non_top1=100 non_top1_pos=50 top1_changed=0 t_kl=0.00173112 t_clip=0 u_kl=0.0487322 u_clip=0.429224 delta_max=0.568274 eval_fourth=0.375 deal_in=0.125
- cfg=4 source=93 scale=1 temp=1.25 lr=0.003 epochs=4 clip=0.2 non_top1=119 non_top1_pos=83 top1_changed=0 t_kl=0.00149212 t_clip=0.0172414 u_kl=0.0508264 u_clip=0.956897 delta_max=0.521267 eval_fourth=0.375 deal_in=0.125
- cfg=5 source=93 scale=1 temp=1.25 lr=0.003 epochs=4 clip=0.3 non_top1=103 non_top1_pos=58 top1_changed=0 t_kl=0.0259358 t_clip=0.162996 u_kl=0.0839889 u_clip=0.431718 delta_max=1.92301 eval_fourth=0.375 deal_in=0.125
- cfg=6 source=93 scale=1 temp=1.25 lr=0.003 epochs=8 clip=0.2 non_top1=120 non_top1_pos=105 top1_changed=0 t_kl=0.0158764 t_clip=0.237668 u_kl=0.0698992 u_clip=0.717489 delta_max=2.12598 eval_fourth=0.375 deal_in=0.125
- cfg=7 source=93 scale=1 temp=1.25 lr=0.003 epochs=8 clip=0.3 non_top1=126 non_top1_pos=96 top1_changed=0 t_kl=0.0255187 t_clip=0.165179 u_kl=0.0737016 u_clip=0.459821 delta_max=1.05174 eval_fourth=0.375 deal_in=0.125

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
