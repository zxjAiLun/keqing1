# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
source_config_ids: `93`
episodes: `16`
iterations: `3`
temperatures: `1.25`
lrs: `0.0001,0.0003`
clip_eps_values: `0.1,0.2`
rule_kl_coef: `0.001`
entropy_coef: `0.005`
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`

## Results

- cfg=0 source=93 temp=1.25 lr=0.0001 clip=0.1 non_top1=102 non_top1_pos=40 top1_changed=0 t_kl=1.62585e-10 t_clip=0 u_kl=0.0457698 u_clip=1 delta_max=0.000429372 eval_fourth=0.375 deal_in=0.125
- cfg=1 source=93 temp=1.25 lr=0.0001 clip=0.2 non_top1=118 non_top1_pos=71 top1_changed=0 t_kl=1.04611e-10 t_clip=0 u_kl=0.0458329 u_clip=1 delta_max=0.000374617 eval_fourth=0.375 deal_in=0.125
- cfg=2 source=93 temp=1.25 lr=0.0003 clip=0.1 non_top1=112 non_top1_pos=82 top1_changed=0 t_kl=2.52357e-09 t_clip=0 u_kl=0.0467154 u_clip=1 delta_max=0.00298159 eval_fourth=0.375 deal_in=0.125
- cfg=3 source=93 temp=1.25 lr=0.0003 clip=0.2 non_top1=99 non_top1_pos=51 top1_changed=0 t_kl=1.21271e-09 t_clip=0 u_kl=0.0448106 u_clip=1 delta_max=0.0010236 eval_fourth=0.375 deal_in=0.125

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
