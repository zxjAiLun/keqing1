# KeqingRL Train-Time Temperature Pilot

source_type: `checkpoint`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
source_config_ids: `93`
episodes: `16`
iterations: `3`
temperatures: `1.25,1.5,2.0`
lrs: `0.0001,0.0003`
rule_kl_coef: `0.001`
entropy_coef: `0.005`
eval_seed_registry_id: `base=202604290000:stride=1:count=16`
eval_seed_hash: `be68b6700c85e610`

## Results

- cfg=0 source=93 temp=1.25 lr=0.0001 non_top1=117 non_top1_pos=106 top1_changed=0 kl=0.046625 clip=1 delta_max=0.00103132 eval_fourth=0.375 deal_in=0.1875
- cfg=1 source=93 temp=1.25 lr=0.0003 non_top1=104 non_top1_pos=73 top1_changed=0 kl=0.0433963 clip=1 delta_max=0.00151862 eval_fourth=0.375 deal_in=0.1875
- cfg=2 source=93 temp=1.5 lr=0.0001 non_top1=147 non_top1_pos=128 top1_changed=0 kl=0.128517 clip=1 delta_max=0.00170119 eval_fourth=0.375 deal_in=0.1875
- cfg=3 source=93 temp=1.5 lr=0.0003 non_top1=130 non_top1_pos=64 top1_changed=0 kl=0.128954 clip=1 delta_max=0.00244076 eval_fourth=0.375 deal_in=0.1875
- cfg=4 source=93 temp=2 lr=0.0001 non_top1=180 non_top1_pos=148 top1_changed=0 kl=0.270703 clip=1 delta_max=0.00110184 eval_fourth=0.375 deal_in=0.1875
- cfg=5 source=93 temp=2 lr=0.0003 non_top1=179 non_top1_pos=96 top1_changed=0 kl=0.271855 clip=1 delta_max=0.00226959 eval_fourth=0.375 deal_in=0.1875

## Artifacts

- `temperature_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
