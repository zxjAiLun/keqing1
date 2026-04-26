# KeqingRL Fixed-Batch To Online Bridge

source_type: `fixed_batch_overfit_bridge`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
fixed_episodes: `16`
online_episodes: `16`
fixed_seed_registry_id: `base=202604260000:stride=1:count=16`
online_seed_registry_id: `base=202604270000:stride=1:count=16`

## Results

- source=93 rerun=0 overfit_pass=True overfit_epochs=74 fixed_before=0 fixed_after=0.0120219 online_pre=0.0285401 online_post=0.0285401 fixed_post_online=0.0174863 online_delta_max=7.40011 checkpoint=reports/keqingrl_fixed_online_bridge_20260426_step36d/checkpoints/source_93_rerun_0_fixed_batch_overfit.pt post_online_checkpoint=reports/keqingrl_fixed_online_bridge_20260426_step36d/checkpoints/source_93_rerun_0_post_online.pt
- source=57 rerun=1 overfit_pass=True overfit_epochs=56 fixed_before=0 fixed_after=0.0107991 online_pre=0 online_post=0 fixed_post_online=0 online_delta_max=1.02426 checkpoint=reports/keqingrl_fixed_online_bridge_20260426_step36d/checkpoints/source_57_rerun_1_fixed_batch_overfit.pt post_online_checkpoint=reports/keqingrl_fixed_online_bridge_20260426_step36d/checkpoints/source_57_rerun_1_post_online.pt
- source=8 rerun=3 overfit_pass=True overfit_epochs=64 fixed_before=0 fixed_after=0.0110619 online_pre=0 online_post=0 fixed_post_online=0.00995575 online_delta_max=1.97946 checkpoint=reports/keqingrl_fixed_online_bridge_20260426_step36d/checkpoints/source_8_rerun_3_fixed_batch_overfit.pt post_online_checkpoint=reports/keqingrl_fixed_online_bridge_20260426_step36d/checkpoints/source_8_rerun_3_post_online.pt

## Artifacts

- `fixed_online_bridge.json`
- `summary.csv`
- `overfit_curve.csv`
- `checkpoints/*.pt`
