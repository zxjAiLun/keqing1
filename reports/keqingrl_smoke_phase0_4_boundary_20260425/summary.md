# KeqingRL-Lite Smoke Report

seed: `20260425`
max_kyokus: `1`

## Scope

This is a smoke/non-regression report. It does not claim model strength.
Metric fidelity note: illegal/fallback/forced-terminal counters are hard failure gates until env/review expose recoverable counters.
Call/riichi rates are coarse learner-step rates in discard-only scope; opportunity-denominator metrics are a later unlock requirement.
Fixed-seed eval is seat-rotation smoke, not paired duplicate strength evaluation with confidence intervals.

## Results

- zero_delta: learner_steps=224, autopilot_steps=668, rule_kl_mean=3.01097e-09, delta_max=0
- critic_pretrain: value_loss=0.214028, rank_loss=1.3861395120620728, actor_logits_diff=0, optimizer_actor_params=0
- discard_only_ppo: iterations=3, stopped_early=False, final_delta_max=0.000350865, approx_kl_last=1.89869e-11, clip_fraction_last=0
- fixed_seed_eval: games=8, passed=True, avg_rank=2.5, rank_pt=0, fourth_rate=0.25, deal_in_rate=0
- latency: decisions=70, decisions_per_sec=46.4773, avg_ms=21.4678, p95_ms=27.525

## Artifacts

- `zero_delta_selfplay.json`
- `critic_pretrain.json`
- `discard_only_ppo.json`
- `fixed_seed_eval.json`
- `latency.json`
