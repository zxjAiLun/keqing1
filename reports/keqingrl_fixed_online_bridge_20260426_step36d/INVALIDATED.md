# INVALIDATED: opponent step leak

This report is invalidated and must not be used as research evidence.

Reason: opponent-controlled steps leaked into the learner PPO batch before
`SeatPolicyAssignment.is_learner` became the source of truth for learner-step
filtering. As a result, the apparent fixed-to-online transfer signal in this
report can be contaminated by opponent policy steps.

Invalidated artifacts:

- `summary.md`
- `summary.csv`
- `fixed_online_bridge.json`
- `overfit_curve.csv`
- `checkpoints/*.pt`

Use a regenerated learner-only report after the `is_learner` fix instead.
