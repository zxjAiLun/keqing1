# Step 3.14 Result Interpretation

Run count: 768 half-hands.

Elapsed: 1728 seconds.

This run is a paired seat-rotation sanity/strength-proxy evaluation from the
Step 3.13C diagnostic gate candidates. It is not strength evidence.

The eval loader reapplied the checkpoint `support-only-topK=3` policy contract.
The earlier unwrapped run is invalidated.

## Result

The support-only topK direction guard holds on held-out paired diagnostics:

- `rank_ge5 = 0` for all three candidates.
- `changed_action_prior_rank_mean = 2.465 / 2.526 / 2.667`.

The rank proxy does not show a positive strength signal:

- seed `202604360000`: `delta_vs_zero = 0`
- seed `202604370000`: `delta_vs_zero = -0.0234375`
- seed `202604340000`: `delta_vs_zero = -0.0703125`

Baseline was `rank_pt = 0`, `fourth_rate = 0.25`,
`learner_deal_in_rate = 0.1171875`.

## Conclusion

Step 3.14 passes the structural movement-quality check but not the paired
strength-proxy check. Do not proceed to strength claims or long training from
these checkpoints. The next work should improve topK teacher/reranker signal
quality rather than relax the support guard.
