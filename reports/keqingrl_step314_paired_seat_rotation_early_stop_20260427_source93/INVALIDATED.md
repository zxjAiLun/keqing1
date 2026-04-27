# Invalidated

This first Step 3.14 run loaded saved candidates as raw `RulePriorDeltaPolicy`
instances and did not reapply the checkpoint `support-only-topK=3` policy
contract. The movement-quality diagnostics from this directory therefore
measure the unwrapped delta model, not the diagnostic candidate policy.

Use the wrapper-fixed rerun instead:

`reports/keqingrl_step314_paired_seat_rotation_early_stop_20260427_source93_wrapper_fixed/`
