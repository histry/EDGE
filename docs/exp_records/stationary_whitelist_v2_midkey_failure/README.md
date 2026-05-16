# stationary_whitelist_v2 mid-keyframe failure

Observation:
Most generated videos show step-wise / stuttering pose changes.

Diagnosis:
A/B/C ablation on unit_49122 using train-1800.pt:
- A_mid_hard: start + 3 mid + end + hard projection
  jump_p95 ≈ 83.28, jerk_p95 ≈ 216.73
- B_endpoint_hard: start + end + hard projection
  jump_p95 ≈ 1.58, jerk_p95 ≈ 3.10
- C_endpoint_soft: start + end, no hard projection
  jump_p95 ≈ 1.08, jerk_p95 ≈ 3.38

Conclusion:
The stuttering is mainly induced by mid-keyframe attraction/projection.
Next:
Train v2b with no mid keyframes or very low mid-keyframe probability, lower keyframe loss, and evaluate endpoint-only first.
