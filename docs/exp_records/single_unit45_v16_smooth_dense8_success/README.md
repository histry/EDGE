# v16 smooth dense8 single-unit reconstruction success

Best checkpoint:
runs/train_nextgen/strict_single_unit45_recon_v16_smooth_dense8_from_v15/weights/train-5000.pt

Main result:
- phys_mse = 0.00011363
- rootxz_mse = 0.0
- jump_ratio_p95 = 4.097
- jerk_ratio_p95 = 7.873
- upper_mse = 0.00010069
- lower_mse = 0.00020452

Conclusion:
v16 significantly improves smoothness over v15 dense8 while preserving strict reconstruction quality.
This checkpoint is the current best single-unit reconstruction model.

Next:
1. Test sparse3 / endpoint with v16 e5000.
2. Test without all-frame root_xz_reference.
3. Expand to 5–10 stationary whitelist units.
