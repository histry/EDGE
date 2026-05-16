# v16 single-unit final evidence

Best checkpoint:
runs/train_nextgen/strict_single_unit45_recon_v16_smooth_dense8_from_v15/weights/train-5000.pt

Key conclusions:
1. v16 reconstructs the 45-frame Dunhuang unit with very low MSE and much lower jump/jerk.
2. Keyframe-density ablation passed:
   - dense4 / dense8 / sparse3 / endpoint all produce good reconstruction.
   - endpoint-only still works, indicating the model is not merely copying dense keyframes.
3. Root-reference ablation passed:
   - removing all-frame root_xz_reference barely changes rootXZ MSE or visual quality.
   - the model has learned stable root behavior for this stationary unit.

Next:
Move from single-unit overfit to 5–10 stationary whitelist units.
