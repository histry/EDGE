import numpy as np
from pathlib import Path

pred_path = "output/single_unit45_recon_norm/e3000_normkey_normpose_v11clean.npy"
gt_path = "output/single_unit45_recon_norm/gt_clip_normalized.npy"
out_path = "output/single_unit45_recon_norm/e3000_v11clean_postproj.npy"

frames = [0, 11, 22, 34, 44]
width = 0

pred = np.load(pred_path).astype("float32")
gt = np.load(gt_path).astype("float32")

out = pred.copy()
for f in frames:
    lo = max(0, f - width)
    hi = min(len(out), f + width + 1)
    out[lo:hi] = gt[lo:hi]

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
np.save(out_path, out)

print("saved:", out_path)
print("checking keyframes after projection:")
for f in frames:
    all_mse = float(np.mean((out[f] - gt[f]) ** 2))
    rot_mse = float(np.mean((out[f, 7:151] - gt[f, 7:151]) ** 2))
    root_mse = float(np.mean((out[f, [4, 6]] - gt[f, [4, 6]]) ** 2))
    print(f"frame {f:02d}: all_mse={all_mse:.8f}, rot_mse={rot_mse:.8f}, rootxz_mse={root_mse:.8f}")

print("overall MSE vs GT_NORM:", float(np.mean((out - gt) ** 2)))
print("overall rot MSE vs GT_NORM:", float(np.mean((out[:, 7:151] - gt[:, 7:151]) ** 2)))
