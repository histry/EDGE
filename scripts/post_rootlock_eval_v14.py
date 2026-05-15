import numpy as np
from pathlib import Path

pred_path = "output/single_unit45_recon_norm/e1000_v14_x0w50_guidance1_physpose_cleanarch.npy"
gt_path = "output/single_unit45_recon/gt_clip.npy"
out_path = "output/single_unit45_recon_norm/e1000_v14_x0w50_guidance1_physpose_cleanarch_rootlock.npy"

pred = np.load(pred_path).astype("float32")
gt = np.load(gt_path).astype("float32")

out = pred.copy()
out[:, 4] = gt[:, 4]
out[:, 6] = gt[:, 6]

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
np.save(out_path, out)

def root_path(x):
    return float(np.linalg.norm(np.diff(x[:, [4, 6]], axis=0), axis=1).sum())

print("saved:", out_path)
print("MSE all:", float(np.mean((out - gt) ** 2)))
print("MSE rot:", float(np.mean((out[:, 7:151] - gt[:, 7:151]) ** 2)))
print("root path out:", root_path(out))
print("root path gt:", root_path(gt))
