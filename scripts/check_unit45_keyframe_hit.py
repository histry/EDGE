import numpy as np

pred = np.load("output/single_unit45_recon_norm/e3000_normkey_normpose_v11clean.npy").astype("float32")
gt = np.load("output/single_unit45_recon_norm/gt_clip_normalized.npy").astype("float32")

frames = [0, 11, 22, 34, 44]

print("pred shape:", pred.shape)
print("gt shape:", gt.shape)

for f in frames:
    all_mse = float(np.mean((pred[f] - gt[f]) ** 2))
    rot_mse = float(np.mean((pred[f, 7:151] - gt[f, 7:151]) ** 2))
    root_mse = float(np.mean((pred[f, [4, 6]] - gt[f, [4, 6]]) ** 2))
    print(f"frame {f:02d}: all_mse={all_mse:.6f}, rot_mse={rot_mse:.6f}, rootxz_mse={root_mse:.6f}")
