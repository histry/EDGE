import numpy as np

pred = np.load("output/single_unit45_recon_norm/e1000_v14_x0w50_guidance1_physpose_cleanarch_v2.npy").astype("float32")
gt = np.load("output/single_unit45_recon/gt_clip.npy").astype("float32")

frames = [0, 11, 22, 34, 44]

def root_path(x):
    return float(np.linalg.norm(np.diff(x[:, [4, 6]], axis=0), axis=1).sum())

print("phys MSE:", float(np.mean((pred - gt) ** 2)))
print("phys rot MSE:", float(np.mean((pred[:, 7:151] - gt[:, 7:151]) ** 2)))
print("phys rootXZ MSE:", float(np.mean((pred[:, [4, 6]] - gt[:, [4, 6]]) ** 2)))
print("root path pred:", root_path(pred))
print("root path gt:", root_path(gt))

print("\nkeyframes:")
for f in frames:
    print(
        f"frame {f:02d}: "
        f"all={np.mean((pred[f]-gt[f])**2):.8f}, "
        f"rot={np.mean((pred[f,7:151]-gt[f,7:151])**2):.8f}, "
        f"rootxz={np.mean((pred[f,[4,6]]-gt[f,[4,6]])**2):.8f}"
    )
