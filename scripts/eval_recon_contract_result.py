import numpy as np

pred_phys = np.load("output/single_unit45_recon_norm/e3000_recon_contract_fixed.npy").astype("float32")
pred_norm = np.load("output/single_unit45_recon_norm/e3000_recon_contract_fixed_norm.npy").astype("float32")
gt_phys = np.load("output/single_unit45_recon_norm/gt_clip_physical.npy").astype("float32")
gt_norm = np.load("output/single_unit45_recon_norm/gt_clip_normalized.npy").astype("float32")

frames = [0, 11, 22, 34, 44]

def root_path(x):
    return float(np.linalg.norm(np.diff(x[:, [4, 6]], axis=0), axis=1).sum())

print("=== overall ===")
print("norm MSE vs GT_NORM:", float(np.mean((pred_norm - gt_norm) ** 2)))
print("norm rot MSE vs GT_NORM:", float(np.mean((pred_norm[:, 7:151] - gt_norm[:, 7:151]) ** 2)))
print("phys MSE vs GT_PHYS:", float(np.mean((pred_phys - gt_phys) ** 2)))
print("phys rot MSE vs GT_PHYS:", float(np.mean((pred_phys[:, 7:151] - gt_phys[:, 7:151]) ** 2)))
print("phys root path pred:", root_path(pred_phys))
print("phys root path gt:", root_path(gt_phys))

print("\n=== keyframes normalized ===")
for f in frames:
    print(
        f"frame {f:02d}: "
        f"all={np.mean((pred_norm[f]-gt_norm[f])**2):.8f}, "
        f"rot={np.mean((pred_norm[f,7:151]-gt_norm[f,7:151])**2):.8f}, "
        f"rootxz={np.mean((pred_norm[f,[4,6]]-gt_norm[f,[4,6]])**2):.8f}"
    )

print("\n=== keyframes physical ===")
for f in frames:
    print(
        f"frame {f:02d}: "
        f"all={np.mean((pred_phys[f]-gt_phys[f])**2):.8f}, "
        f"rot={np.mean((pred_phys[f,7:151]-gt_phys[f,7:151])**2):.8f}, "
        f"rootxz={np.mean((pred_phys[f,[4,6]]-gt_phys[f,[4,6]])**2):.8f}"
    )
