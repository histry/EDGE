import numpy as np

path = "output/single_unit45_recon_norm/e10000_v15_bridge_dense4_strict_single_unit45_recon_v15_bridge_dense4_bodyrot_10000steps_b1.npy"
pred = np.load(path).astype("float32")
gt = np.load("output/single_unit45_recon/gt_clip.npy").astype("float32")

KEYS = {0,4,8,12,16,20,24,28,32,36,40,44}

def root_path(x):
    return float(np.linalg.norm(np.diff(x[:, [4, 6]], axis=0), axis=1).sum())

def rot_idx(joints):
    out = []
    for j in joints:
        s = 7 + 6 * j
        out.extend(range(s, s + 6))
    return out

parts = {
    "all_rot": list(range(7, 151)),
    "pelvis": rot_idx([0]),
    "lower": rot_idx([1,2,4,5,7,8,10,11]),
    "torso": rot_idx([3,6,9]),
    "upper": rot_idx([12,13,14,15,16,17,18,19,20,21,22,23]),
}

print("path:", path)
print("phys MSE:", float(np.mean((pred - gt) ** 2)))
print("phys rot MSE:", float(np.mean((pred[:, 7:151] - gt[:, 7:151]) ** 2)))
print("phys rootXZ MSE:", float(np.mean((pred[:, [4, 6]] - gt[:, [4, 6]]) ** 2)))
print("root path pred:", root_path(pred))
print("root path gt:", root_path(gt))

print("\n=== MSE by part ===")
for name, idx in parts.items():
    print(f"{name:8s}: {float(np.mean((pred[:, idx] - gt[:, idx]) ** 2)):.8f}")

print("\n=== frame-to-frame jump, pred vs gt ===")
pred_jump = np.linalg.norm(np.diff(pred[:, 7:151], axis=0), axis=1)
gt_jump = np.linalg.norm(np.diff(gt[:, 7:151], axis=0), axis=1)
ratio = pred_jump / (gt_jump + 1e-8)

top = np.argsort(-ratio)[:15]
for i in top:
    print(
        f"{i:02d}->{i+1:02d} "
        f"pred_jump={pred_jump[i]:.6f} "
        f"gt_jump={gt_jump[i]:.6f} "
        f"ratio={ratio[i]:.2f} "
        f"end_is_key={(i+1) in KEYS}"
    )

print("\n=== keyframes ===")
for f in sorted(KEYS):
    all_mse = float(np.mean((pred[f] - gt[f]) ** 2))
    rot_mse = float(np.mean((pred[f, 7:151] - gt[f, 7:151]) ** 2))
    root_mse = float(np.mean((pred[f, [4,6]] - gt[f, [4,6]]) ** 2))
    print(f"frame {f:02d}: all={all_mse:.8f}, rot={rot_mse:.8f}, rootxz={root_mse:.8f}")
