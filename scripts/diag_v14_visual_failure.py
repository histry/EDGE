import numpy as np

pred = np.load("output/single_unit45_recon_norm/e1000_v14_x0w50_guidance1_rootxzref.npy").astype("float32")
gt = np.load("output/single_unit45_recon/gt_clip.npy").astype("float32")

KEYS = {0, 11, 22, 34, 44}

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

print("=== MSE by part ===")
for name, idx in parts.items():
    mse = float(np.mean((pred[:, idx] - gt[:, idx]) ** 2))
    print(f"{name:8s}: {mse:.8f}")

print("\n=== frame-to-frame jump, pred vs gt ===")
pred_jump = np.linalg.norm(np.diff(pred[:, 7:151], axis=0), axis=1)
gt_jump = np.linalg.norm(np.diff(gt[:, 7:151], axis=0), axis=1)
ratio = pred_jump / (gt_jump + 1e-8)

top = np.argsort(-ratio)[:10]
for i in top:
    print(
        f"{i:02d}->{i+1:02d} "
        f"pred_jump={pred_jump[i]:.6f} "
        f"gt_jump={gt_jump[i]:.6f} "
        f"ratio={ratio[i]:.2f} "
        f"end_is_key={(i+1) in KEYS}"
    )

print("\n=== interval MSE excluding keyframes ===")
bounds = [(0,11), (11,22), (22,34), (34,44)]
for a,b in bounds:
    frames = [i for i in range(a, b + 1) if i not in KEYS]
    if not frames:
        continue
    mse = float(np.mean((pred[frames, 7:151] - gt[frames, 7:151]) ** 2))
    print(f"{a:02d}-{b:02d} non-key rot MSE: {mse:.8f}")
