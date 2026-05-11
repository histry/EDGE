import os, glob, json
import numpy as np
import matplotlib.pyplot as plt

ROOT_X_IDX = 4
ROOT_Z_IDX = 6

pairs = [
    ("final_dhw2_s_main_p003", "S main"),
    ("final_dhw3_s_main_p003", "S main"),
    ("final_dhw4_s_main_p003", "S main"),
    ("final_dhw2_wide_main_p003", "wide raw-final"),
    ("final_dhw2_wide_repaired_static055", "wide repaired"),
    ("final_dhw3_wide_repaired_static055", "wide repaired"),
    ("final_dhw4_wide_repaired_static055", "wide repaired"),
]

out_dir = "output/final_evidence_package/figures"
os.makedirs(out_dir, exist_ok=True)

for name, label in pairs:
    motion_path = f"output/final_evidence_package/{name}.npy"

    if "wide_repaired" in name:
        base = name.replace("_wide_repaired_static055", "_wide_main_p003")
        traj_path = f"output/final_evidence_package/{base}_target_traj.npy"
    else:
        traj_path = f"output/final_evidence_package/{name}_target_traj.npy"

    if not os.path.exists(motion_path) or not os.path.exists(traj_path):
        print("skip", name)
        continue

    motion = np.load(motion_path, allow_pickle=True)
    if motion.ndim == 0 and isinstance(motion.item(), dict):
        motion = motion.item().get("motion", motion.item().get("motion_final"))
    if motion.ndim == 3:
        motion = motion[0]

    traj = np.load(traj_path, allow_pickle=True)
    if traj.ndim == 3:
        traj = traj[0]
    traj = traj[:, :2]

    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]

    plt.figure(figsize=(5, 5))
    plt.plot(traj[:, 0], traj[:, 1], linestyle="--", label="target")
    plt.plot(root[:, 0], root[:, 1], label="generated")
    plt.scatter([traj[0,0]], [traj[0,1]], marker="o", label="start")
    plt.scatter([traj[-1,0]], [traj[-1,1]], marker="x", label="end")
    plt.title(f"{name} ({label})")
    plt.xlabel("root X")
    plt.ylabel("root Z")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    out = f"{out_dir}/{name}_trajectory.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print("saved", out)
