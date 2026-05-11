import glob, os
import numpy as np
from postprocess_footlock import load_motion, motion_to_joints

LOWER_ROT = []
for j in [1,2,4,5,7,8,10,11]:
    LOWER_ROT.extend(range(7 + 6*j, 7 + 6*j + 6))

UPPER_ROT = []
for j in [12,13,14,15,16,17,18,19,20,21,22,23]:
    UPPER_ROT.extend(range(7 + 6*j, 7 + 6*j + 6))

print("name\troot_speed\tfoot_speed\tlower_rot\tupper_rot\tfoot/root")
for path in sorted(glob.glob("output/decoder_adapter_ckpt_eval/dhw4_decoder_e*_hd6000_upper_lower*.npy")):
    if path.endswith("_target_traj.npy") or path.endswith("_unit.npy") or "_mid" in path:
        continue

    name = os.path.basename(path).replace(".npy", "")
    m = load_motion(path)

    root = m[:, [4,6]]
    root_speed = np.linalg.norm(root[1:] - root[:-1], axis=1).mean()

    lower = np.sqrt(np.mean((m[1:, LOWER_ROT] - m[:-1, LOWER_ROT]) ** 2))
    upper = np.sqrt(np.mean((m[1:, UPPER_ROT] - m[:-1, UPPER_ROT]) ** 2))

    joints = motion_to_joints(m, device="cpu")
    feet = joints[:, [7,8,10,11], :][:, :, [0,2]]
    foot_speed = np.linalg.norm(feet[1:] - feet[:-1], axis=-1).mean()

    ratio = foot_speed / max(root_speed, 1e-8)

    print(f"{name}\t{root_speed:.6f}\t{foot_speed:.6f}\t{lower:.6f}\t{upper:.6f}\t{ratio:.3f}")
