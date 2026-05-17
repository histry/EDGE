#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import numpy as np


def rot_dims(joints):
    out = []
    for j in joints:
        out.extend(range(7 + 6 * j, 7 + 6 * (j + 1)))
    return out


# 151D layout:
# 0:4 contacts
# 4:7 root position
# 7:151 24 joints * 6D rotation

PELVIS = rot_dims([0])
HIPS = rot_dims([1, 2])
SPINE_TORSO = rot_dims([3, 6, 9])
NECK_HEAD = rot_dims([12, 15])
SHOULDERS_ARMS_HANDS = rot_dims([13, 14, 16, 17, 18, 19, 20, 21, 22, 23])

UPPER_SAFE_PLUS = SPINE_TORSO + NECK_HEAD + SHOULDERS_ARMS_HANDS


def load_motion(path):
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    if path.suffix == ".pkl":
        d = pickle.load(open(path, "rb"))
        for k in ["motion", "motion_151", "poses", "unit_motions_physical"]:
            if k in d:
                arr = np.asarray(d[k], dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                return arr

    raise ValueError(f"Cannot load motion from {path}")


def blend(dst, lower, upper, dims, alpha):
    if not dims or alpha <= 0:
        return
    if alpha >= 1:
        dst[:, dims] = upper[:, dims]
    else:
        dst[:, dims] = (1.0 - alpha) * lower[:, dims] + alpha * upper[:, dims]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lower", required=True, help="base motion: stable root/contact/lower from generated sample")
    ap.add_argument("--upper", required=True, help="GT/retrieved motion unit")
    ap.add_argument("--out", required=True)

    ap.add_argument("--pelvis_blend", type=float, default=0.25)
    ap.add_argument("--hip_blend", type=float, default=0.15)
    ap.add_argument("--torso_blend", type=float, default=0.65)
    ap.add_argument("--neck_head_blend", type=float, default=0.90)
    ap.add_argument("--arms_blend", type=float, default=1.00)

    args = ap.parse_args()

    lower = load_motion(args.lower)[:45].copy()
    upper = load_motion(args.upper)[:45].copy()

    if lower.shape != upper.shape or lower.shape[-1] != 151:
        raise ValueError(f"shape mismatch lower={lower.shape}, upper={upper.shape}")

    merged = lower.copy()

    # 保留 contacts + root position + lower legs/feet，防止脚底和根节点被 GT 硬替换。
    blend(merged, lower, upper, PELVIS, args.pelvis_blend)
    blend(merged, lower, upper, HIPS, args.hip_blend)
    blend(merged, lower, upper, SPINE_TORSO, args.torso_blend)
    blend(merged, lower, upper, NECK_HEAD, args.neck_head_blend)
    blend(merged, lower, upper, SHOULDERS_ARMS_HANDS, args.arms_blend)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, merged.astype(np.float32))

    print("saved:", args.out, merged.shape)
    print("pelvis_blend:", args.pelvis_blend)
    print("hip_blend:", args.hip_blend)
    print("torso_blend:", args.torso_blend)
    print("neck_head_blend:", args.neck_head_blend)
    print("arms_blend:", args.arms_blend)
    print("upper_safe_plus_dims:", len(UPPER_SAFE_PLUS))


if __name__ == "__main__":
    main()
