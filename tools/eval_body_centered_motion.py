#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from vis import SMPLSkeleton
from dataset.quaternion import ax_from_6v


def load_motion(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True).astype(np.float32)
    elif path.suffix == ".pkl":
        data = pickle.load(open(path, "rb"))
        arr = None
        for k in ["motion", "motion_151", "poses", "unit_motions_physical"]:
            if k in data:
                arr = np.asarray(data[k], dtype=np.float32)
                break
        if arr is None:
            raise ValueError(f"No 151D motion key found in {path}")
    else:
        raise ValueError(f"Unsupported file: {path}")

    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim == 3 and arr.shape[-1] == 151:
        return arr[:, :45]
    raise ValueError(f"Expected [T,151] or [B,T,151], got {arr.shape} from {path}")


def collect_gt(gt_dir: Path) -> np.ndarray:
    motions = []
    for p in sorted(gt_dir.glob("*.pkl")):
        motions.append(load_motion(p)[0])
    if not motions:
        raise ValueError(f"No pkl found in {gt_dir}")
    return np.stack(motions, axis=0)


def fk_positions(motion_batch: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(motion_batch, dtype=torch.float32, device=device)
    b, t, _ = x.shape
    pos = x[:, :, 4:7]
    q6 = x[:, :, 7:].reshape(b, t, 24, 6)
    qax = ax_from_6v(q6)
    smpl = SMPLSkeleton(device=device)
    return smpl.forward(qax, pos).detach().cpu().numpy()


def group_stats(pos: np.ndarray) -> dict:
    # pos: [B,T,24,3]
    bc = pos - pos[:, :, 0:1, :]

    groups = {
        "root_world": [0],
        "torso_bc": [3, 6, 9, 12, 15],
        "arms_bc": [16, 17, 18, 19, 20, 21, 22, 23],
        "hands_bc": [20, 21, 22, 23],
        "upper_bc": [3, 6, 9, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    }

    out = {}

    root_xz = pos[:, :, 0, :][:, :, [0, 2]]
    root_v = np.linalg.norm(root_xz[:, 1:] - root_xz[:, :-1], axis=-1)
    out["root_xz_range_mean"] = float(np.linalg.norm(root_xz.max(axis=1) - root_xz.min(axis=1), axis=-1).mean())
    out["root_xz_speed_mean"] = float(root_v.mean())

    for name, idxs in groups.items():
        p = pos[:, :, idxs, :] if name == "root_world" else bc[:, :, idxs, :]
        v = np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1)
        r = np.linalg.norm(p.max(axis=1) - p.min(axis=1), axis=-1)
        out[f"{name}_range_mean"] = float(r.mean())
        out[f"{name}_speed_mean"] = float(v.mean())

        if p.shape[1] >= 4:
            vel = p[:, 1:] - p[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]
            jerk = acc[:, 1:] - acc[:, :-1]
            out[f"{name}_jerk_mean"] = float(np.linalg.norm(jerk, axis=-1).mean())

    # coupling ratio
    torso_r = max(out["torso_bc_range_mean"], 1e-8)
    arms_r = max(out["arms_bc_range_mean"], 1e-8)
    hands_r = max(out["hands_bc_range_mean"], 1e-8)
    out["torso_to_arms_range_ratio"] = float(torso_r / arms_r)
    out["torso_to_hands_range_ratio"] = float(torso_r / hands_r)

    return out


def print_block(title, d):
    print(f"\n==== {title} ====")
    for k in sorted(d):
        print(f"{k}: {d[k]:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt_dir", default="")
    args = ap.parse_args()

    pred = load_motion(Path(args.pred))
    pred_stats = group_stats(fk_positions(pred))
    print("pred_shape:", pred.shape)
    print_block("PRED body-centered stats", pred_stats)

    if args.gt_dir:
        gt = collect_gt(Path(args.gt_dir))
        gt_stats = group_stats(fk_positions(gt))
        print("gt_shape:", gt.shape)
        print_block("GT body-centered stats", gt_stats)

        print("\n==== PRED / GT ratios ====")
        for k in sorted(pred_stats):
            if k in gt_stats:
                print(f"{k}_ratio: {pred_stats[k] / max(gt_stats[k], 1e-8):.4f}")


if __name__ == "__main__":
    main()
