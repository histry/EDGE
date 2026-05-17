#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
        return arr
    raise ValueError(f"Expected [T,151] or [B,T,151], got {arr.shape} from {path}")

def fk_positions(motion_batch: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(motion_batch, dtype=torch.float32, device=device)
    b, t, _ = x.shape
    pos = x[:, :, 4:7]
    q6 = x[:, :, 7:].reshape(b, t, 24, 6)
    qax = ax_from_6v(q6)
    smpl = SMPLSkeleton(device=device)
    return smpl.forward(qax, pos).detach().cpu().numpy()

def stats_for_batch(motion_batch: np.ndarray) -> dict:
    pos = fk_positions(motion_batch)
    groups = {
        "root": [0],
        "torso": [3, 6, 9, 12, 13, 14, 15],
        "arms": [16, 17, 18, 19, 20, 21, 22, 23],
        "wrists_hands": [20, 21, 22, 23],
        "upper_safe_plus": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    }
    out = {}
    for name, idxs in groups.items():
        p = pos[:, :, idxs, :]
        v = np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1)
        r = np.linalg.norm(p.max(axis=1) - p.min(axis=1), axis=-1)
        out[f"{name}_range_mean"] = float(r.mean())
        out[f"{name}_range_median"] = float(np.median(r))
        out[f"{name}_speed_mean"] = float(v.mean())
        out[f"{name}_speed95"] = float(np.percentile(v, 95))
    return out

def collect_gt(gt_dir: Path) -> np.ndarray:
    motions = []
    for p in sorted(gt_dir.glob("*.pkl")):
        arr = load_motion(p)
        motions.append(arr[0, :45])
    if not motions:
        raise ValueError(f"No GT pkl under {gt_dir}")
    return np.stack(motions, axis=0)

def print_block(title, stats):
    print(f"\n==== {title} ====")
    for k in sorted(stats):
        print(f"{k}: {stats[k]:.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt_dir", default="")
    args = ap.parse_args()
    pred = load_motion(Path(args.pred))
    print("pred_shape:", pred.shape)
    pred_stats = stats_for_batch(pred)
    print_block("PRED FK-visible stats", pred_stats)
    if args.gt_dir:
        gt = collect_gt(Path(args.gt_dir))
        print("gt_shape:", gt.shape)
        gt_stats = stats_for_batch(gt)
        print_block("GT FK-visible stats", gt_stats)
        print("\n==== PRED / GT ratios ====")
        for key in [
            "wrists_hands_range_mean",
            "wrists_hands_speed_mean",
            "arms_range_mean",
            "arms_speed_mean",
            "upper_safe_plus_range_mean",
            "upper_safe_plus_speed_mean",
        ]:
            print(f"{key}_ratio: {pred_stats[key] / max(gt_stats[key], 1e-8):.4f}")

if __name__ == "__main__":
    main()
