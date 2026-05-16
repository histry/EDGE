import argparse
import json
import numpy as np
from pathlib import Path

ROOT_X_IDX = 4
ROOT_Z_IDX = 6
ROT_SLICE = slice(7, 151)

LOWER_JOINTS = [1,2,4,5,7,8,10,11]
TORSO_JOINTS = [3,6,9]
UPPER_JOINTS = [12,13,14,15,16,17,18,19,20,21,22,23]

def rot_indices(joints):
    idx = []
    for j in joints:
        idx.extend(range(7 + 6*j, 7 + 6*(j+1)))
    return np.array(idx, dtype=np.int64)

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def jump_stats(pred, gt):
    pj = np.linalg.norm(pred[1:, ROT_SLICE] - pred[:-1, ROT_SLICE], axis=1)
    gj = np.linalg.norm(gt[1:, ROT_SLICE] - gt[:-1, ROT_SLICE], axis=1)
    ratio = pj / np.maximum(gj, 1e-8)
    return {
        "pred_jump_mean": float(pj.mean()),
        "gt_jump_mean": float(gj.mean()),
        "jump_ratio_mean": float(ratio.mean()),
        "jump_ratio_max": float(ratio.max()),
        "jump_ratio_p95": float(np.percentile(ratio, 95)),
    }

def jerk_stats(pred, gt):
    if len(pred) < 3:
        return {}
    pa = pred[2:, ROT_SLICE] - 2 * pred[1:-1, ROT_SLICE] + pred[:-2, ROT_SLICE]
    ga = gt[2:, ROT_SLICE] - 2 * gt[1:-1, ROT_SLICE] + gt[:-2, ROT_SLICE]
    pj = np.linalg.norm(pa, axis=1)
    gj = np.linalg.norm(ga, axis=1)
    ratio = pj / np.maximum(gj, 1e-8)
    return {
        "pred_jerk_mean": float(pj.mean()),
        "gt_jerk_mean": float(gj.mean()),
        "jerk_ratio_mean": float(ratio.mean()),
        "jerk_ratio_max": float(ratio.max()),
        "jerk_ratio_p95": float(np.percentile(ratio, 95)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = np.load(args.pred).astype(np.float32)
    gt = np.load(args.gt).astype(np.float32)

    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    lower = rot_indices(LOWER_JOINTS)
    torso = rot_indices(TORSO_JOINTS)
    upper = rot_indices(UPPER_JOINTS)

    report = {
        "pred": args.pred,
        "gt": args.gt,
        "frames": int(n),
        "phys_mse": mse(pred, gt),
        "rot_mse": mse(pred[:, ROT_SLICE], gt[:, ROT_SLICE]),
        "rootxz_mse": mse(pred[:, [ROOT_X_IDX, ROOT_Z_IDX]], gt[:, [ROOT_X_IDX, ROOT_Z_IDX]]),
        "lower_mse": mse(pred[:, lower], gt[:, lower]),
        "torso_mse": mse(pred[:, torso], gt[:, torso]),
        "upper_mse": mse(pred[:, upper], gt[:, upper]),
    }
    report.update(jump_stats(pred, gt))
    report.update(jerk_stats(pred, gt))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
