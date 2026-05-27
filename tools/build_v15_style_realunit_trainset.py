import argparse
import json
import pickle
from pathlib import Path

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)

def find_motion_array(npz):
    candidates = []
    for k in npz.files:
        arr = npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 151:
            candidates.append((k, arr.shape, arr))
    if not candidates:
        raise RuntimeError(
            "No [N,T,151] motion array found in npz. Keys="
            + ", ".join(npz.files)
        )
    # Prefer physical/unit motion names if present
    priority = [
        "unit_motions_physical",
        "motions_physical",
        "unit_motions",
        "motions",
        "motion",
        "poses",
    ]
    for name in priority:
        for k, shape, arr in candidates:
            if k == name:
                return k, arr
    # Otherwise choose the largest candidate
    candidates.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)
    return candidates[0][0], candidates[0][2]

def crop_or_resample(m, target_len):
    m = np.asarray(m, dtype=np.float32)
    if len(m) == target_len:
        return m
    if len(m) > target_len:
        start = max(0, (len(m) - target_len) // 2)
        return m[start:start + target_len]
    # pad short clips
    pad = np.repeat(m[-1:], target_len - len(m), axis=0)
    return np.concatenate([m, pad], axis=0)

def metrics(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT]
    droot = np.linalg.norm(root[1:] - root[:-1], axis=1)
    drot = np.linalg.norm(rot[1:] - rot[:-1], axis=1)

    # Approximate body sections in 24 joints * 6D.
    # We do not assume exact SMPL names here; this is a robust proxy.
    rot_j = rot.reshape(len(rot), 24, 6)
    lower = rot_j[:, 0:8].reshape(len(rot), -1)
    torso = rot_j[:, 8:14].reshape(len(rot), -1)
    upper = rot_j[:, 14:24].reshape(len(rot), -1)

    def act(x):
        if len(x) <= 1:
            return 0.0
        return float(np.linalg.norm(x[1:] - x[:-1], axis=1).mean())

    out = {
        "root_radius": float(np.linalg.norm(root - root[:1], axis=1).max()),
        "root_final": float(np.linalg.norm(root[-1] - root[0])),
        "root_speed_mean": float(droot.mean()) if len(droot) else 0.0,
        "global_rot_jump_p95": float(np.percentile(drot, 95)) if len(drot) else 0.0,
        "global_rot_jump_max": float(drot.max()) if len(drot) else 0.0,
        "activity": float(drot.mean()) if len(drot) else 0.0,
        "lower_activity": act(lower),
        "torso_activity": act(torso),
        "upper_activity": act(upper),
        "contact_switch": float(np.abs(m[1:, CONTACT] - m[:-1, CONTACT]).sum(axis=1).mean()) if len(m) > 1 else 0.0,
    }
    return out

def score_unit(met, args):
    # Hard reject bad / static / too mobile units.
    if met["root_radius"] > args.max_root_radius:
        return None
    if met["global_rot_jump_p95"] > args.max_rot_jump_p95:
        return None
    if met["activity"] < args.min_activity:
        return None
    if met["upper_activity"] < args.min_upper_activity:
        return None
    if met["torso_activity"] < args.min_torso_activity:
        return None

    # Style-preserving score:
    # prefer upper/torso expression, stable root, low jump, non-static motion.
    expr = 1.40 * met["upper_activity"] + 1.00 * met["torso_activity"] + 0.35 * met["lower_activity"]
    stable = np.exp(-4.0 * met["root_radius"])
    jump_safe = np.exp(-1.5 * met["global_rot_jump_p95"])
    not_static = min(met["activity"] / 0.18, 1.0)

    return float(expr * stable * jump_safe * (0.35 + 0.65 * not_static))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_len", type=int, default=45)
    ap.add_argument("--top_k", type=int, default=1200)
    ap.add_argument("--max_root_radius", type=float, default=0.20)
    ap.add_argument("--max_rot_jump_p95", type=float, default=0.95)
    ap.add_argument("--min_activity", type=float, default=0.035)
    ap.add_argument("--min_upper_activity", type=float, default=0.025)
    ap.add_argument("--min_torso_activity", type=float, default=0.010)
    ap.add_argument("--audio_dim", type=int, default=803)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz, allow_pickle=True)
    motion_key, motions = find_motion_array(npz)
    print(f"motion_key={motion_key}, shape={motions.shape}")

    rows = []
    for i in range(motions.shape[0]):
        m = crop_or_resample(motions[i], args.target_len)
        met = metrics(m)
        score = score_unit(met, args)
        if score is None:
            continue
        rows.append((score, i, m, met))

    rows.sort(key=lambda x: x[0], reverse=True)
    selected = rows[: args.top_k]

    pkl_dir = out_dir / "dunhuang_style_realunit_pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for rank, (score, idx, m, met) in enumerate(selected, start=1):
        item = {
            "motion": m.astype(np.float32),
            "motion_151": m.astype(np.float32),
            "poses": m.astype(np.float32),
            "audio_feature": np.zeros((len(m), args.audio_dim), dtype=np.float32),
            "original_filename": f"style_realunit_idx{idx:06d}",
            "source_file": f"style_realunit_idx{idx:06d}",
            "metadata": {
                "source_npz": str(args.npz),
                "source_index": int(idx),
                "rank": int(rank),
                "score": float(score),
                **met,
            },
        }
        out_pkl = pkl_dir / f"style_realunit_rank{rank:04d}_idx{idx:06d}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(item, f)

        report_rows.append({
            "rank": rank,
            "source_index": int(idx),
            "score": float(score),
            "pkl": str(out_pkl),
            **met,
        })

    report = {
        "npz": str(args.npz),
        "motion_key": motion_key,
        "input_shape": list(motions.shape),
        "target_len": args.target_len,
        "selected": len(selected),
        "thresholds": {
            "max_root_radius": args.max_root_radius,
            "max_rot_jump_p95": args.max_rot_jump_p95,
            "min_activity": args.min_activity,
            "min_upper_activity": args.min_upper_activity,
            "min_torso_activity": args.min_torso_activity,
        },
        "items": report_rows,
    }

    report_path = out_dir / "style_realunit_selection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected={len(selected)}")
    print(f"pkl_dir={pkl_dir}")
    print(f"report={report_path}")

if __name__ == "__main__":
    main()
