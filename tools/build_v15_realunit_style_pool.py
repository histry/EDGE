import argparse
import json
import pickle
from pathlib import Path

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Z = 6
ROT = slice(7, 151)

def as_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def find_motion_array(npz):
    candidates = []
    for k in npz.files:
        arr = npz[k]
        if not isinstance(arr, np.ndarray):
            continue

        if arr.ndim == 3 and arr.shape[-1] == 151:
            candidates.append((k, arr))
        elif arr.ndim == 2 and arr.shape[-1] == 151:
            # [N,151] single-frame fallback, less preferred
            candidates.append((k, arr[:, None, :]))

    if not candidates:
        raise RuntimeError("No motion array [N,T,151] or [N,151] found. Keys=" + ",".join(npz.files))

    priority = [
        "unit_motions_physical",
        "motions_physical",
        "motion_151",
        "unit_motions",
        "motions",
        "motion",
        "poses",
    ]

    for name in priority:
        for k, arr in candidates:
            if k == name:
                return k, arr

    candidates.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    return candidates[0]

def crop_or_pad(m, target_len):
    m = np.asarray(m, dtype=np.float32)
    if len(m) == target_len:
        return m
    if len(m) > target_len:
        s = max(0, (len(m) - target_len) // 2)
        return m[s:s + target_len]
    pad = np.repeat(m[-1:], target_len - len(m), axis=0)
    return np.concatenate([m, pad], axis=0)

def motion_metrics(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT]
    droot = np.linalg.norm(root[1:] - root[:-1], axis=1) if len(m) > 1 else np.zeros(1)
    drot = np.linalg.norm(rot[1:] - rot[:-1], axis=1) if len(m) > 1 else np.zeros(1)

    rot_j = rot.reshape(len(rot), 24, 6)
    lower = rot_j[:, 0:8].reshape(len(rot), -1)
    torso = rot_j[:, 8:14].reshape(len(rot), -1)
    upper = rot_j[:, 14:24].reshape(len(rot), -1)

    def act(x):
        if len(x) <= 1:
            return 0.0
        return float(np.linalg.norm(x[1:] - x[:-1], axis=1).mean())

    root_radius = float(np.linalg.norm(root - root[:1], axis=1).max())
    activity = float(drot.mean())
    upper_activity = act(upper)
    torso_activity = act(torso)
    lower_activity = act(lower)

    return {
        "root_radius": root_radius,
        "root_final": float(np.linalg.norm(root[-1] - root[0])),
        "root_speed_mean": float(droot.mean()),
        "global_rot_jump_p95": float(np.percentile(drot, 95)),
        "global_rot_jump_max": float(drot.max()),
        "activity": activity,
        "upper_activity": upper_activity,
        "torso_activity": torso_activity,
        "lower_activity": lower_activity,
        "contact_switch": float(np.abs(m[1:, CONTACT] - m[:-1, CONTACT]).sum(axis=1).mean()) if len(m) > 1 else 0.0,
    }

def score(met, args):
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

    expr = 1.50 * met["upper_activity"] + 1.10 * met["torso_activity"] + 0.30 * met["lower_activity"]
    root_safe = np.exp(-5.0 * met["root_radius"])
    jump_safe = np.exp(-1.2 * met["global_rot_jump_p95"])
    non_static = min(met["activity"] / 0.16, 1.0)
    return float(expr * root_safe * jump_safe * (0.35 + 0.65 * non_static))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_len", type=int, default=45)
    ap.add_argument("--top_k", type=int, default=900)
    ap.add_argument("--render_top_k", type=int, default=24)
    ap.add_argument("--audio_dim", type=int, default=803)

    ap.add_argument("--max_root_radius", type=float, default=0.25)
    ap.add_argument("--max_rot_jump_p95", type=float, default=1.10)
    ap.add_argument("--min_activity", type=float, default=0.030)
    ap.add_argument("--min_upper_activity", type=float, default=0.020)
    ap.add_argument("--min_torso_activity", type=float, default=0.008)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pkl_dir = out_dir / "dunhuang_realunit_style_pkl"
    npy_dir = out_dir / "top_unit_npy"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz, allow_pickle=True)
    motion_key, motions = find_motion_array(npz)
    print(f"motion_key={motion_key}, shape={motions.shape}")

    rows = []
    for i in range(motions.shape[0]):
        m = crop_or_pad(motions[i], args.target_len)
        met = motion_metrics(m)
        sc = score(met, args)
        if sc is None:
            continue
        rows.append((sc, i, m, met))

    rows.sort(key=lambda x: x[0], reverse=True)
    selected = rows[:args.top_k]

    report_items = []
    for rank, (sc, idx, m, met) in enumerate(selected, start=1):
        item = {
            "motion": m.astype(np.float32),
            "motion_151": m.astype(np.float32),
            "poses": m.astype(np.float32),
            "audio_feature": np.zeros((len(m), args.audio_dim), dtype=np.float32),
            "original_filename": f"realunit_idx{idx:06d}",
            "source_file": f"realunit_idx{idx:06d}",
            "metadata": {
                "rank": rank,
                "source_npz": str(args.npz),
                "source_index": int(idx),
                "score": float(sc),
                **met,
            }
        }

        pkl_path = pkl_dir / f"realunit_rank{rank:04d}_idx{idx:06d}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(item, f)

        if rank <= args.render_top_k:
            np.save(npy_dir / f"realunit_rank{rank:04d}_idx{idx:06d}.npy", m[None].astype(np.float32))

        report_items.append({
            "rank": rank,
            "source_index": int(idx),
            "score": float(sc),
            "pkl": str(pkl_path),
            **met,
        })

    report = {
        "npz": str(args.npz),
        "motion_key": motion_key,
        "input_shape": list(motions.shape),
        "target_len": args.target_len,
        "selected": len(selected),
        "pkl_dir": str(pkl_dir),
        "top_unit_npy_dir": str(npy_dir),
        "thresholds": {
            "max_root_radius": args.max_root_radius,
            "max_rot_jump_p95": args.max_rot_jump_p95,
            "min_activity": args.min_activity,
            "min_upper_activity": args.min_upper_activity,
            "min_torso_activity": args.min_torso_activity,
        },
        "items": report_items,
    }

    report_path = out_dir / "realunit_style_pool_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected={len(selected)}")
    print(f"pkl_dir={pkl_dir}")
    print(f"top_unit_npy_dir={npy_dir}")
    print(f"report={report_path}")

if __name__ == "__main__":
    main()
