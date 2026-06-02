#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)

# Approximate SMPL joint layout used by EDGE rendering:
# 0 pelvis/root, 7/8 ankles, 10/11 feet, 12 neck, 15 head,
# 16/17 shoulders, 18/19 elbows, 20/21 wrists, 22/23 hands.
WRIST_IDS = [20, 21, 22, 23]
FOOT_IDS = [7, 8, 10, 11]
END_IDS = WRIST_IDS + FOOT_IDS


def parse_bool(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def find_motion_array(npz):
    candidates = []
    for k in npz.files:
        arr = npz[k]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim == 3 and arr.shape[-1] == 151:
            candidates.append((k, arr))
        elif arr.ndim == 2 and arr.shape[-1] == 151:
            candidates.append((k, arr[:, None, :]))

    if not candidates:
        raise RuntimeError("No [N,T,151] or [N,151] motion array found. Keys=" + ",".join(npz.files))

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


def safe_percentile_norm(values, lo=5, hi=95):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    p_lo = float(np.percentile(arr, lo))
    p_hi = float(np.percentile(arr, hi))
    if p_hi <= p_lo + 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)


def motion_metrics_151(m):
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

    def var_vel(x):
        if len(x) <= 2:
            return 0.0
        v = np.linalg.norm(x[1:] - x[:-1], axis=1)
        return float(np.var(v))

    upper_activity = act(upper)
    torso_activity = act(torso)
    lower_activity = act(lower)
    activity = float(drot.mean())

    rot_range = float(np.mean(np.std(rot_j.reshape(len(rot_j), -1), axis=0)))
    upper_var = var_vel(upper)
    torso_var = var_vel(torso)

    return {
        "root_radius": float(np.linalg.norm(root - root[:1], axis=1).max()),
        "root_final": float(np.linalg.norm(root[-1] - root[0])),
        "root_speed_mean": float(droot.mean()),
        "global_rot_jump_p95": float(np.percentile(drot, 95)),
        "global_rot_jump_max": float(drot.max()),
        "activity": activity,
        "upper_activity": upper_activity,
        "torso_activity": torso_activity,
        "lower_activity": lower_activity,
        "upper_velocity_variance": upper_var,
        "torso_velocity_variance": torso_var,
        "rotation_range_proxy": rot_range,
        "contact_switch": float(np.abs(m[1:, CONTACT] - m[:-1, CONTACT]).sum(axis=1).mean()) if len(m) > 1 else 0.0,
    }


def try_fk_batch(motions, device="cuda"):
    try:
        import torch
        from dataset.quaternion import ax_from_6v
        from vis import SMPLSkeleton

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        device_t = torch.device(device)
        smpl = SMPLSkeleton(device=device_t)

        with torch.no_grad():
            x = torch.from_numpy(motions).float().to(device_t)
            pos = x[:, :, 4:7]
            q6d = x[:, :, 7:].reshape(x.shape[0], x.shape[1], 24, 6)
            qax = ax_from_6v(q6d)
            joints = smpl.forward(qax, pos).detach().cpu().numpy().astype(np.float32)
        return joints
    except Exception as exc:
        print(f"⚠️ FK disabled/fallback because: {type(exc).__name__}: {exc}")
        return None


def fk_visual_metrics(m, joints):
    if joints is None:
        return {
            "fk_extension": 0.0,
            "fk_wrist_span": 0.0,
            "fk_bbox_xy_area": 0.0,
            "fk_bbox_xz_area": 0.0,
            "fk_endpoint_velocity_variance": 0.0,
            "fk_three_bend_proxy": 0.0,
        }

    root = joints[:, 0:1, :]
    end = joints[:, END_IDS, :]
    wrist = joints[:, WRIST_IDS, :]

    end_dist = np.linalg.norm(end - root, axis=-1)
    extension = float(np.max(end_dist))

    wrist_span = float(np.mean(np.linalg.norm(wrist[:, 0] - wrist[:, 1], axis=-1))) if wrist.shape[1] >= 2 else 0.0

    xy = joints[:, :, [0, 1]]
    xz = joints[:, :, [0, 2]]
    bbox_xy = (xy[:, :, 0].max(axis=1) - xy[:, :, 0].min(axis=1)) * (xy[:, :, 1].max(axis=1) - xy[:, :, 1].min(axis=1))
    bbox_xz = (xz[:, :, 0].max(axis=1) - xz[:, :, 0].min(axis=1)) * (xz[:, :, 1].max(axis=1) - xz[:, :, 1].min(axis=1))

    if len(end) > 1:
        ev = np.linalg.norm(end[1:] - end[:-1], axis=-1).mean(axis=1)
        endpoint_var = float(np.var(ev))
    else:
        endpoint_var = 0.0

    # A rough Dunhuang "S-curve / three-bend" proxy:
    # lateral displacement among pelvis, neck/head, wrists.
    pelvis = joints[:, 0, :]
    neck = joints[:, 12, :] if joints.shape[1] > 12 else joints[:, 0, :]
    head = joints[:, 15, :] if joints.shape[1] > 15 else neck
    left_wrist = joints[:, 20, :] if joints.shape[1] > 20 else head
    right_wrist = joints[:, 21, :] if joints.shape[1] > 21 else head

    lateral_head = np.abs(head[:, 0] - pelvis[:, 0])
    lateral_neck = np.abs(neck[:, 0] - pelvis[:, 0])
    wrist_asym = np.abs(left_wrist[:, 1] - right_wrist[:, 1]) + np.abs(left_wrist[:, 0] - right_wrist[:, 0])
    three_bend = float(np.mean(lateral_head + 0.5 * lateral_neck + 0.25 * wrist_asym))

    return {
        "fk_extension": extension,
        "fk_wrist_span": wrist_span,
        "fk_bbox_xy_area": float(np.max(bbox_xy)),
        "fk_bbox_xz_area": float(np.max(bbox_xz)),
        "fk_endpoint_velocity_variance": endpoint_var,
        "fk_three_bend_proxy": three_bend,
    }


def hard_gate(met, args):
    if met["root_radius"] > args.max_root_radius:
        return False
    if met["global_rot_jump_p95"] > args.max_rot_jump_p95:
        return False
    if met["activity"] < args.min_activity:
        return False
    if met["upper_activity"] < args.min_upper_activity:
        return False
    if met["torso_activity"] < args.min_torso_activity:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_len", type=int, default=45)
    ap.add_argument("--top_k", type=int, default=1200)
    ap.add_argument("--render_top_k", type=int, default=32)
    ap.add_argument("--audio_dim", type=int, default=803)

    ap.add_argument("--use_fk", type=int, default=int(os.environ.get("EDGE_V16_USE_FK", "1")))
    ap.add_argument("--fk_device", default=os.environ.get("EDGE_V16_FK_DEVICE", "auto"))
    ap.add_argument("--fk_batch_size", type=int, default=int(os.environ.get("EDGE_V16_FK_BATCH", "256")))

    ap.add_argument("--max_root_radius", type=float, default=0.25)
    ap.add_argument("--max_rot_jump_p95", type=float, default=1.10)
    ap.add_argument("--min_activity", type=float, default=0.030)
    ap.add_argument("--min_upper_activity", type=float, default=0.020)
    ap.add_argument("--min_torso_activity", type=float, default=0.008)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pkl_dir = out_dir / "dunhuang_visual_first_pkl"
    npy_dir = out_dir / "top_visual_unit_npy"
    render_manifest = out_dir / "top_visual_render_manifest.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz, allow_pickle=True)
    motion_key, motions_raw = find_motion_array(npz)
    print(f"motion_key={motion_key}, shape={motions_raw.shape}")

    rows = []
    for start in range(0, motions_raw.shape[0], args.fk_batch_size):
        end = min(motions_raw.shape[0], start + args.fk_batch_size)
        batch = np.stack([crop_or_pad(motions_raw[i], args.target_len) for i in range(start, end)], axis=0)

        joints_batch = None
        if args.use_fk:
            joints_batch = try_fk_batch(batch, device=args.fk_device)

        for local_i, m in enumerate(batch):
            idx = start + local_i
            met = motion_metrics_151(m)

            if not hard_gate(met, args):
                continue

            fk_met = fk_visual_metrics(m, None if joints_batch is None else joints_batch[local_i])
            met.update(fk_met)
            rows.append({
                "source_index": int(idx),
                "motion": m,
                **met,
            })

    if not rows:
        raise RuntimeError("No units passed visual/safety gates. Loosen thresholds.")

    # Normalize visual fields across surviving candidates.
    fields = [
        "upper_activity",
        "torso_activity",
        "upper_velocity_variance",
        "torso_velocity_variance",
        "rotation_range_proxy",
        "fk_extension",
        "fk_wrist_span",
        "fk_bbox_xy_area",
        "fk_bbox_xz_area",
        "fk_endpoint_velocity_variance",
        "fk_three_bend_proxy",
    ]

    norm = {}
    for f in fields:
        norm[f] = safe_percentile_norm([r.get(f, 0.0) for r in rows])

    for i, r in enumerate(rows):
        visual = (
            1.35 * norm["upper_activity"][i]
            + 1.10 * norm["torso_activity"][i]
            + 0.75 * norm["upper_velocity_variance"][i]
            + 0.45 * norm["torso_velocity_variance"][i]
            + 0.55 * norm["rotation_range_proxy"][i]
            + 1.10 * norm["fk_extension"][i]
            + 0.65 * norm["fk_wrist_span"][i]
            + 0.65 * norm["fk_bbox_xy_area"][i]
            + 0.35 * norm["fk_bbox_xz_area"][i]
            + 0.75 * norm["fk_endpoint_velocity_variance"][i]
            + 0.90 * norm["fk_three_bend_proxy"][i]
        )

        safety = (
            np.exp(-4.0 * r["root_radius"])
            * np.exp(-0.85 * r["global_rot_jump_p95"])
        )

        non_static = min(r["activity"] / 0.16, 1.0)

        r["visual_score"] = float(visual)
        r["safety_score"] = float(safety)
        r["final_score"] = float(visual * safety * (0.35 + 0.65 * non_static))

    rows.sort(key=lambda r: r["final_score"], reverse=True)
    selected = rows[: args.top_k]

    report_items = []
    csv_path = out_dir / "visual_first_scores.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = None

        for rank, r in enumerate(selected, start=1):
            m = r.pop("motion")
            idx = r["source_index"]

            item = {
                "motion": m.astype(np.float32),
                "motion_151": m.astype(np.float32),
                "poses": m.astype(np.float32),
                "audio_feature": np.zeros((len(m), args.audio_dim), dtype=np.float32),
                "original_filename": f"visual_first_idx{idx:06d}",
                "source_file": f"visual_first_idx{idx:06d}",
                "metadata": {
                    "rank": rank,
                    "source_npz": str(args.npz),
                    **{k: v for k, v in r.items() if isinstance(v, (int, float, str))},
                },
            }

            pkl_path = pkl_dir / f"visual_first_rank{rank:04d}_idx{idx:06d}.pkl"
            with open(pkl_path, "wb") as pf:
                pickle.dump(item, pf)

            if rank <= args.render_top_k:
                npy_path = npy_dir / f"visual_first_rank{rank:04d}_idx{idx:06d}.npy"
                np.save(npy_path, m[None].astype(np.float32))

            report_row = {
                "rank": rank,
                "pkl": str(pkl_path),
                **r,
            }
            report_items.append(report_row)

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(report_row.keys()))
                writer.writeheader()
            writer.writerow(report_row)

    report = {
        "npz": str(args.npz),
        "motion_key": motion_key,
        "input_shape": list(motions_raw.shape),
        "target_len": args.target_len,
        "use_fk": bool(args.use_fk),
        "fk_device": args.fk_device,
        "selected": len(selected),
        "pkl_dir": str(pkl_dir),
        "top_visual_unit_npy_dir": str(npy_dir),
        "csv": str(csv_path),
        "thresholds": {
            "max_root_radius": args.max_root_radius,
            "max_rot_jump_p95": args.max_rot_jump_p95,
            "min_activity": args.min_activity,
            "min_upper_activity": args.min_upper_activity,
            "min_torso_activity": args.min_torso_activity,
        },
        "score_definition": {
            "visual_score": "upper/torso activity + endpoint variance + FK extension/silhouette/three-bend proxy",
            "safety_score": "root stability and rotation jump penalty",
            "final_score": "visual_score * safety_score * non_static",
        },
        "items": report_items,
    }

    report_path = out_dir / "visual_first_pool_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(render_manifest, "w", encoding="utf-8") as f:
        for npy in sorted(npy_dir.glob("*.npy")):
            f.write(str(npy) + "\n")

    md = out_dir / "VISUAL_FIRST_POOL_SUMMARY.md"
    md.write_text(
        "# Visual-First Prior Pool Summary\n\n"
        f"- Source NPZ: `{args.npz}`\n"
        f"- Motion key: `{motion_key}`\n"
        f"- Input shape: `{list(motions_raw.shape)}`\n"
        f"- Selected units: `{len(selected)}`\n"
        f"- Use FK visual metrics: `{bool(args.use_fk)}`\n"
        f"- PKL dir: `{pkl_dir}`\n"
        f"- Top unit npy dir: `{npy_dir}`\n"
        f"- Report: `{report_path}`\n"
        f"- CSV: `{csv_path}`\n\n"
        "## Interpretation\n\n"
        "This pool prioritizes Dunhuang-like visual tension before music matching. "
        "It should be used as the first funnel stage before physics and onset scheduling.\n",
        encoding="utf-8",
    )

    print(f"selected={len(selected)}")
    print(f"pkl_dir={pkl_dir}")
    print(f"top_visual_unit_npy_dir={npy_dir}")
    print(f"report={report_path}")
    print(f"csv={csv_path}")
    print(f"summary={md}")


if __name__ == "__main__":
    main()
