#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np


ROOT_X = 4
ROOT_Z = 6


def load_motion_file(path: Path, uid: Optional[str] = None) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            for k in ["motion", "motion_151", "poses", "unit_motions_physical", "unit_motion", "x"]:
                if k in data:
                    arr = np.asarray(data[k], dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr[0]
                    if arr.ndim == 2 and arr.shape[-1] == 151:
                        return arr.astype(np.float32)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[-1] == 151:
            return arr.astype(np.float32)

    if suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if data.ndim == 0 and isinstance(data.item(), dict):
            data = data.item()
            for k in ["motion", "motion_151", "poses", "unit_motions_physical", "unit_motion", "x"]:
                if k in data:
                    arr = np.asarray(data[k], dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr[0]
                    if arr.ndim == 2 and arr.shape[-1] == 151:
                        return arr.astype(np.float32)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[-1] == 151:
            return arr.astype(np.float32)

    if suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        keys = set(z.files)

        # Big RAG DB case: unit_ids + unit_motions_physical
        if uid is not None and "unit_ids" in keys and "unit_motions_physical" in keys:
            ids = [str(x) for x in z["unit_ids"]]
            if str(uid) in ids:
                idx = ids.index(str(uid))
                arr = np.asarray(z["unit_motions_physical"][idx], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[-1] == 151:
                    return arr.astype(np.float32)

        for k in ["motion", "motion_151", "poses", "unit_motions_physical", "unit_motions"]:
            if k in keys:
                arr = np.asarray(z[k], dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.ndim == 2 and arr.shape[-1] == 151:
                    return arr.astype(np.float32)

    raise ValueError(f"Cannot load [T,151] motion from {path}")


def find_unit_motion(uid: str, extra_roots: List[str]) -> Tuple[Path, np.ndarray]:
    roots = [
        *[Path(x) for x in extra_roots if x],
        Path("output/whitelist_candidate_units_v2"),
        Path("data/dunhuang_bvh/stationary_whitelist_v2_27units"),
        Path("data/dunhuang_bvh/stationary_whitelist_v2"),
        Path("data/dunhuang_bvh/single_unit45_recon_physical"),
        Path("data/dunhuang_choreo_unit_rag"),
    ]

    patterns = [
        f"*{uid}*.pkl",
        f"*{uid}*.npy",
        f"*{uid}*.npz",
    ]

    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            candidates.extend(sorted(root.rglob(pat)))

    # Also try common full DBs even if filename does not contain uid.
    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob("index*.npz")))

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        try:
            motion = load_motion_file(p, uid=uid)
            if motion.ndim == 2 and motion.shape[1] == 151 and len(motion) >= 2:
                return p, motion[:45].astype(np.float32)
        except Exception:
            pass

    raise FileNotFoundError(
        f"Cannot find unit {uid}. Try passing --unit_root to the directory containing exported whitelist pkl/npy files."
    )


def p95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 95))


def motion_speed(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(x, axis=0), axis=-1)


def motion_jerk(x: np.ndarray) -> np.ndarray:
    if len(x) < 4:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(np.diff(x, n=3, axis=0), axis=-1)


def rot_dims_for_joints(joints):
    dims = []
    for j in joints:
        base = 7 + 6 * int(j)
        dims.extend(range(base, base + 6))
    return [d for d in dims if 0 <= d < 151]


# SMPL-like rough groups for quick screening.
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
UPPER_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_DIMS = rot_dims_for_joints(LOWER_JOINTS)
UPPER_DIMS = rot_dims_for_joints(UPPER_JOINTS)


def eval_metrics(gen: np.ndarray, gt: np.ndarray) -> dict:
    T = min(len(gen), len(gt), 45)
    gen = gen[:T].astype(np.float32)
    gt = gt[:T].astype(np.float32)

    gen_speed = motion_speed(gen)
    gt_speed = motion_speed(gt)
    gen_jerk = motion_jerk(gen)
    gt_jerk = motion_jerk(gt)

    return {
        "phys_mse": float(np.mean((gen - gt) ** 2)),
        "rootxz_mse": float(np.mean((gen[:, [ROOT_X, ROOT_Z]] - gt[:, [ROOT_X, ROOT_Z]]) ** 2)),
        "jump_ratio_p95": float(p95(gen_speed) / max(p95(gt_speed), 1e-8)),
        "jerk_ratio_p95": float(p95(gen_jerk) / max(p95(gt_jerk), 1e-8)),
        "upper_mse": float(np.mean((gen[:, UPPER_DIMS] - gt[:, UPPER_DIMS]) ** 2)) if UPPER_DIMS else 0.0,
        "lower_mse": float(np.mean((gen[:, LOWER_DIMS] - gt[:, LOWER_DIMS]) ** 2)) if LOWER_DIMS else 0.0,
    }


def run_generate(args, uid: str, gt: np.ndarray, unit_dir: Path, out_npy: Path):
    start = unit_dir / f"unit_{uid}_start.npy"
    end = unit_dir / f"unit_{uid}_end.npy"
    gt_path = unit_dir / f"unit_{uid}_gt.npy"

    np.save(start, gt[0].astype(np.float32))
    np.save(end, gt[-1].astype(np.float32))
    np.save(gt_path, gt.astype(np.float32))

    env = os.environ.copy()
    env.update({
        # Keep this endpoint-only and clean.
        "EDGE_DISABLE_TRAJ_COND": "1",
        "EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER": "1",
        "EDGE_UNIT_SOFT_PRIOR": "0",
        "EDGE_ENABLE_TEXT_CONTEXT_RAG": "0",
        "EDGE_ENABLE_RAG_SUMMARY_TOKEN": "0",
        "EDGE_BEAT_GUIDANCE": "0",
        "EDGE_ENERGY_COND": "0",
        "EDGE_HARD_KEYFRAME_PROJECT": "1",
        "EDGE_INFER_PROJECT_XSTART": "1",
        "EDGE_FINAL_KEYFRAME_PROJECT": "1",
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    })

    cmd = [
        sys.executable, "generate_controlled.py",
        "--checkpoint", args.ckpt,
        "--music", args.music,
        "--start_pose", str(start),
        "--end_pose", str(end),
        "--out", str(out_npy),
        "--feature_type", args.feature_type,
        "--audio_dim", str(args.audio_dim),
        "--seq_len", "45",
        "--num_frames", "45",
        "--pose_space", "physical",
        "--sampler", args.sampler,
        "--guidance_weight", "1.0",
        "--endpoint_keyframe_strength", str(args.endpoint_strength),
        "--mid_keyframe_strength", "0.0",
        "--infer_keyframe_width", "0",
        "--disable_traj_cond",
        "--hard_keyframe_project",
        "--infer_project_xstart",
        "--no_tto",
        "--trajectory", "0,0;0,0",
        "--mixed_precision", args.mixed_precision,
    ]

    if args.constrain_contacts:
        cmd.append("--constrain_contacts")

    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--unit_list", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--unit_root", action="append", default=[])
    ap.add_argument("--music", default="test_music_bank/dunhuangwu2.wav")
    ap.add_argument("--feature_type", default="hybrid")
    ap.add_argument("--audio_dim", type=int, default=803)
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--endpoint_strength", type=float, default=1.0)
    ap.add_argument("--constrain_contacts", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_ids = [x.strip() for x in Path(args.unit_list).read_text().splitlines() if x.strip() and not x.strip().startswith("#")]

    summary = []
    for uid in unit_ids:
        unit_dir = out_dir / f"unit_{uid}_assets"
        unit_dir.mkdir(parents=True, exist_ok=True)

        src, gt = find_unit_motion(uid, args.unit_root)
        out_npy = out_dir / f"unit_{uid}.npy"
        eval_json = out_dir / f"unit_{uid}_eval.json"

        run_generate(args, uid, gt, unit_dir, out_npy)

        gen = np.load(out_npy).astype(np.float32)
        m = eval_metrics(gen, gt)
        m.update({
            "unit": uid,
            "ckpt": args.ckpt,
            "gt_source": str(src),
            "gen_motion": str(out_npy),
            "gt_motion": str(unit_dir / f"unit_{uid}_gt.npy"),
            "sampler": args.sampler,
            "endpoint_strength": args.endpoint_strength,
        })

        eval_json.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print("EVAL", json.dumps(m, ensure_ascii=False), flush=True)
        summary.append(m)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
