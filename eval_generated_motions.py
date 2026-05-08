#!/usr/bin/env python3
"""Formal raw-vs-final evaluator for EDGE generated motions.

This script intentionally refuses to produce a formal evaluation from only a
post-anchor/final motion.  It reports raw/native and final/post-anchor metrics
in separate dictionaries so final_ADE=0 cannot be mistaken for native trajectory
ability.

v2 fixes:
- Clear missing-file diagnostics instead of the vague "requires both" error.
- --raw_motion can be omitted if it can be inferred from --final_motion or --meta.
- --target_traj can be inferred from common sibling names when omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT_SLICE = slice(7, 151)
UPPER_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def _rot_indices(joints):
    idx = []
    for j in joints:
        idx.extend(range(7 + 6 * int(j), 7 + 6 * int(j) + 6))
    return np.asarray([i for i in idx if 0 <= i < 151], dtype=np.int64)


UPPER_IDX = _rot_indices(UPPER_JOINTS)
LOWER_IDX = _rot_indices(LOWER_JOINTS)


def load_json(path: str) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_motion(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"motion file not found: {p}")
    arr = np.load(str(p), allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        for key in ("motion", "motion_final", "motion_raw", "pose", "poses"):
            if key in d:
                arr = d[key]
                break
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"Expected [T,151] motion from {path}, got {arr.shape}")
    return arr.astype(np.float32)


def _resample_2d(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected trajectory [T,2+] got {arr.shape}")
    arr = arr[:, :2]
    if len(arr) == target_len:
        return arr.astype(np.float32)
    if len(arr) <= 1:
        return np.repeat(arr[:1], target_len, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, target_len)
    x = np.interp(dst, src, arr[:, 0])
    z = np.interp(dst, src, arr[:, 1])
    return np.stack([x, z], axis=-1).astype(np.float32)


def load_traj(path: str, target_len: int) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"target trajectory file not found: {p}")
    return _resample_2d(np.load(str(p), allow_pickle=True), target_len)


def infer_raw_from_final(final_path: Optional[Path], meta: Dict) -> Optional[Path]:
    # Prefer explicit meta fields written by generate_controlled.py variants.
    for key in ("raw_motion_path", "motion_raw_path", "raw_path", "native_motion_path"):
        value = meta.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value))

    if final_path is None:
        return None

    stem = final_path.stem
    parent = final_path.parent
    candidates = []

    # Common EDGE output: <stem>_raw.npy next to <stem>.npy.
    candidates.append(parent / f"{stem}_raw.npy")

    # If final is <stem>_anchor.npy or <stem>_final.npy, raw is often <stem>_raw.npy.
    for suffix in ("_anchor", "_final", "_post_anchor", "_ik", "_legik"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            candidates.append(parent / f"{base}_raw.npy")
            candidates.append(parent / f"{base}_native.npy")

    # If final itself already ends with _raw, do not infer same file unless requested.
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else None


def infer_traj_from_final(final_path: Optional[Path], meta: Dict) -> Optional[Path]:
    for key in ("target_traj_path", "trajectory_target_path", "traj_path"):
        value = meta.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value))
    if final_path is None:
        return None
    stem = final_path.stem
    parent = final_path.parent
    candidates = [parent / f"{stem}_target_traj.npy"]
    for suffix in ("_anchor", "_final", "_post_anchor", "_ik", "_legik"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            candidates.append(parent / f"{base}_target_traj.npy")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def trajectory_metrics(motion: np.ndarray, target: Optional[np.ndarray]) -> Dict[str, float]:
    if target is None:
        return {"trajectory_ade_m": None, "trajectory_fde_m": None}
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
    target = _resample_2d(target, len(motion))
    err = np.linalg.norm(root - target, axis=-1)
    return {
        "trajectory_ade_m": float(err.mean()),
        "trajectory_fde_m": float(err[-1]),
    }


def transition_jerk_global(motion: np.ndarray) -> float:
    if len(motion) < 4:
        return 0.0
    x = motion[:, ROT_SLICE]
    acc = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return float(np.linalg.norm(acc, axis=-1).mean())


def contact_phase_break_global(motion: np.ndarray) -> float:
    if len(motion) < 2:
        return 0.0
    c = (motion[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    changes = np.abs(c[1:] - c[:-1]).mean(axis=1)
    return float(changes.mean())


def freezing_score(motion: np.ndarray, threshold: float = 0.015) -> float:
    if len(motion) < 2:
        return 1.0
    v = np.linalg.norm(motion[1:, ROT_SLICE] - motion[:-1, ROT_SLICE], axis=-1)
    return float((v < threshold).mean())


def activity_stats(motion: np.ndarray) -> Dict[str, float]:
    if len(motion) < 2:
        return {
            "motion_energy": 0.0,
            "upper_activity": 0.0,
            "lower_activity": 0.0,
            "root_speed": 0.0,
            "spatial_range": 0.0,
        }
    diff = motion[1:] - motion[:-1]
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    root_vel = root[1:] - root[:-1]
    return {
        "motion_energy": float(np.sqrt(np.mean(diff[:, ROT_SLICE] ** 2))),
        "upper_activity": float(np.sqrt(np.mean(diff[:, UPPER_IDX] ** 2))) if UPPER_IDX.size else 0.0,
        "lower_activity": float(np.sqrt(np.mean(diff[:, LOWER_IDX] ** 2))) if LOWER_IDX.size else 0.0,
        "root_speed": float(np.linalg.norm(root_vel, axis=1).mean()) if len(root_vel) else 0.0,
        "spatial_range": float(np.linalg.norm(root.max(axis=0) - root.min(axis=0))),
    }


def foot_slide_proxy(motion: np.ndarray, contact_threshold: float = 0.5, speed_threshold: float = 0.015) -> float:
    """Proxy metric when FK foot positions are unavailable.

    It flags frames where the motion claims a foot is in contact while root X/Z
    is still moving.  Replace with an FK foot-slide evaluator if available.
    """
    if len(motion) < 2:
        return 0.0
    contact = (motion[:-1, CONTACT_SLICE] > contact_threshold).astype(np.float32)
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    root_speed = np.linalg.norm(root[1:] - root[:-1], axis=-1)
    contact_any = contact.mean(axis=1) > 0.25
    if not np.any(contact_any):
        return 0.0
    slide = (root_speed > speed_threshold) & contact_any
    return float(slide.mean())


def root_lower_sync(motion: np.ndarray) -> float:
    """Proxy root-lower kinematic sync in [0,1], higher is better."""
    if len(motion) < 2:
        return 1.0
    diff = motion[1:] - motion[:-1]
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    root_speed = np.linalg.norm(root[1:] - root[:-1], axis=-1)
    lower = np.sqrt(np.mean(diff[:, LOWER_IDX] ** 2, axis=1)) if LOWER_IDX.size else np.zeros_like(root_speed)
    if float(root_speed.max()) <= 1e-8:
        return 1.0
    active = root_speed > np.percentile(root_speed, 60)
    if not np.any(active):
        return 1.0
    # If root moves while lower body remains nearly static, sync is poor.
    ratio = lower[active] / np.maximum(root_speed[active], 1e-8)
    score = np.clip(np.median(ratio) / 0.30, 0.0, 1.0)
    return float(score)


def compute_metrics(motion: np.ndarray, target: Optional[np.ndarray], stage: str) -> Dict[str, object]:
    out: Dict[str, object] = {"stage": stage, "num_frames": int(len(motion))}
    out.update(trajectory_metrics(motion, target))
    out.update(activity_stats(motion))
    out.update({
        "transition_jerk": transition_jerk_global(motion),
        "contact_phase_break": contact_phase_break_global(motion),
        "freezing_score": freezing_score(motion),
        "foot_slide_rate": foot_slide_proxy(motion),
        "root_lower_sync": root_lower_sync(motion),
    })
    return out


def numeric_delta(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, float]:
    out = {}
    for k, va in a.items():
        vb = b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = float(va) - float(vb)
    return out


def build_warnings(metrics_raw: Dict[str, object], metrics_final: Dict[str, object]) -> List[str]:
    warnings: List[str] = []
    final_ade = metrics_final.get("trajectory_ade_m")
    raw_ade = metrics_raw.get("trajectory_ade_m")
    final_slide = float(metrics_final.get("foot_slide_rate", 0.0))
    raw_slide = float(metrics_raw.get("foot_slide_rate", 0.0))
    final_sync = float(metrics_final.get("root_lower_sync", 0.0))
    raw_sync = float(metrics_raw.get("root_lower_sync", 0.0))

    if final_ade is not None and raw_ade is not None:
        if float(final_ade) <= 1e-6 and float(raw_ade) > 1e-3:
            warnings.append("final ADE is zero while raw ADE is non-zero; this is post-anchor system control, not native trajectory ability.")
    if final_slide > max(raw_slide * 1.5, raw_slide + 0.05):
        warnings.append("foot_slide_rate increased after final/post-anchor motion; trajectory may be achieved by dragging.")
    if final_sync + 0.10 < raw_sync:
        warnings.append("root_lower_sync dropped after final/post-anchor motion; lower body may not follow root naturally.")
    return warnings


def _missing_report(paths: List[Tuple[str, Optional[Path]]]) -> str:
    rows = []
    for name, path in paths:
        if path is None:
            rows.append(f"  - {name}: <not provided and could not infer>")
        elif not path.exists():
            rows.append(f"  - {name}: {path}  [missing]")
        else:
            rows.append(f"  - {name}: {path}  [ok]")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_motion", default="", help="Native/raw model motion, e.g. *_raw.npy. If omitted, inferred from --final_motion or --meta.")
    ap.add_argument("--final_motion", required=True, help="Final/post-anchor motion, e.g. *.npy or *_anchor.npy")
    ap.add_argument("--target_traj", default="", help="Target trajectory [T,2]. If omitted, inferred from siblings/meta when possible.")
    ap.add_argument("--meta", default="", help="Optional generation meta JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--formal", action="store_true", help="Reject single-stage or suspicious evaluation usage")
    ap.add_argument("--allow_same_file", action="store_true")
    args = ap.parse_args()

    meta = load_json(args.meta)
    final_path = Path(args.final_motion) if args.final_motion else None
    raw_path = Path(args.raw_motion) if args.raw_motion else infer_raw_from_final(final_path, meta)
    target_path = Path(args.target_traj) if args.target_traj else infer_traj_from_final(final_path, meta)

    missing_required = []
    if raw_path is None or not raw_path.exists():
        missing_required.append(("raw_motion", raw_path))
    if final_path is None or not final_path.exists():
        missing_required.append(("final_motion", final_path))

    if args.formal and missing_required:
        raise FileNotFoundError(
            "Formal evaluation requires existing raw and final motion files.\n"
            + _missing_report([("raw_motion", raw_path), ("final_motion", final_path), ("target_traj", target_path)])
            + "\n\nTip: the names in the README are examples. Use your actual generated files, e.g. `find output -name '*_raw.npy' | head`."
        )

    if raw_path is None or final_path is None:
        raise ValueError("raw_motion/final_motion could not be resolved.")

    if args.formal and raw_path.exists() and final_path.exists():
        if raw_path.resolve() == final_path.resolve() and not args.allow_same_file:
            raise ValueError("Formal evaluation requires distinct raw and final motion files.")

    raw = load_motion(str(raw_path))
    final = load_motion(str(final_path))
    if len(raw) != len(final):
        raise ValueError(f"raw/final length mismatch: raw={len(raw)}, final={len(final)}")

    target = None
    if target_path is not None and target_path.exists():
        target = load_traj(str(target_path), target_len=len(raw))
    elif args.formal:
        print("⚠️ No target trajectory found; trajectory ADE/FDE will be null.", file=sys.stderr)

    metrics_raw = compute_metrics(raw, target, stage="raw_native")
    metrics_final = compute_metrics(final, target, stage="post_anchor_or_final")
    delta = numeric_delta(metrics_final, metrics_raw)
    warnings = build_warnings(metrics_raw, metrics_final)

    payload: Dict[str, object] = {
        "raw_motion": str(raw_path),
        "final_motion": str(final_path),
        "target_traj": str(target_path or ""),
        "meta": str(args.meta or ""),
        "metrics_raw": metrics_raw,
        "metrics_final": metrics_final,
        "delta_final_minus_raw": delta,
        "warnings": warnings,
        "notes": [
            "trajectory_ade_m in metrics_final may be post-anchor/system-level control.",
            "Use metrics_raw for native trajectory ability.",
            "foot_slide_rate is a contact-root-speed proxy unless replaced by an FK foot-slide evaluator.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Formal raw/final evaluation saved: {out_path}")
    if warnings:
        print("⚠️ Warnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
