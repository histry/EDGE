#!/usr/bin/env python3
"""Formal raw-vs-final evaluator for EDGE/ChoreoRAG motions.

Drop-in replacement for ``scripts/eval_generated_motions.py``.

Formal mode intentionally refuses to evaluate only a post-anchor/final motion.
It reports:
    metrics_raw
    metrics_final
    delta_final_minus_raw

Important metric naming
-----------------------
``foot_slide_proxy_rate`` is a contact-root-speed proxy computed from the 151D
representation.  It is not true FK foot sliding.  ``foot_slide_rate`` is kept as
a backwards-compatible alias but formal reports should cite the proxy name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROOT_XZ_IDX = [ROOT_X_IDX, ROOT_Z_IDX]
ROT_SLICE = slice(7, 151)
UPPER_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def _rot_indices(joints):
    idx = []
    for j in joints:
        idx.extend(range(7 + 6 * int(j), 7 + 6 * int(j) + 6))
    return np.asarray(idx, dtype=np.int64)


UPPER_IDX = _rot_indices(UPPER_JOINTS)
LOWER_IDX = _rot_indices(LOWER_JOINTS)


def load_motion(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        for key in ("motion", "motion_final", "pose", "poses"):
            if key in d:
                arr = d[key]
                break
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"Expected [T,151] motion from {path}, got {arr.shape}")
    return arr.astype(np.float32)


def load_traj(path: str, target_len: Optional[int] = None) -> np.ndarray:
    traj = np.load(path, allow_pickle=True).astype(np.float32)
    if traj.ndim == 3 and traj.shape[0] == 1:
        traj = traj[0]
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"Expected [T,2+] target trajectory from {path}, got {traj.shape}")
    traj = traj[:, :2]
    if target_len is not None and len(traj) != target_len:
        traj = resample_traj_array(traj, target_len)
    return traj.astype(np.float32)


def resample_traj_array(traj: np.ndarray, target_len: int) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    idx = np.linspace(0, len(traj) - 1, target_len)
    x = np.interp(idx, np.arange(len(traj)), traj[:, 0])
    z = np.interp(idx, np.arange(len(traj)), traj[:, 1])
    return np.stack([x, z], axis=-1).astype(np.float32)


def _safe_float(x) -> float:
    x = float(x)
    if not np.isfinite(x):
        return 0.0
    return x


def trajectory_metrics(motion: np.ndarray, target_traj: Optional[np.ndarray]) -> Dict[str, float]:
    if target_traj is None:
        return {}
    traj = np.asarray(target_traj, dtype=np.float32)
    if len(traj) != len(motion):
        traj = resample_traj_array(traj, len(motion))
    root = motion[:, ROOT_XZ_IDX]
    err = np.linalg.norm(root - traj[:, :2], axis=-1)
    root_path = float(np.linalg.norm(root[1:] - root[:-1], axis=-1).sum()) if len(root) > 1 else 0.0
    traj_path = float(np.linalg.norm(traj[1:] - traj[:-1], axis=-1).sum()) if len(traj) > 1 else 0.0
    return {
        "trajectory_ade_m": _safe_float(err.mean()),
        "trajectory_fde_m": _safe_float(err[-1]),
        "trajectory_error_p95_m": _safe_float(np.percentile(err, 95)),
        "trajectory_path_len_gen_m": _safe_float(root_path),
        "trajectory_path_len_target_m": _safe_float(traj_path),
        "trajectory_path_len_ratio": _safe_float(root_path / max(traj_path, 1e-8)),
    }


def motion_activity_stats(motion: np.ndarray) -> Dict[str, float]:
    if len(motion) < 2:
        return {
            "motion_energy": 0.0,
            "upper_activity": 0.0,
            "lower_activity": 0.0,
            "root_speed_mean": 0.0,
            "root_speed_p95": 0.0,
            "spatial_range_m": 0.0,
            "turning_mean": 0.0,
        }
    diff = motion[1:] - motion[:-1]
    root = motion[:, ROOT_XZ_IDX]
    root_vel = root[1:] - root[:-1]
    root_speed = np.linalg.norm(root_vel, axis=1) if len(root_vel) else np.zeros((0,), dtype=np.float32)

    if len(root_vel) >= 2:
        v1, v2 = root_vel[:-1], root_vel[1:]
        n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
        cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-8, None)
        turning = float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))
    else:
        turning = 0.0

    return {
        "motion_energy": _safe_float(np.sqrt(np.mean(diff[:, ROT_SLICE] ** 2))),
        "upper_activity": _safe_float(np.sqrt(np.mean(diff[:, UPPER_IDX] ** 2))),
        "lower_activity": _safe_float(np.sqrt(np.mean(diff[:, LOWER_IDX] ** 2))),
        "root_speed_mean": _safe_float(root_speed.mean() if len(root_speed) else 0.0),
        "root_speed_p95": _safe_float(np.percentile(root_speed, 95) if len(root_speed) else 0.0),
        "spatial_range_m": _safe_float(np.linalg.norm(root.max(axis=0) - root.min(axis=0))),
        "turning_mean": _safe_float(max(0.0, turning)),
    }


def jerk_metrics(motion: np.ndarray) -> Dict[str, float]:
    if len(motion) < 4:
        return {"transition_jerk": 0.0, "rot_acc_mean": 0.0, "rot_acc_p95": 0.0, "root_acc_mean": 0.0}
    rot = motion[:, ROT_SLICE]
    rot_acc = rot[2:] - 2.0 * rot[1:-1] + rot[:-2]
    rot_acc_norm = np.linalg.norm(rot_acc, axis=-1)
    root = motion[:, ROOT_XZ_IDX]
    root_acc = root[2:] - 2.0 * root[1:-1] + root[:-2]
    root_acc_norm = np.linalg.norm(root_acc, axis=-1)
    return {
        "transition_jerk": _safe_float(rot_acc_norm.mean()),
        "rot_acc_mean": _safe_float(rot_acc_norm.mean()),
        "rot_acc_p95": _safe_float(np.percentile(rot_acc_norm, 95)),
        "root_acc_mean": _safe_float(root_acc_norm.mean()),
        "root_acc_p95": _safe_float(np.percentile(root_acc_norm, 95)),
    }


def contact_metrics(motion: np.ndarray) -> Dict[str, float]:
    contacts = (motion[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    if len(contacts) < 2:
        return {"contact_phase_break": 0.0, "contact_change_rate": 0.0, "contact_occupancy": float(contacts.mean())}
    changes = np.abs(contacts[1:] - contacts[:-1]).mean(axis=1)
    return {
        "contact_phase_break": _safe_float(changes.mean()),
        "contact_change_rate": _safe_float(changes.mean()),
        "contact_change_p95": _safe_float(np.percentile(changes, 95)),
        "contact_occupancy": _safe_float(contacts.mean()),
    }


def foot_slide_proxy(motion: np.ndarray) -> Dict[str, object]:
    """Contact-root-speed proxy, not true FK foot slide."""
    if len(motion) < 2:
        return {
            "foot_slide_proxy_rate": 0.0,
            "foot_contact_root_speed_p95_mps": 0.0,
            "foot_slide_metric_type": "contact_root_speed_proxy",
        }
    contacts = (motion[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    root = motion[:, ROOT_XZ_IDX]
    speed = np.zeros((len(motion),), dtype=np.float32)
    speed[1:] = np.linalg.norm(root[1:] - root[:-1], axis=-1)
    contact_any = contacts.mean(axis=1) > 0.25
    threshold = float(np.percentile(speed, 75)) if np.any(speed > 0) else 1e-6
    threshold = max(threshold, 1e-6)
    slide = contact_any & (speed > threshold)
    contact_speed = speed[contact_any]
    rate = _safe_float(slide.mean())
    out = {
        "foot_slide_proxy_rate": rate,
        "foot_contact_root_speed_p95_mps": _safe_float(np.percentile(contact_speed, 95) if len(contact_speed) else 0.0),
        "foot_slide_speed_threshold": _safe_float(threshold),
        "foot_slide_metric_type": "contact_root_speed_proxy",
        # Backward-compatible aliases.  Prefer foot_slide_proxy_rate in reports.
        "foot_slide_rate": rate,
        "foot_contact_speed_p95_mps": _safe_float(np.percentile(contact_speed, 95) if len(contact_speed) else 0.0),
    }
    return out


def root_lower_sync(motion: np.ndarray) -> Dict[str, float]:
    if len(motion) < 2:
        return {"root_lower_sync": 0.0, "root_lower_sync_deficit": 0.0}
    diff = motion[1:] - motion[:-1]
    lower = np.sqrt(np.mean(diff[:, LOWER_IDX] ** 2, axis=1)) if LOWER_IDX.size else np.zeros((len(diff),), dtype=np.float32)
    root_speed = np.linalg.norm(motion[1:, ROOT_XZ_IDX] - motion[:-1, ROOT_XZ_IDX], axis=1)
    gate = root_speed > max(1e-6, float(np.percentile(root_speed, 50)))
    if not np.any(gate):
        return {"root_lower_sync": 1.0, "root_lower_sync_deficit": 0.0}
    lower_active = lower[gate]
    root_active = root_speed[gate]
    target = 0.30 * root_active + 1e-6
    deficit = np.maximum(0.0, target - lower_active) / target
    return {
        "root_lower_sync": _safe_float(1.0 - np.clip(deficit.mean(), 0.0, 1.0)),
        "root_lower_sync_deficit": _safe_float(deficit.mean()),
        "root_lower_active_frames": int(gate.sum()),
    }


def freezing_score(motion: np.ndarray, threshold: float = 0.015) -> float:
    if len(motion) < 2:
        return 1.0
    v = np.linalg.norm(motion[1:, ROT_SLICE] - motion[:-1, ROT_SLICE], axis=-1)
    return _safe_float((v < threshold).mean())


def compute_metrics(motion: np.ndarray, target_traj: Optional[np.ndarray], stage: str) -> Dict[str, object]:
    out: Dict[str, object] = {"eval_stage": stage, "num_frames": int(len(motion))}
    out.update(trajectory_metrics(motion, target_traj))
    out.update(motion_activity_stats(motion))
    out.update(jerk_metrics(motion))
    out.update(contact_metrics(motion))
    out.update(foot_slide_proxy(motion))
    out.update(root_lower_sync(motion))
    out["freezing_score"] = freezing_score(motion)
    return out


def numeric_delta(final: Dict[str, object], raw: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, fval in final.items():
        rval = raw.get(key)
        if isinstance(fval, (int, float)) and isinstance(rval, (int, float)):
            out[key] = _safe_float(float(fval) - float(rval))
    return out


def build_warnings(metrics_raw: Dict[str, object], metrics_final: Dict[str, object]) -> List[str]:
    warnings: List[str] = []
    final_ade = float(metrics_final.get("trajectory_ade_m", 999.0))
    raw_ade = float(metrics_raw.get("trajectory_ade_m", 999.0))
    final_slide = float(metrics_final.get("foot_slide_proxy_rate", metrics_final.get("foot_slide_rate", 0.0)))
    raw_slide = float(metrics_raw.get("foot_slide_proxy_rate", metrics_raw.get("foot_slide_rate", 0.0)))
    final_sync = float(metrics_final.get("root_lower_sync", 0.0))
    raw_sync = float(metrics_raw.get("root_lower_sync", 0.0))

    if final_ade <= 1e-6 and raw_ade > 1e-3:
        warnings.append("final ADE is zero while raw ADE is non-zero; this is post-anchor system control, not native trajectory ability.")
    if final_slide > max(raw_slide * 1.5, raw_slide + 0.05):
        warnings.append("foot_slide_proxy_rate increased after final/post-anchor motion; trajectory may be achieved by dragging.")
    if final_sync + 0.10 < raw_sync:
        warnings.append("root_lower_sync dropped after final/post-anchor motion; lower body may not follow root naturally.")
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_motion", required=True, help="Native/raw model motion, e.g. *_raw.npy")
    ap.add_argument("--final_motion", required=True, help="Final/post-anchor motion, e.g. *.npy or *_anchor.npy")
    ap.add_argument("--target_traj", default="", help="Target trajectory [T,2]")
    ap.add_argument("--meta", default="", help="Optional generation meta JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--formal", action="store_true", help="Reject single-stage or suspicious evaluation usage")
    ap.add_argument("--allow_same_file", action="store_true")
    args = ap.parse_args()

    raw_path = Path(args.raw_motion)
    final_path = Path(args.final_motion)
    if args.formal:
        if not raw_path.exists() or not final_path.exists():
            raise ValueError("Formal evaluation requires both --raw_motion and --final_motion files.")
        if raw_path.resolve() == final_path.resolve() and not args.allow_same_file:
            raise ValueError("Formal evaluation requires distinct raw and final motion files.")

    raw = load_motion(str(raw_path))
    final = load_motion(str(final_path))
    if len(raw) != len(final):
        raise ValueError(f"raw/final length mismatch: raw={len(raw)}, final={len(final)}")

    target = load_traj(args.target_traj, target_len=len(raw)) if args.target_traj else None
    metrics_raw = compute_metrics(raw, target, stage="raw_native")
    metrics_final = compute_metrics(final, target, stage="post_anchor_or_final")
    delta = numeric_delta(metrics_final, metrics_raw)
    warnings = build_warnings(metrics_raw, metrics_final)

    payload: Dict[str, object] = {
        "raw_motion": str(raw_path),
        "final_motion": str(final_path),
        "target_traj": str(args.target_traj or ""),
        "metrics_raw": metrics_raw,
        "metrics_final": metrics_final,
        "delta_final_minus_raw": delta,
        "warnings": warnings,
        "formal": bool(args.formal),
        "metric_notes": {
            "foot_slide_proxy_rate": "contact-root-speed proxy from 151D representation, not true FK foot sliding",
            "foot_slide_rate": "backward-compatible alias of foot_slide_proxy_rate",
        },
    }

    if args.meta:
        try:
            with open(args.meta, "r", encoding="utf-8") as f:
                payload["generation_meta"] = json.load(f)
        except Exception as exc:
            payload["generation_meta_error"] = str(exc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Formal raw/final evaluation saved: {out}")
    if warnings:
        print("⚠️ warnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
