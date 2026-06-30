#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V41.2 FK-verified native-floor metadata injector for EDGE Event-RAG.

This script fixes the 151D index-alignment trap.

IMPORTANT:
  [T,151] in this EDGE pipeline is not a simple joint-position table.
  The layout used by V21/V34/V40 is:
      root channels around [4,5,6]
      24 local joint rotations in 6D at [7:151]
  Therefore raw channels such as x[:, 7] or x[:, 10] are rotation components,
  not foot-Y positions.

The only safe path is:
      rot6d -> FK -> joints[:, foot_joint_ids, 1] -> relative native penetration

This script injects FK-derived fields into JSON and keeps JSON/NPZ length
unchanged. It never deletes events.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
OFFSETS = np.array(
    [
        [0.00, 0.00, 0.00], [-0.10, -0.10, 0.00], [0.10, -0.10, 0.00], [0.00, 0.13, 0.00],
        [0.00, -0.42, 0.00], [0.00, -0.42, 0.00], [0.00, 0.14, 0.00], [0.00, -0.40, 0.00],
        [0.00, -0.40, 0.00], [0.00, 0.14, 0.00], [0.00, -0.08, 0.12], [0.00, -0.08, 0.12],
        [0.00, 0.14, 0.00], [-0.10, 0.08, 0.00], [0.10, 0.08, 0.00], [0.00, 0.16, 0.00],
        [-0.18, 0.00, 0.00], [0.18, 0.00, 0.00], [-0.28, 0.00, 0.00], [0.28, 0.00, 0.00],
        [-0.25, 0.00, 0.00], [0.25, 0.00, 0.00], [-0.08, 0.00, 0.00], [0.08, 0.00, 0.00],
    ],
    dtype=np.float32,
)

# FK joint ids, NOT raw feature channel ids.
# In this local skeleton these correspond to lower leg/foot/toe endpoints used
# by the existing V40 native-floor audit and V34 retrieval FK utilities.
DEFAULT_FOOT_JOINTS = (7, 8, 10, 11)


def _rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _items_of(obj: Any) -> Tuple[list, Optional[str]]:
    if isinstance(obj, list):
        return obj, None
    for key in ("items", "events", "index", "data"):
        if isinstance(obj, dict) and isinstance(obj.get(key), list):
            return obj[key], key
    raise ValueError("Cannot locate event item list in JSON. Expected list or dict with items/events/index/data.")


def _find_motion_path(item: Dict[str, Any], roots: Sequence[Path]) -> Optional[Path]:
    keys = ("motion_path", "npy_path", "path", "file", "motion_file", "event_path", "source_path", "clip_path")
    for key in keys:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            continue
        p = Path(value)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        for root in roots:
            candidates.append(root / p)
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    # Fallback by id-like filename.
    for key in ("event_id", "id", "uid", "name"):
        value = str(item.get(key, ""))
        if not value:
            continue
        patterns = [value, value + ".npy", value.replace("/", "_") + ".npy"]
        for root in roots:
            if not root.exists():
                continue
            for pattern in patterns:
                hits = list(root.rglob(pattern))
                if hits:
                    return hits[0]
    return None


def _load_motion(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.ndim == 0 and isinstance(obj.item(), dict):
        d = obj.item()
        obj = d.get("motion", d.get("pose", d.get("arr_0", obj)))
    x = np.asarray(obj, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] < 151:
        raise ValueError(f"{path}: expected [T,151], got {x.shape}")
    return x[:, :151]


def _fk_24(motion: np.ndarray) -> np.ndarray:
    """Forward-kinematics from EDGE [T,151] to [T,24,3] joint positions."""
    t = int(motion.shape[0])
    root = motion[:, [4, 5, 6]]
    local_r = _rot6d_to_matrix(motion[:, 7:151].reshape(t, 24, 6))
    joints = np.zeros((t, 24, 3), dtype=np.float32)
    global_r = np.zeros((t, 24, 3, 3), dtype=np.float32)
    joints[:, 0] = root
    global_r[:, 0] = local_r[:, 0]
    for j in range(1, 24):
        parent = int(PARENTS[j])
        global_r[:, j] = np.matmul(global_r[:, parent], local_r[:, j])
        joints[:, j] = joints[:, parent] + np.matmul(
            global_r[:, parent],
            OFFSETS[j][None, :, None],
        )[..., 0]
    return joints


def _barrier(pen: float, safe: float, dead: float, alpha: float, beta: float, cap: float) -> Tuple[float, bool, float]:
    if pen <= safe:
        return 0.0, False, 0.0
    hard = bool(pen >= dead)
    ratio = max(0.0, (pen - safe) / max(dead - safe, 1e-8))
    value = float(alpha * (ratio ** beta))
    if cap > 0:
        value = min(value, float(cap))
    return value, hard, float(ratio)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--audit_json", default="")
    parser.add_argument("--search_root", action="append", default=[])
    parser.add_argument("--quantile", type=float, default=float(os.getenv("V41_NATIVE_FLOOR_QUANTILE", "0.05")))
    parser.add_argument("--margin", type=float, default=float(os.getenv("V41_NATIVE_FLOOR_MARGIN", "0.006")))
    parser.add_argument("--tau_safe", type=float, default=float(os.getenv("V41_NATIVE_FLOOR_TAU_SAFE_M", "0.012")))
    parser.add_argument("--tau_dead", type=float, default=float(os.getenv("V41_NATIVE_FLOOR_TAU_DEAD_M", "0.052")))
    parser.add_argument("--alpha", type=float, default=float(os.getenv("V41_NATIVE_FLOOR_ALPHA", "9.0")))
    parser.add_argument("--beta", type=float, default=float(os.getenv("V41_NATIVE_FLOOR_BETA", "2.5")))
    parser.add_argument("--cap", type=float, default=float(os.getenv("V41_NATIVE_FLOOR_PENALTY_CAP", "18.0")))
    parser.add_argument("--foot_joints", default=os.getenv("V41_NATIVE_FLOOR_FOOT_JOINTS", "7,8,10,11"))
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--max_missing_fraction", type=float, default=float(os.getenv("V41_NATIVE_FLOOR_MAX_MISSING_FRACTION", "0.01")))
    parser.add_argument("--fail_on_missing_fraction", action="store_true", default=os.getenv("V41_NATIVE_FLOOR_FAIL_ON_MISSING", "1").lower() in {"1", "true", "yes", "on"})
    args = parser.parse_args()

    foot_joints = tuple(int(x.strip()) for x in args.foot_joints.split(",") if x.strip())
    if not foot_joints:
        raise ValueError("--foot_joints is empty")
    if any(j < 0 or j >= 24 for j in foot_joints):
        raise ValueError(f"--foot_joints must be FK joint ids in [0,23], got {foot_joints}")

    index_path = Path(args.index_json)
    obj = json.loads(index_path.read_text(encoding="utf-8"))
    items, list_key = _items_of(obj)
    roots = [Path(p) for p in args.search_root] + [index_path.parent, Path("."), Path("data"), Path("output")]

    rows = []
    new_items = []
    scored = reused = missing = failed = 0
    min_values: list[float] = []
    floor_values: list[float] = []
    pen_values: list[float] = []
    potential_values: list[float] = []

    for i, item in enumerate(items):
        row = dict(item)
        stat: Dict[str, Any] = {
            "index": i,
            "event_id": str(row.get("event_id", row.get("id", i))),
        }

        if (not args.overwrite_existing) and "native_floor_penetration_m" in row:
            reused += 1
            pen = float(row.get("native_floor_penetration_m", 0.0) or 0.0)
            min_y = float(row.get("native_min_foot_y", row.get("min_foot_y", 0.0)) or 0.0)
            floor_y = float(row.get("native_floor_y", 0.0) or 0.0)
            pot, hard, ratio = _barrier(pen, args.tau_safe, args.tau_dead, args.alpha, args.beta, args.cap)
            row.update({
                "native_min_foot_y": min_y,
                "min_foot_y": min_y,
                "native_floor_y": floor_y,
                "native_floor_penetration_m": pen,
                "v41_native_floor_barrier_potential": float(pot),
                "v41_native_floor_hard_mask": bool(hard),
                "v41_native_floor_violation_ratio": float(ratio),
                "v41_native_floor_available": True,
                "v41_native_floor_extraction": "json_reused",
                "v41_native_floor_foot_joint_ids": list(foot_joints),
                "v41_native_floor_raw_column_mode": False,
            })
            stat.update({"status": "reused", **{k: row[k] for k in (
                "native_min_foot_y", "native_floor_y", "native_floor_penetration_m",
                "v41_native_floor_barrier_potential", "v41_native_floor_hard_mask",
                "v41_native_floor_violation_ratio",
            )}})
            min_values.append(min_y); floor_values.append(floor_y); pen_values.append(pen); potential_values.append(float(pot))
            rows.append(stat); new_items.append(row)
            continue

        motion_path = _find_motion_path(row, roots)
        stat["motion_path"] = str(motion_path) if motion_path else None
        if motion_path is None:
            missing += 1
            row.update({
                "v41_native_floor_available": False,
                "v41_native_floor_extraction": "missing_motion_path",
                "v41_native_floor_raw_column_mode": False,
            })
            stat["status"] = "missing"
        else:
            try:
                motion = _load_motion(motion_path)
                joints = _fk_24(motion)
                foot_y = joints[:, foot_joints, 1].reshape(-1)
                floor_y = float(np.quantile(foot_y, args.quantile))
                min_y = float(np.min(foot_y))
                max_y = float(np.max(foot_y))
                mean_y = float(np.mean(foot_y))
                pen = max(0.0, floor_y + float(args.margin) - min_y)
                pot, hard, ratio = _barrier(pen, args.tau_safe, args.tau_dead, args.alpha, args.beta, args.cap)

                fields = {
                    "native_min_foot_y": min_y,
                    "min_foot_y": min_y,
                    "native_floor_y": floor_y,
                    "native_floor_penetration_m": float(pen),
                    "v41_native_floor_barrier_potential": float(pot),
                    "v41_native_floor_hard_mask": bool(hard),
                    "v41_native_floor_violation_ratio": float(ratio),
                    "v41_native_floor_available": True,
                    "v41_native_floor_extraction": "fk_24_from_rot6d",
                    "v41_native_floor_foot_joint_ids": list(foot_joints),
                    "v41_native_floor_raw_column_mode": False,
                    "v41_native_foot_y_max": max_y,
                    "v41_native_foot_y_mean": mean_y,
                }
                row.update(fields)
                stat.update({"status": "ok", **fields})
                scored += 1
                min_values.append(min_y); floor_values.append(floor_y); pen_values.append(float(pen)); potential_values.append(float(pot))
            except Exception as exc:
                failed += 1
                row.update({
                    "v41_native_floor_available": False,
                    "v41_native_floor_extraction": "failed_fk",
                    "v41_native_floor_error": repr(exc),
                    "v41_native_floor_raw_column_mode": False,
                })
                stat.update({"status": "failed", "error": repr(exc)})

        rows.append(stat)
        new_items.append(row)

    if isinstance(obj, list):
        out_obj = new_items
    else:
        if list_key is None:
            raise ValueError("Cannot preserve JSON structure because item list key is unknown")
        out_obj = dict(obj)
        out_obj[list_key] = new_items

    missing_fraction = float((missing + failed) / max(len(items), 1))
    summary = {
        "version": "v41_2_fk_verified_native_floor_metadata_injector",
        "input": str(index_path),
        "output": str(args.out_json),
        "num_items": len(items),
        "scored": int(scored),
        "reused": int(reused),
        "missing": int(missing),
        "failed": int(failed),
        "missing_fraction": float(missing_fraction),
        "max_missing_fraction": float(args.max_missing_fraction),
        "quantile": float(args.quantile),
        "margin": float(args.margin),
        "tau_safe": float(args.tau_safe),
        "tau_dead": float(args.tau_dead),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "cap": float(args.cap),
        "foot_joint_ids": list(foot_joints),
        "raw_column_mode": False,
        "extraction": "FK(root[4:7] + rot6d[7:151] -> joints[7,8,10,11].Y)",
        "hard_mask_count": int(sum(bool(r.get("v41_native_floor_hard_mask", False)) for r in rows)),
        "soft_penalty_count": int(sum(float(r.get("v41_native_floor_barrier_potential", 0.0)) > 0 for r in rows)),
    }
    if pen_values:
        summary.update({
            "native_min_foot_y_min": float(min(min_values)),
            "native_min_foot_y_p05": _percentile(min_values, 5),
            "native_min_foot_y_median": _percentile(min_values, 50),
            "native_min_foot_y_p95": _percentile(min_values, 95),
            "native_floor_y_median": _percentile(floor_values, 50),
            "pen_max": float(max(pen_values)),
            "pen_mean": float(np.mean(np.asarray(pen_values, dtype=np.float32))),
            "pen_p95": _percentile(pen_values, 95),
            "potential_max": float(max(potential_values)) if potential_values else 0.0,
            "potential_mean": float(np.mean(np.asarray(potential_values, dtype=np.float32))) if potential_values else 0.0,
        })

    if args.fail_on_missing_fraction and missing_fraction > args.max_missing_fraction:
        # Write audit before failing so the user can inspect missing paths.
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.audit_json:
            Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.audit_json).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Native-floor FK extraction missing_fraction={missing_fraction:.4f} exceeds "
            f"max_missing_fraction={args.max_missing_fraction:.4f}. "
            "Do not continue with phantom-safe defaults; inspect audit_json."
        )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.audit_json:
        Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_json).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
