#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V40B native-floor source pruning for V34 Event-RAG.

Physically remove native floor-toxic snippets from the JSON/NPZ candidate
index before V34 graph planning, forcing the planner to reroute.

Important: EDGE/SMPL foot Y is not zero-centered.  Do not prune by raw absolute
Y by default.  V40B uses event-local floor penetration:

    floor_y = quantile(all lower-foot Y, q)
    native_penetration = max(0, floor_y + margin - min(lower-foot Y))

Default hard remove threshold: 0.08 m.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21], dtype=np.int64)
OFFSETS = np.array([
    [0.00,0.00,0.00],[-0.10,-0.10,0.00],[0.10,-0.10,0.00],[0.00,0.13,0.00],
    [0.00,-0.42,0.00],[0.00,-0.42,0.00],[0.00,0.14,0.00],[0.00,-0.40,0.00],
    [0.00,-0.40,0.00],[0.00,0.14,0.00],[0.00,-0.08,0.12],[0.00,-0.08,0.12],
    [0.00,0.14,0.00],[-0.10,0.08,0.00],[0.10,0.08,0.00],[0.00,0.16,0.00],
    [-0.18,0.00,0.00],[0.18,0.00,0.00],[-0.28,0.00,0.00],[0.28,0.00,0.00],
    [-0.25,0.00,0.00],[0.25,0.00,0.00],[-0.08,0.00,0.00],[0.08,0.00,0.00],
], dtype=np.float32)
FOOT_JOINTS = np.array([7, 8, 10, 11], dtype=np.int64)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _item_list_ref(obj: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if isinstance(obj, list):
        return obj, None
    if isinstance(obj, dict):
        for key in ("items", "events", "index", "data"):
            value = obj.get(key)
            if isinstance(value, list):
                return value, key
    raise ValueError("Cannot locate event item list. Expected list or dict with items/events/index/data.")


def _rot6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _fk_joints(motion: np.ndarray) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] < 151:
        raise ValueError(f"expected [T,151+] motion, got {x.shape}")
    x = x[:, :151]
    t = x.shape[0]
    root = x[:, [4, 5, 6]]
    local = _rot6d_to_matrix_np(x[:, 7:151].reshape(t, 24, 6))
    joints = np.zeros((t, 24, 3), dtype=np.float32)
    global_r = np.zeros((t, 24, 3, 3), dtype=np.float32)
    joints[:, 0] = root
    global_r[:, 0] = local[:, 0]
    for j in range(1, 24):
        p = int(PARENTS[j])
        global_r[:, j] = np.matmul(global_r[:, p], local[:, j])
        joints[:, j] = joints[:, p] + np.matmul(global_r[:, p], OFFSETS[j][None, :, None])[..., 0]
    return joints


def _load_motion_from_path(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        candidate = None
        for key in ("motion", "pose", "arr_0", "motions"):
            if key in arr:
                candidate = arr[key]
                break
        if candidate is None:
            for key in arr.files:
                value = arr[key]
                if isinstance(value, np.ndarray) and value.ndim in (2, 3) and value.shape[-1] >= 151:
                    candidate = value
                    break
        if candidate is None:
            raise ValueError(f"{path}: no motion-like array in npz")
        x = candidate
    else:
        x = arr
    if isinstance(x, np.ndarray) and x.ndim == 0 and isinstance(x.item(), dict):
        d = x.item()
        x = d.get("motion", d.get("pose", d.get("arr_0", x)))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] < 151:
        raise ValueError(f"{path}: expected [T,151+], got {x.shape}")
    return x[:, :151]


def _find_motion_path(item: Mapping[str, Any], roots: Sequence[Path]) -> Optional[Path]:
    keys = ("motion_path", "npy_path", "path", "file", "motion_file", "event_path", "source_path", "clip_path", "mirrored_motion_path", "v34_motion_path")
    values: List[str] = []
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    for raw in values:
        p = Path(raw)
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
        for root in roots:
            candidates.append(root / p)
            candidates.append(root / p.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    ids: List[str] = []
    for key in ("uid", "event_id", "id", "name", "motion_id", "source_uid"):
        value = item.get(key)
        if value is not None and str(value):
            s = str(value)
            ids.extend([s, s.replace("/", "_"), Path(s).name])
    patterns: List[str] = []
    for s in ids:
        if s.endswith((".npy", ".npz")):
            patterns.append(s)
        else:
            patterns.extend([s, s + ".npy", s + ".npz"])
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits = list(root.rglob(pat))
            if hits:
                return hits[0].resolve()
    return None


def _motion_from_npz_event_axis(npz: np.lib.npyio.NpzFile, index: int, n_items: int) -> Optional[np.ndarray]:
    for key in ("motions", "motion", "clips", "events", "event_motion", "motion_array"):
        if key not in npz:
            continue
        arr = np.asarray(npz[key])
        if arr.ndim >= 3 and arr.shape[0] == n_items and arr.shape[-1] >= 151:
            return np.asarray(arr[index], dtype=np.float32)[..., :151]
        if arr.dtype == object and arr.ndim == 1 and len(arr) == n_items:
            try:
                x = np.asarray(arr[index], dtype=np.float32)
                if x.ndim == 2 and x.shape[1] >= 151:
                    return x[:, :151]
            except Exception:
                pass
    return None


def _event_identity(item: Mapping[str, Any], index: int) -> str:
    for key in ("uid", "event_id", "id", "name", "motion_id", "source_uid"):
        value = item.get(key)
        if value is not None and str(value):
            return str(value)
    return f"event_{index:06d}"


def audit_motion(motion: np.ndarray, *, quantile: float, margin: float, tolerance: float, penalty_weight: float) -> Dict[str, Any]:
    joints = _fk_joints(motion)
    foot_y = joints[:, FOOT_JOINTS, 1].astype(np.float32)
    flat = foot_y.reshape(-1)
    q = float(np.clip(quantile, 0.001, 0.499))
    floor_y = float(np.quantile(flat, q))
    min_y = float(np.min(flat))
    p01_y = float(np.quantile(flat, 0.01))
    median_y = float(np.quantile(flat, 0.50))
    max_y = float(np.max(flat))
    native_pen = float(max(0.0, floor_y + float(margin) - min_y))
    excess = float(max(0.0, native_pen - float(tolerance)))
    return {
        "native_floor_available": True,
        "native_min_foot_y": min_y,
        "native_p01_foot_y": p01_y,
        "native_median_foot_y": median_y,
        "native_max_foot_y": max_y,
        "native_floor_y": floor_y,
        "native_floor_margin_m": float(margin),
        "native_floor_penetration_m": native_pen,
        "native_floor_excess_m": excess,
        "native_floor_penalty": float(penalty_weight * excess * excess),
        "native_floor_ok": bool(native_pen <= float(tolerance)),
        "native_absolute_min_foot_y": min_y,
    }


def _quality_adjust(row: Dict[str, Any], penalty: float) -> None:
    for key in ("quality_score", "quality", "score_quality"):
        if key in row:
            try:
                row[key] = float(row[key]) - float(penalty)
                row["v40b_native_floor_quality_adjusted"] = True
            except Exception:
                pass


def _filter_npz(in_npz: Path, out_npz: Path, keep_mask: np.ndarray, n_items: int) -> Dict[str, Any]:
    arrays = np.load(in_npz, allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    filtered_keys: List[str] = []
    unfiltered_keys: List[str] = []
    for key in arrays.files:
        value = arrays[key]
        if hasattr(value, "shape") and len(value.shape) >= 1 and int(value.shape[0]) == int(n_items):
            out[key] = value[keep_mask]
            filtered_keys.append(key)
        else:
            out[key] = value
            unfiltered_keys.append(key)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out)
    return {"filtered_keys": filtered_keys, "unfiltered_keys": unfiltered_keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--index_npz", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--audit_json", required=True)
    ap.add_argument("--removed_txt", default="")
    ap.add_argument("--search_root", action="append", default=[])
    ap.add_argument("--quantile", type=float, default=_env_float("V40B_NATIVE_FLOOR_QUANTILE", 0.05))
    ap.add_argument("--margin", type=float, default=_env_float("V40B_NATIVE_FLOOR_MARGIN", 0.006))
    ap.add_argument("--tolerance", type=float, default=_env_float("V40B_NATIVE_FLOOR_TOLERANCE_M", 0.04))
    ap.add_argument("--soft_threshold", type=float, default=_env_float("V40B_NATIVE_FLOOR_SOFT_THRESHOLD", 0.055))
    ap.add_argument("--remove_threshold", type=float, default=_env_float("V40B_NATIVE_FLOOR_REMOVE_THRESHOLD", 0.08))
    ap.add_argument("--penalty_weight", type=float, default=_env_float("V40B_NATIVE_FLOOR_PENALTY_WEIGHT", 18.0))
    ap.add_argument("--max_remove_fraction", type=float, default=_env_float("V40B_MAX_REMOVE_FRACTION", 0.35))
    ap.add_argument("--min_remaining", type=int, default=_env_int("V40B_MIN_REMAINING_EVENTS", 128))
    ap.add_argument("--mode", choices=["remove", "quality", "mark"], default=os.getenv("V40B_NATIVE_FLOOR_MODE", "remove"))
    ap.add_argument("--missing_policy", choices=["keep", "remove"], default=os.getenv("V40B_MISSING_POLICY", "keep"))
    ap.add_argument("--absolute_y_threshold", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    index_json = Path(args.index_json).resolve()
    index_npz = Path(args.index_npz).resolve()
    out_json = Path(args.out_json).resolve()
    out_npz = Path(args.out_npz).resolve()
    audit_json = Path(args.audit_json).resolve()

    obj = _load_json(index_json)
    items, item_key = _item_list_ref(obj)
    n = len(items)
    if n == 0:
        raise RuntimeError("No event items found.")
    arrays = np.load(index_npz, allow_pickle=True)
    for key in ("motion_desc", "mmr_embed", "entry_pose", "exit_pose", "entry_vel", "exit_vel", "length"):
        if key in arrays and len(arrays[key]) != n:
            raise RuntimeError(f"Index mismatch before pruning: {key} has {len(arrays[key])}, JSON has {n}")

    roots: List[Path] = []
    for raw in args.search_root:
        for token in str(raw).split(":"):
            if token.strip():
                roots.append(Path(token.strip()).expanduser())
    roots.extend([index_json.parent, Path.cwd(), Path("data"), Path("output")])

    keep_mask = np.zeros((n,), dtype=bool)
    new_items: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    counters = {"scored": 0, "missing": 0, "failed": 0, "removed": 0, "soft_flagged": 0, "quality_adjusted": 0}

    for i, old_item in enumerate(items):
        item = dict(old_item)
        row: Dict[str, Any] = {"index": int(i), "identity": _event_identity(item, i), "removed": False, "remove_reason": ""}
        motion = None
        mp = _find_motion_path(item, roots)
        if mp is not None:
            row["motion_path"] = str(mp)
            try:
                motion = _load_motion_from_path(mp)
            except Exception as exc:
                row["motion_path_load_error"] = repr(exc)
        if motion is None:
            try:
                motion = _motion_from_npz_event_axis(arrays, i, n)
                if motion is not None:
                    row["motion_path"] = "__npz_event_axis__"
            except Exception as exc:
                row["npz_motion_error"] = repr(exc)
        if motion is None:
            counters["missing"] += 1
            item["native_floor_available"] = False
            row["status"] = "missing_motion"
            if args.missing_policy == "remove":
                row["removed"] = True
                row["remove_reason"] = "missing_motion"
                counters["removed"] += 1
            else:
                keep_mask[i] = True
                new_items.append(item)
            rows.append(row)
            continue
        try:
            metrics = audit_motion(motion, quantile=args.quantile, margin=args.margin, tolerance=args.tolerance, penalty_weight=args.penalty_weight)
            item.update(metrics)
            row.update(metrics)
            row["status"] = "ok"
            counters["scored"] += 1
            pen = float(metrics["native_floor_penetration_m"])
            if pen > float(args.soft_threshold):
                item["v40b_native_floor_soft_flag"] = True
                counters["soft_flagged"] += 1
            if args.mode == "quality":
                _quality_adjust(item, float(metrics["native_floor_penalty"]))
                counters["quality_adjusted"] += int(bool(item.get("v40b_native_floor_quality_adjusted", False)))
            should_remove = False
            reason = ""
            if args.mode == "remove" and pen > float(args.remove_threshold):
                should_remove = True
                reason = f"native_floor_penetration_m>{float(args.remove_threshold):.4f}"
            if args.absolute_y_threshold is not None and float(metrics["native_absolute_min_foot_y"]) < float(args.absolute_y_threshold):
                should_remove = True
                reason = f"absolute_min_foot_y<{float(args.absolute_y_threshold):.4f}"
            if should_remove:
                row["removed"] = True
                row["remove_reason"] = reason
                counters["removed"] += 1
            else:
                keep_mask[i] = True
                new_items.append(item)
        except Exception as exc:
            counters["failed"] += 1
            item["native_floor_available"] = False
            row["status"] = "failed"
            row["error"] = repr(exc)
            if args.missing_policy == "remove":
                row["removed"] = True
                row["remove_reason"] = "audit_failed"
                counters["removed"] += 1
            else:
                keep_mask[i] = True
                new_items.append(item)
        rows.append(row)

    num_after = int(np.sum(keep_mask))
    remove_fraction = float(1.0 - num_after / max(n, 1))
    if not args.force:
        if num_after < int(args.min_remaining):
            raise RuntimeError(f"V40B safety stop: only {num_after} events remain; min_remaining={args.min_remaining}")
        if remove_fraction > float(args.max_remove_fraction):
            raise RuntimeError(f"V40B safety stop: remove_fraction={remove_fraction:.3f} exceeds max_remove_fraction={args.max_remove_fraction}")

    if item_key is None:
        out_obj: Any = new_items
    else:
        out_obj = dict(obj)
        out_obj[item_key] = new_items
        out_obj["v40b_native_floor_pruning"] = {
            "enabled": True,
            "mode": args.mode,
            "num_before": int(n),
            "num_after": int(num_after),
            "removed": int(counters["removed"]),
            "remove_threshold_m": float(args.remove_threshold),
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    npz_summary = _filter_npz(index_npz, out_npz, keep_mask, n)

    values = [float(r["native_floor_penetration_m"]) for r in rows if "native_floor_penetration_m" in r]
    removed_rows = [r for r in rows if r.get("removed")]
    summary: Dict[str, Any] = {
        "version": "v40b_native_floor_source_pruning_reroute",
        "input_json": str(index_json),
        "input_npz": str(index_npz),
        "output_json": str(out_json),
        "output_npz": str(out_npz),
        "audit_json": str(audit_json),
        "mode": args.mode,
        "num_before": int(n),
        "num_after": int(num_after),
        "removed": int(counters["removed"]),
        "remove_fraction": remove_fraction,
        "scored": int(counters["scored"]),
        "missing": int(counters["missing"]),
        "failed": int(counters["failed"]),
        "soft_flagged": int(counters["soft_flagged"]),
        "quality_adjusted": int(counters["quality_adjusted"]),
        "thresholds": {
            "quantile": float(args.quantile),
            "margin": float(args.margin),
            "tolerance_m": float(args.tolerance),
            "soft_threshold_m": float(args.soft_threshold),
            "remove_threshold_m": float(args.remove_threshold),
            "absolute_y_threshold": args.absolute_y_threshold,
        },
        "npz": npz_summary,
    }
    if values:
        summary.update({
            "native_floor_penetration_max_m": float(np.max(values)),
            "native_floor_penetration_mean_m": float(np.mean(values)),
            "native_floor_penetration_p95_m": float(np.percentile(values, 95)),
            "num_over_tolerance": int(sum(v > float(args.tolerance) for v in values)),
            "num_over_soft_threshold": int(sum(v > float(args.soft_threshold) for v in values)),
            "num_over_remove_threshold": int(sum(v > float(args.remove_threshold) for v in values)),
        })
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps({"summary": summary, "removed": removed_rows, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.removed_txt:
        p = Path(args.removed_txt)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(str(r.get("identity", r.get("index"))) for r in removed_rows) + ("\n" if removed_rows else ""), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
