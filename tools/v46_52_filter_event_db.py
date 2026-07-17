#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anatomy-gate and augment a V46.50 heading Event-RAG database.

The input database is built by the unmodified V46.50 event slicer.  This tool
loads each saved event motion, rejects anatomically invalid events, appends
posture/floor/anatomy arrays, and recomputes descriptor normalization.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tools.v46_52_anatomy_contract import env_float, env_int, event_anatomy_features


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _load_event(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] < 151:
        raise ValueError(f"Invalid event motion {path}: {x.shape}")
    return x[:, :151]


def filter_database(db_path: Path, meta_path: Path, audit_path: Path) -> Dict[str, Any]:
    obj = np.load(db_path, allow_pickle=True)
    payload = {k: obj[k] for k in obj.files}
    paths = np.asarray(payload["paths"], dtype=object)
    n = len(paths)
    meta: List[Dict[str, Any]] = []
    if meta_path.is_file():
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            meta = [dict(x) for x in raw]
    if len(meta) != n:
        meta = [dict() for _ in range(n)]

    features: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    keep = np.zeros(n, dtype=bool)
    min_quality = env_float("V46_52_EVENT_ANATOMY_QUALITY_MIN", 0.48)
    max_warp_source_frames = env_int("V46_52_EVENT_MIN_FRAMES", 18)

    for i, path in enumerate(paths.tolist()):
        try:
            motion = _load_event(str(path))
            feat = event_anatomy_features(motion)
            valid = bool(feat["anatomy_valid"] and feat["anatomy_quality"] >= min_quality and len(motion) >= max_warp_source_frames)
            feat["event_index_before_filter"] = int(i)
            feat["event_path"] = str(path)
            feat["frames"] = int(len(motion))
            feat["kept"] = bool(valid)
            keep[i] = valid
            features.append(feat)
            if i < len(meta):
                meta[i].update({k: v for k, v in feat.items() if k not in {"anatomy_reasons"}})
        except Exception as exc:
            feat = {
                "event_index_before_filter": int(i),
                "event_path": str(path),
                "anatomy_valid": False,
                "anatomy_quality": 0.0,
                "anatomy_reasons": [str(exc)],
                "kept": False,
            }
            features.append(feat)
            failures.append(feat)

    kept = np.where(keep)[0]
    min_keep = env_int("V46_52_EVENT_DB_MIN_EVENTS", 64)
    min_ratio = env_float("V46_52_EVENT_DB_MIN_KEEP_RATIO", 0.45)
    if len(kept) < min_keep or len(kept) / max(1, n) < min_ratio:
        raise RuntimeError(
            f"V46.52 event gate retained {len(kept)}/{n}; "
            f"requires >= {min_keep} and ratio >= {min_ratio:.3f}"
        )

    out: Dict[str, Any] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        # Only arrays whose first dimension is exactly the event count are event-wise.
        out[key] = arr[keep] if arr.ndim >= 1 and arr.shape[0] == n else value

    kept_features = [features[i] for i in kept]
    out.update({
        "anatomy_contract_schema_version": np.asarray("v46_52_event_anatomy_contract", dtype=object),
        "anatomy_valid": np.ones(len(kept), dtype=np.bool_),
        "anatomy_quality": np.asarray([f["anatomy_quality"] for f in kept_features], dtype=np.float32),
        "posture_entry": np.asarray([f["posture_entry"] for f in kept_features], dtype=object),
        "posture_exit": np.asarray([f["posture_exit"] for f in kept_features], dtype=object),
        "posture_mode": np.asarray([f["posture_mode"] for f in kept_features], dtype=object),
        "pelvis_height_entry_norm": np.asarray([f["pelvis_height_entry_norm"] for f in kept_features], dtype=np.float32),
        "pelvis_height_exit_norm": np.asarray([f["pelvis_height_exit_norm"] for f in kept_features], dtype=np.float32),
        "pelvis_height_median_norm": np.asarray([f["pelvis_height_median_norm"] for f in kept_features], dtype=np.float32),
        "body_height_entry_norm": np.asarray([f["body_height_entry_norm"] for f in kept_features], dtype=np.float32),
        "body_height_exit_norm": np.asarray([f["body_height_exit_norm"] for f in kept_features], dtype=np.float32),
        "body_height_median_norm": np.asarray([f["body_height_median_norm"] for f in kept_features], dtype=np.float32),
        "entry_floor_offset_m": np.asarray([f["entry_floor_offset_m"] for f in kept_features], dtype=np.float32),
        "exit_floor_offset_m": np.asarray([f["exit_floor_offset_m"] for f in kept_features], dtype=np.float32),
        "torso_compression_ratio_p05": np.asarray([f["torso_compression_ratio_p05"] for f in kept_features], dtype=np.float32),
        "local_angle_violation_ratio": np.asarray([f["local_angle_violation_ratio"] for f in kept_features], dtype=np.float32),
        "self_collision_severe_ratio": np.asarray([f["self_collision_severe_ratio"] for f in kept_features], dtype=np.float32),
        "spine_cumulative_angle_p95_deg": np.asarray([f["spine_cumulative_angle_p95_deg"] for f in kept_features], dtype=np.float32),
        "event_index_before_anatomy_filter": kept.astype(np.int32),
    })

    if "desc" in out:
        desc = np.asarray(out["desc"], dtype=np.float32)
        mean = desc.mean(axis=0, keepdims=True)
        std = desc.std(axis=0, keepdims=True) + 1e-6
        out["desc_mean"] = mean.astype(np.float32)
        out["desc_std"] = std.astype(np.float32)
        out["desc_z"] = ((desc - mean) / std).astype(np.float32)

    backup = db_path.with_name(db_path.stem + ".pre_v46_52.npz")
    if not backup.exists():
        shutil.copy2(db_path, backup)
    np.savez_compressed(db_path, **out)
    kept_meta = [meta[i] for i in kept]
    meta_path.write_text(json.dumps(_jsonable(kept_meta), ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "schema": "v46_52_event_anatomy_filter",
        "db": str(db_path),
        "backup": str(backup),
        "events_before": int(n),
        "events_after": int(len(kept)),
        "keep_ratio": float(len(kept) / max(1, n)),
        "quality_min": float(np.min(out["anatomy_quality"])),
        "quality_median": float(np.median(out["anatomy_quality"])),
        "posture_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(out["posture_mode"], return_counts=True))
        },
        "rejected": [features[i] for i in np.where(~keep)[0]],
        "load_failures": failures,
        "ok": True,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--audit", default=None)
    args = ap.parse_args(argv)
    db = Path(args.db)
    meta = Path(args.meta) if args.meta else db.with_name("events_meta.json")
    audit = Path(args.audit) if args.audit else db.with_name("events.v46_52_anatomy.audit.json")
    result = filter_database(db, meta, audit)
    print(json.dumps({k: result[k] for k in ("events_before", "events_after", "keep_ratio", "quality_min", "ok")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
