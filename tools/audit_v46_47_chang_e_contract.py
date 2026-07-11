#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.47 Chang-E contract audit.

Checks a Chang-E BVH directory or an Event-RAG DB directory after V46.44/V46.45/V46.47:
  - root scale stays meter-level;
  - root upright mode is effective;
  - Rot6D/FK is finite;
  - contact ratio is reasonable;
  - event duration and root ranges are not collapsed;
  - source-disjoint metadata is present.

Run examples:
  python tools/audit_v46_47_chang_e_contract.py --bvh_dir change --out output/v46_47_audit_change.json
  python tools/audit_v46_47_chang_e_contract.py --db output/v46_47_db --out output/v46_47_audit_db.json --csv output/v46_47_audit_db.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import v46_motionrag_diff as v46  # noqa: E402


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def root_tilt_stats(motion: np.ndarray) -> Dict[str, float]:
    R = v46.rot6d_to_matrix_np(motion[:, v46.ROT6D_START:v46.ROT6D_START + 6].reshape(-1, 1, 6))[:, 0]
    up = R[:, :, 1]
    tilt = np.arccos(np.clip(np.abs(up[:, 1]), 0.0, 1.0))
    fwd = R[:, :, 2]
    yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
    return {
        "root_tilt_p95_rad": float(np.percentile(tilt, 95)) if tilt.size else 0.0,
        "root_tilt_max_rad": float(np.max(tilt)) if tilt.size else 0.0,
        "root_yaw_range_rad": float(np.ptp(yaw)) if yaw.size else 0.0,
    }


def motion_stats(motion: np.ndarray, source: str = "") -> Dict[str, Any]:
    m = np.asarray(motion, dtype=np.float32)
    out: Dict[str, Any] = {"source": source, "frames": int(m.shape[0]), "dim": int(m.shape[1]) if m.ndim == 2 else -1}
    if m.ndim != 2 or m.shape[1] < v46.ROT6D_END or m.shape[0] < 2:
        out["valid"] = False
        out["reason"] = "bad_shape"
        return out
    out["valid"] = bool(np.isfinite(m).all())
    root = m[:, [v46.ROOT_X_IDX, v46.ROOT_Y_IDX, v46.ROOT_Z_IDX]]
    root_xz = root[:, [0, 2]]
    out.update({
        "root_xz_range_m": float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0))),
        "root_xz_start_end_m": float(np.linalg.norm(root_xz[-1] - root_xz[0])),
        "root_y_range_m": float(root[:, 1].max() - root[:, 1].min()),
        "duration_s_at_30fps": float(m.shape[0] / 30.0),
    })
    try:
        out.update(root_tilt_stats(m))
    except Exception as exc:
        out["root_tilt_error"] = str(exc)
    try:
        contacts, conf, floor_y, foot = v46.derive_contacts_np(m, v46.V46Config().apply_env())
        out.update({
            "contact_ratio": float(contacts.mean()),
            "contact_conf_mean": float(conf.mean()),
            "floor_y": float(floor_y),
            "foot_y_min_minus_floor": float(np.min(foot[..., 1] - floor_y)),
        })
    except Exception as exc:
        out["contact_error"] = str(exc)
    try:
        audit = v46.audit_motion_np(m, v46.V46Config().apply_env())
        for k, val in audit.items():
            out[f"audit_{k}"] = val
    except Exception as exc:
        out["audit_error"] = str(exc)
    return out


def audit_bvh_dir(bvh_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(bvh_dir.rglob("*.bvh")):
        try:
            seqs = v46.load_bvh_file(p)
            if not seqs:
                rows.append({"source": str(p), "valid": False, "reason": "load_empty"})
                continue
            for i, m in enumerate(seqs):
                row = motion_stats(m, source=str(p))
                row["seq_id"] = int(i)
                rows.append(row)
        except Exception as exc:
            rows.append({"source": str(p), "valid": False, "reason": str(exc)})
    return rows


def audit_db(db_dir: Path) -> List[Dict[str, Any]]:
    npz_path = db_dir / "events.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    db = np.load(npz_path, allow_pickle=True)
    paths = np.asarray(db["paths"], dtype=object)
    source_groups = np.asarray(db.get("source_groups", np.array([""] * len(paths), dtype=object)), dtype=object)
    source_uids = np.asarray(db.get("source_uids", source_groups), dtype=object)
    labels = np.asarray(db.get("labels", np.array([""] * len(paths), dtype=object)), dtype=object)
    dance_keys = np.asarray(db.get("dance_keys", labels), dtype=object)
    support = np.asarray(db.get("support_labels", np.array([""] * len(paths), dtype=object)), dtype=object)
    starts = np.asarray(db.get("event_starts", np.zeros(len(paths), dtype=np.int32)))
    ends = np.asarray(db.get("event_ends", np.zeros(len(paths), dtype=np.int32)))
    rows: List[Dict[str, Any]] = []
    for i, p0 in enumerate(paths):
        p = Path(str(p0))
        row: Dict[str, Any] = {
            "event_id": int(i),
            "path": str(p),
            "source_group": str(source_groups[i]),
            "source_uid": str(source_uids[i]),
            "label": str(labels[i]),
            "dance_key": str(dance_keys[i]),
            "support_label": str(support[i]),
            "event_start": int(starts[i]) if i < len(starts) else 0,
            "event_end": int(ends[i]) if i < len(ends) else 0,
        }
        try:
            m = np.load(p).astype(np.float32)
            row.update(motion_stats(m, source=str(p)))
        except Exception as exc:
            row.update({"valid": False, "reason": str(exc)})
        rows.append(row)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("valid", False)]
    def arr(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in valid if key in r and r[key] is not None and np.isfinite(float(r[key]))], dtype=np.float32)
    summary: Dict[str, Any] = {"num_rows": len(rows), "num_valid": len(valid), "num_invalid": len(rows) - len(valid)}
    for key in ["root_xz_range_m", "root_y_range_m", "root_tilt_p95_rad", "root_tilt_max_rad", "contact_ratio", "audit_foot_skate_p95_mpf", "audit_mean_joint_jerk_p95"]:
        a = arr(key)
        if a.size:
            summary[key] = {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}
    # Fail flags useful for bash grep.
    summary["flags"] = {
        "root_xz_collapsed_count": int(sum(float(r.get("root_xz_range_m", 999.0)) < 0.05 for r in valid)),
        "root_tilt_large_count": int(sum(float(r.get("root_tilt_p95_rad", 0.0)) > 0.20 for r in valid)),
        "contact_empty_count": int(sum(float(r.get("contact_ratio", 0.0)) < 0.01 for r in valid)),
        "nonfinite_count": int(sum(not bool(r.get("valid", False)) for r in rows)),
    }
    return summary


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(jsonable(r.get(k)), ensure_ascii=False) if isinstance(r.get(k), (dict, list, tuple)) else r.get(k) for k in keys})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_dir", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows: List[Dict[str, Any]] = []
    if args.bvh_dir:
        rows.extend(audit_bvh_dir(Path(args.bvh_dir)))
    if args.db:
        rows.extend(audit_db(Path(args.db)))
    if not rows:
        raise SystemExit("Provide --bvh_dir and/or --db")
    report = {"summary": summarize(rows), "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv:
        write_csv(rows, Path(args.csv))
    print(json.dumps(jsonable(report["summary"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
