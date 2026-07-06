#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict audit for V46 Event-RAG databases (V46.31 semantic-event aware).

Independent from tools/v46_motionrag_diff.py so it can be run before and after
patching.  It checks saved events against the EDGE-151D contract:
- shape and finite values;
- [0:4] contact channel range;
- root scale / height range;
- raw rot6d validity before projection;
- SO(3) validity after conversion;
- DB-level source/semantic distribution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np

ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX = 4, 5, 6
ROT6D_START, ROT6D_END = 7, 151
EDGE_DIM = 151
NUM_JOINTS = 24


def pct(x, q, default=0.0):
    try:
        x = np.asarray(x)
        return float(np.nanpercentile(x, q)) if x.size else float(default)
    except Exception:
        return float(default)


def as_iterable_values(value: Any) -> list:
    """Safely convert npz metadata fields to a list for Counter."""
    try:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                scalar = value.item()
                return [] if scalar is None else [scalar]
            return value.reshape(-1).tolist()
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]
    except Exception:
        return []


def rot6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def raw_rot6d_error(rot6d: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Check whether raw saved 6D rotations already look projected."""
    try:
        rot = rot6d.reshape(-1, NUM_JOINTS, 6)
        a1 = rot[..., 0:3]
        a2 = rot[..., 3:6]
        finite = np.isfinite(rot).all(axis=-1)
        a1c = np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0)
        a2c = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0)
        n1 = np.linalg.norm(a1c, axis=-1)
        n2 = np.linalg.norm(a2c, axis=-1)
        dot = np.sum(a1c * a2c, axis=-1)
        cross_norm = np.linalg.norm(np.cross(a1c, a2c), axis=-1) / np.maximum(n1 * n2, 1e-8)
        bad_finite_ratio = float(1.0 - np.mean(finite)) if finite.size else 1.0
        degenerate_ratio = float(np.mean((n1 < 1e-5) | (n2 < 1e-5) | (cross_norm < 1e-5))) if n1.size else 1.0
        return (
            pct(np.abs(n1 - 1.0), 95, 999.0),
            pct(np.abs(n2 - 1.0), 95, 999.0),
            pct(np.abs(dot), 95, 999.0),
            bad_finite_ratio,
            degenerate_ratio,
            pct(cross_norm, 5, 0.0),
        )
    except Exception:
        return 999.0, 999.0, 999.0, 1.0, 1.0, 0.0


def rot6d_orthogonality_error(rot6d: np.ndarray) -> tuple[float, float]:
    """Orthogonality after conversion, useful but not sufficient by itself."""
    try:
        rot = rot6d.reshape(-1, NUM_JOINTS, 6)
        R = rot6d_to_matrix_np(rot)
        I = np.eye(3, dtype=np.float32)
        RtR = np.matmul(np.swapaxes(R, -1, -2), R)
        err = np.abs(RtR - I).reshape(-1)
        det = np.linalg.det(R.reshape(-1, 3, 3))
        return pct(err, 95), pct(np.abs(det - 1.0), 95)
    except Exception:
        return 999.0, 999.0


def audit_event(path: str) -> dict:
    m = np.load(path).astype(np.float32)
    out = {
        "path": str(path),
        "shape": list(m.shape),
        "valid_shape": bool(m.ndim == 2 and m.shape[1] >= EDGE_DIM and m.shape[0] >= 8),
        "finite": bool(np.isfinite(m).all()),
    }
    if not out["valid_shape"]:
        out["bad"] = True
        out["reasons"] = ["invalid_shape"]
        return out

    x = m[:, :EDGE_DIM]
    contact = x[:, 0:4]
    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    rot = x[:, ROT6D_START:ROT6D_END]
    reasons: list[str] = []

    c_abs_p95 = pct(np.abs(contact), 95)
    c_min = float(np.nanmin(contact))
    c_max = float(np.nanmax(contact))
    c_unit_range = bool(np.all((contact >= -1e-4) & (contact <= 1.0001)))
    root_y_range = float(np.nanmax(root[:, 1]) - np.nanmin(root[:, 1]))
    root_xz_travel = float(np.linalg.norm(root[-1, [0, 2]] - root[0, [0, 2]]))
    root_abs_p99 = pct(np.abs(root), 99)
    rot_abs_p95 = pct(np.abs(rot), 95)
    rot_abs_p50 = pct(np.abs(rot), 50)
    raw_n1_err, raw_n2_err, raw_dot_err, raw_bad_finite_ratio, raw_degenerate_ratio, raw_cross_norm_p05 = raw_rot6d_error(rot)
    orth_p95, det_err_p95 = rot6d_orthogonality_error(rot)

    if not out["finite"]:
        reasons.append("nan_or_inf")
    if c_abs_p95 > 1.5 or c_min < -0.05 or c_max > 1.5:
        reasons.append("contact_channel_metadata_or_out_of_range")
    if not c_unit_range:
        reasons.append("contact_not_in_unit_range")
    if root_y_range > 3.0:
        reasons.append("root_y_range_too_large")
    if root_abs_p99 > 50.0:
        reasons.append("root_translation_scale_suspicious")
    if rot_abs_p95 < 0.05 or rot_abs_p95 > 2.5:
        reasons.append("rot6d_distribution_suspicious")
    if raw_bad_finite_ratio > 0.0 or raw_degenerate_ratio > 0.0:
        reasons.append("raw_rot6d_contains_nan_or_degenerate_vectors")
    # Strong signature of the historical row-major matrix_to_rot6d bug: the
    # second 6D vector has near-zero norm and/or becomes almost collinear with
    # the first vector after saving.
    if raw_n2_err > 0.80 and raw_dot_err > 0.80:
        reasons.append("rot6d_row_major_matrix_to_6d_bug_signature")
    if raw_n1_err > 0.15 or raw_n2_err > 0.15 or raw_dot_err > 0.20:
        reasons.append("raw_rot6d_not_projected")
    if orth_p95 > 1e-3 or det_err_p95 > 1e-3:
        reasons.append("rot6d_projection_or_orthogonality_suspicious")

    out.update({
        "contact_min": c_min,
        "contact_max": c_max,
        "contact_abs_p95": c_abs_p95,
        "contact_unit_range": c_unit_range,
        "root_min": [float(v) for v in np.nanmin(root, axis=0)],
        "root_max": [float(v) for v in np.nanmax(root, axis=0)],
        "root_y_range_m": root_y_range,
        "root_xz_travel_m": root_xz_travel,
        "root_abs_p99": root_abs_p99,
        "rot6d_abs_p50": rot_abs_p50,
        "rot6d_abs_p95": rot_abs_p95,
        "raw_rot6d_n1_err_p95": raw_n1_err,
        "raw_rot6d_n2_err_p95": raw_n2_err,
        "raw_rot6d_dot_abs_p95": raw_dot_err,
        "raw_rot6d_bad_finite_ratio": raw_bad_finite_ratio,
        "raw_rot6d_degenerate_ratio": raw_degenerate_ratio,
        "raw_rot6d_cross_norm_p05": raw_cross_norm_p05,
        "rot6d_orth_err_p95": orth_p95,
        "rot6d_det_err_p95": det_err_p95,
        "bad": bool(reasons),
        "reasons": reasons,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to events.npz")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max_preview", type=int, default=40)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    db = np.load(args.db, allow_pickle=True)
    paths = [str(x) for x in as_iterable_values(db["paths"])]
    rows = [audit_event(p) for p in paths]
    bad = [r for r in rows if r.get("bad")]

    summary = {
        "db": args.db,
        "num_events": len(paths),
        "num_bad_events": len(bad),
        "bad_ratio": float(len(bad) / max(1, len(paths))),
        "bad_reason_counts": Counter(reason for r in bad for reason in r.get("reasons", [])).most_common(),
        "contact_abs_p95_global_p95": pct([r.get("contact_abs_p95", 0.0) for r in rows], 95),
        "root_y_range_global_p95": pct([r.get("root_y_range_m", 0.0) for r in rows], 95),
        "root_abs_p99_global_p95": pct([r.get("root_abs_p99", 0.0) for r in rows], 95),
        "rot6d_abs_p95_global_p95": pct([r.get("rot6d_abs_p95", 0.0) for r in rows], 95),
        "raw_rot6d_n1_err_global_p95": pct([r.get("raw_rot6d_n1_err_p95", 0.0) for r in rows], 95),
        "raw_rot6d_n2_err_global_p95": pct([r.get("raw_rot6d_n2_err_p95", 0.0) for r in rows], 95),
        "raw_rot6d_dot_abs_global_p95": pct([r.get("raw_rot6d_dot_abs_p95", 0.0) for r in rows], 95),
        "raw_rot6d_degenerate_ratio_global_p95": pct([r.get("raw_rot6d_degenerate_ratio", 0.0) for r in rows], 95),
        "raw_rot6d_cross_norm_global_p05": pct([r.get("raw_rot6d_cross_norm_p05", 1.0) for r in rows], 5),
        "rot6d_orth_err_global_p95": pct([r.get("rot6d_orth_err_p95", 0.0) for r in rows], 95),
        "rot6d_det_err_global_p95": pct([r.get("rot6d_det_err_p95", 0.0) for r in rows], 95),
        "bad_preview": bad[:args.max_preview],
    }
    if "event_quality_scores" in db.files:
        q = np.asarray(db["event_quality_scores"], dtype=np.float32).reshape(-1)
        summary["event_quality_p05"] = pct(q, 5)
        summary["event_quality_p50"] = pct(q, 50)
        summary["event_quality_p95"] = pct(q, 95)
        summary["event_quality_lt_0_22_ratio"] = float(np.mean(q < 0.22)) if q.size else 0.0
    for key in ["source_groups", "dance_keys", "music_alignment_labels", "labels", "event_families", "motion_stage_roles", "cultural_motifs", "prop_proxy_labels", "locomotion_labels", "support_labels"]:
        if key in db.files:
            vals = as_iterable_values(db[key])
            summary[key + "_counts"] = Counter(map(str, vals)).most_common(30)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")
    if args.strict and bad:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
