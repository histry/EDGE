#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobility-aware labels for ChoreoRAG motion units.

Labels:
  0 stationary_expressive
  1 stationary
  2 turn_in_place
  3 mobile
  4 landing
  5 unsuitable

This file is intentionally non-invasive: it does not modify model/model.py or train.py.
It reads an existing .npz RAG DB and writes a new .npz with mobility labels appended.

Usage:
  python mobility_unit_labels.py \
    --input data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz \
    --output data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np


LABELS = [
    "stationary_expressive",
    "stationary",
    "turn_in_place",
    "mobile",
    "landing",
    "unsuitable",
]
LABEL_TO_ID = {v: i for i, v in enumerate(LABELS)}

ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6


def _first_existing_key(data: Dict[str, np.ndarray], names):
    for n in names:
        if n in data:
            return n
    return None


def _as_float_array(x, n=None, default=0.0):
    if x is None:
        if n is None:
            return None
        return np.full((n,), default, dtype=np.float32)
    y = np.asarray(x)
    if y.ndim == 0:
        if n is None:
            return y.astype(np.float32)
        return np.full((n,), float(y), dtype=np.float32)
    return y.astype(np.float32)


def _infer_n(data: Dict[str, np.ndarray]) -> int:
    priority = [
        "unit_motions", "units", "clips", "motions", "motion_units",
        "embeddings", "text_embeddings", "text_embeds", "features",
        "root_path", "energy", "expressiveness", "upper_activity",
    ]
    for k in priority:
        if k in data:
            arr = np.asarray(data[k])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    for k, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    raise ValueError("Cannot infer number of units from DB")


def _find_motion_array(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    candidates = [
        "unit_motions", "motion_units", "units", "clips",
        "clip_motions", "motions", "poses",
    ]
    for k in candidates:
        if k in data:
            arr = np.asarray(data[k])
            if arr.ndim >= 3:
                return arr.astype(np.float32)
    return None


def _root_path_from_motion(m: np.ndarray) -> np.ndarray:
    # m: [N,T,D]
    root = m[..., [ROOT_X_IDX, ROOT_Z_IDX]]
    return np.linalg.norm(np.diff(root, axis=1), axis=-1).sum(axis=1).astype(np.float32)


def _root_disp_from_motion(m: np.ndarray) -> np.ndarray:
    root = m[..., [ROOT_X_IDX, ROOT_Z_IDX]]
    return np.linalg.norm(root[:, -1] - root[:, 0], axis=-1).astype(np.float32)


def _root_y_range_from_motion(m: np.ndarray) -> np.ndarray:
    if m.shape[-1] <= ROOT_Y_IDX:
        return np.zeros((m.shape[0],), dtype=np.float32)
    y = m[..., ROOT_Y_IDX]
    return (y.max(axis=1) - y.min(axis=1)).astype(np.float32)


def _activity_from_motion(m: np.ndarray, sl: slice) -> np.ndarray:
    if m.shape[-1] <= sl.start:
        return np.zeros((m.shape[0],), dtype=np.float32)
    part = m[..., sl]
    if part.shape[-1] == 0:
        return np.zeros((m.shape[0],), dtype=np.float32)
    d = np.diff(part, axis=1)
    return np.linalg.norm(d, axis=-1).mean(axis=1).astype(np.float32)


def _jerk_from_motion(m: np.ndarray) -> np.ndarray:
    if m.shape[1] < 4:
        return np.zeros((m.shape[0],), dtype=np.float32)
    j = np.diff(m, n=3, axis=1)
    return np.linalg.norm(j, axis=-1).mean(axis=1).astype(np.float32)


def _contact_switch_from_motion(m: np.ndarray) -> np.ndarray:
    if m.shape[-1] < 4:
        return np.zeros((m.shape[0],), dtype=np.float32)
    c = m[..., :4] > 0.5
    sw = np.abs(np.diff(c.astype(np.int8), axis=1)).sum(axis=(1, 2))
    return sw.astype(np.float32)


def _turn_proxy_from_motion(m: np.ndarray) -> np.ndarray:
    """
    We do not assume a full FK/yaw pipeline here.
    This proxy measures large torso/upper rotation change with low root path.
    """
    if m.shape[-1] < 151:
        return np.zeros((m.shape[0],), dtype=np.float32)
    # 6D rotation chunks after dim 7.
    rot = m[..., 7:151]
    # torso-ish chunks roughly around first several body joints.
    torso = rot[..., 6 * 1:6 * 8]
    delta = torso[:, -1] - torso[:, 0]
    return np.linalg.norm(delta, axis=-1).astype(np.float32)


def compute_metrics(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    n = _infer_n(data)
    motion = _find_motion_array(data)

    metrics: Dict[str, np.ndarray] = {}

    # Prefer existing DB metrics if present.
    key_map = {
        "root_path": ["root_path", "root_path_len", "path_len", "path_length", "loco_root_path"],
        "root_disp": ["root_disp", "root_displacement", "displacement", "loco_disp"],
        "energy": ["energy", "motion_energy", "unit_energy"],
        "expressiveness": ["expressiveness", "expr", "expr_score", "expressive_score"],
        "upper_activity": ["upper_activity", "upper", "upper_rms", "upper_score"],
        "torso_activity": ["torso_activity", "torso", "torso_rms", "torso_score"],
        "lower_activity": ["lower_activity", "lower", "lower_rms", "lower_score"],
        "contact_switch": ["contact_switch", "contact_switch_count", "support_switch"],
        "turn": ["turn", "turn_score", "heading_change", "yaw_change", "direction_change"],
        "jerk": ["jerk", "transition_jerk", "motion_jerk"],
        "foot_slide": ["foot_slide", "foot_slide_rate", "slide_rate"],
    }

    for out_key, names in key_map.items():
        k = _first_existing_key(data, names)
        if k is not None:
            arr = np.asarray(data[k])
            if arr.ndim >= 1 and arr.shape[0] == n:
                metrics[out_key] = arr.astype(np.float32)

    if motion is not None:
        if "root_path" not in metrics:
            metrics["root_path"] = _root_path_from_motion(motion)
        if "root_disp" not in metrics:
            metrics["root_disp"] = _root_disp_from_motion(motion)
        if "upper_activity" not in metrics:
            metrics["upper_activity"] = _activity_from_motion(motion, slice(79, 151))
        if "torso_activity" not in metrics:
            metrics["torso_activity"] = _activity_from_motion(motion, slice(31, 79))
        if "lower_activity" not in metrics:
            metrics["lower_activity"] = _activity_from_motion(motion, slice(7, 79))
        if "energy" not in metrics:
            metrics["energy"] = _activity_from_motion(motion, slice(7, motion.shape[-1]))
        if "contact_switch" not in metrics:
            metrics["contact_switch"] = _contact_switch_from_motion(motion)
        if "turn" not in metrics:
            metrics["turn"] = _turn_proxy_from_motion(motion)
        if "jerk" not in metrics:
            metrics["jerk"] = _jerk_from_motion(motion)
        metrics["root_y_range"] = _root_y_range_from_motion(motion)

    # Fill missing metrics.
    for k in [
        "root_path", "root_disp", "energy", "expressiveness",
        "upper_activity", "torso_activity", "lower_activity",
        "contact_switch", "turn", "jerk", "foot_slide", "root_y_range",
    ]:
        if k not in metrics:
            metrics[k] = np.zeros((n,), dtype=np.float32)

    return metrics


def robust_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(x[finite], 5)
    hi = np.percentile(x[finite], 95)
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def label_units(
    metrics: Dict[str, np.ndarray],
    stationary_root_path_max: float = 0.45,
    mobile_root_path_min: float = 0.75,
    turn_root_path_max: float = 0.65,
    high_turn_min_norm: float = 0.60,
    expressive_min_norm: float = 0.55,
    contact_switch_min_norm: float = 0.25,
    landing_speed_drop_min: float = 0.20,
    jerk_unsuitable_norm: float = 0.92,
    root_y_unsuitable_norm: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    n = len(metrics["root_path"])

    root_path = metrics["root_path"]
    root_disp = metrics["root_disp"]
    upper = metrics["upper_activity"]
    torso = metrics["torso_activity"]
    lower = metrics["lower_activity"]
    energy = metrics["energy"]
    expr = metrics["expressiveness"]
    turn = metrics["turn"]
    contact_switch = metrics["contact_switch"]
    jerk = metrics["jerk"]
    root_y_range = metrics.get("root_y_range", np.zeros_like(root_path))

    root_n = robust_norm(root_path)
    upper_n = robust_norm(upper)
    torso_n = robust_norm(torso)
    lower_n = robust_norm(lower)
    energy_n = robust_norm(energy)
    expr_n = robust_norm(expr)
    turn_n = robust_norm(turn)
    contact_n = robust_norm(contact_switch)
    jerk_n = robust_norm(jerk)
    root_y_n = robust_norm(root_y_range)

    expressive_score = np.maximum.reduce([upper_n, torso_n, energy_n, expr_n])
    support_score = np.maximum(contact_n, lower_n)
    mobile_score = 0.55 * root_n + 0.35 * support_score + 0.10 * lower_n
    stationary_score = (1.0 - root_n) * (0.55 * expressive_score + 0.45 * (1.0 - contact_n))
    turn_score = (1.0 - root_n) * (0.65 * turn_n + 0.35 * torso_n)
    landing_score = mobile_score * (1.0 - np.maximum(upper_n, torso_n)) * (1.0 - jerk_n)

    unsuitable = (
        (~np.isfinite(root_path))
        | (jerk_n >= jerk_unsuitable_norm)
        | (root_y_n >= root_y_unsuitable_norm)
    )

    label = np.full((n,), LABEL_TO_ID["stationary"], dtype=np.int64)

    # Hard logic first.
    stationary_mask = root_path <= stationary_root_path_max
    mobile_mask = root_path >= mobile_root_path_min
    turn_mask = (root_path <= turn_root_path_max) & (turn_n >= high_turn_min_norm)

    label[stationary_mask] = LABEL_TO_ID["stationary"]
    label[stationary_mask & (expressive_score >= expressive_min_norm)] = LABEL_TO_ID["stationary_expressive"]
    label[turn_mask] = LABEL_TO_ID["turn_in_place"]

    label[mobile_mask & ((contact_n >= contact_switch_min_norm) | (lower_n >= 0.45))] = LABEL_TO_ID["mobile"]

    # Landing proxy: moving unit with lower end intensity / low jerk.
    # Without frame-level velocity, this is conservative.
    landing_mask = mobile_mask & (landing_score >= np.percentile(landing_score, 80))
    label[landing_mask] = LABEL_TO_ID["landing"]

    label[unsuitable] = LABEL_TO_ID["unsuitable"]

    scores = {
        "mobility_score_stationary": stationary_score.astype(np.float32),
        "mobility_score_mobile": mobile_score.astype(np.float32),
        "mobility_score_turn": turn_score.astype(np.float32),
        "mobility_score_landing": landing_score.astype(np.float32),
        "mobility_score_expressive": expressive_score.astype(np.float32),
        "mobility_score_support": support_score.astype(np.float32),
        "mobility_norm_root_path": root_n,
        "mobility_norm_upper": upper_n,
        "mobility_norm_torso": torso_n,
        "mobility_norm_lower": lower_n,
        "mobility_norm_contact_switch": contact_n,
        "mobility_norm_turn": turn_n,
        "mobility_norm_jerk": jerk_n,
    }

    label_name = np.array([LABELS[int(i)] for i in label])
    return label, label_name, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--stationary_root_path_max", type=float, default=0.45)
    ap.add_argument("--mobile_root_path_min", type=float, default=0.75)
    ap.add_argument("--turn_root_path_max", type=float, default=0.65)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(inp, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    metrics = compute_metrics(data)
    label_id, label_name, scores = label_units(
        metrics,
        stationary_root_path_max=args.stationary_root_path_max,
        mobile_root_path_min=args.mobile_root_path_min,
        turn_root_path_max=args.turn_root_path_max,
    )

    data_out = dict(data)
    data_out["mobility_label_id"] = label_id
    data_out["mobility_label"] = label_name
    for k, v in metrics.items():
        data_out[f"mobility_metric_{k}"] = v.astype(np.float32)
    data_out.update(scores)

    np.savez_compressed(out, **data_out)

    report = {
        "input": str(inp),
        "output": str(out),
        "n_units": int(len(label_id)),
        "labels": {name: int((label_name == name).sum()) for name in LABELS},
        "thresholds": {
            "stationary_root_path_max": args.stationary_root_path_max,
            "mobile_root_path_min": args.mobile_root_path_min,
            "turn_root_path_max": args.turn_root_path_max,
        },
        "keys_in_output": sorted(list(data_out.keys())),
    }

    report_path = Path(args.report) if args.report else out.with_suffix(".mobility_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
