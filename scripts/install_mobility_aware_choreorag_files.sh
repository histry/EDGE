#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
mkdir -p scripts

cat > mobility_unit_labels.py <<'PY'
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
PY

cat > mobility_aware_selector.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobility-aware unit selector.

It selects units by intent:
  stationary: stationary_expressive > stationary > turn_in_place
  mobile: mobile > landing
  turn: turn_in_place > stationary_expressive
  landing: landing > mobile

It can export mid poses and unit clips if the RAG DB contains unit_motions/units/clips.
If the DB has only stats/embeddings, it still writes selected indices and reports.

Usage:
  python mobility_aware_selector.py \
    --db data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz \
    --intent stationary \
    --count 2 \
    --out_prefix output/mobility_select/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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


def load_db(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def find_motion_array(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    for k in ["unit_motions", "motion_units", "units", "clips", "clip_motions", "motions", "poses"]:
        if k in data:
            arr = np.asarray(data[k])
            if arr.ndim >= 3:
                return arr
    return None


def intent_allowed_labels(intent: str) -> List[str]:
    if intent == "stationary":
        return ["stationary_expressive", "stationary", "turn_in_place"]
    if intent == "stationary_expressive":
        return ["stationary_expressive", "turn_in_place", "stationary"]
    if intent == "turn":
        return ["turn_in_place", "stationary_expressive", "stationary"]
    if intent == "mobile":
        return ["mobile", "landing"]
    if intent == "landing":
        return ["landing", "mobile"]
    if intent == "all_safe":
        return ["stationary_expressive", "stationary", "turn_in_place", "mobile", "landing"]
    raise ValueError(f"Unknown intent: {intent}")


def score_for_intent(data: Dict[str, np.ndarray], intent: str) -> np.ndarray:
    n = len(data["mobility_label_id"])
    zeros = np.zeros((n,), dtype=np.float32)

    def get(k):
        return np.asarray(data[k], dtype=np.float32) if k in data else zeros

    if intent in ("stationary", "stationary_expressive"):
        return (
            0.45 * get("mobility_score_stationary")
            + 0.40 * get("mobility_score_expressive")
            + 0.10 * get("mobility_score_turn")
            - 0.35 * get("mobility_norm_root_path")
            - 0.15 * get("mobility_norm_jerk")
        )
    if intent == "turn":
        return (
            0.60 * get("mobility_score_turn")
            + 0.25 * get("mobility_score_expressive")
            - 0.25 * get("mobility_norm_root_path")
            - 0.15 * get("mobility_norm_jerk")
        )
    if intent == "mobile":
        return (
            0.55 * get("mobility_score_mobile")
            + 0.30 * get("mobility_score_support")
            + 0.10 * get("mobility_score_expressive")
            - 0.20 * get("mobility_norm_jerk")
        )
    if intent == "landing":
        return (
            0.65 * get("mobility_score_landing")
            + 0.20 * get("mobility_score_support")
            - 0.15 * get("mobility_norm_jerk")
        )
    return (
        get("mobility_score_expressive")
        + get("mobility_score_mobile")
        + get("mobility_score_turn")
        - get("mobility_norm_jerk")
    )


def select_indices(data: Dict[str, np.ndarray], intent: str, count: int, min_gap: int = 1) -> List[int]:
    label_id = np.asarray(data["mobility_label_id"]).astype(np.int64)
    allowed = intent_allowed_labels(intent)
    allowed_ids = {LABEL_TO_ID[x] for x in allowed}
    mask = np.array([x in allowed_ids for x in label_id], dtype=bool)

    # Always remove unsuitable.
    mask &= label_id != LABEL_TO_ID["unsuitable"]

    score = score_for_intent(data, intent)
    score = np.where(mask, score, -1e9)

    order = list(np.argsort(-score))
    selected: List[int] = []
    for idx in order:
        if not np.isfinite(score[idx]) or score[idx] < -1e8:
            continue
        if all(abs(int(idx) - int(j)) >= min_gap for j in selected):
            selected.append(int(idx))
        if len(selected) >= count:
            break
    return selected


def export_selected(data: Dict[str, np.ndarray], indices: List[int], out_prefix: str) -> Dict:
    outp = Path(out_prefix)
    outp.parent.mkdir(parents=True, exist_ok=True)

    motion = find_motion_array(data)
    exported_mid_poses = []
    exported_units = []

    if motion is not None:
        for rank, idx in enumerate(indices, 1):
            clip = np.asarray(motion[idx])
            center = clip.shape[0] // 2
            mid_pose = clip[center]
            unit_path = outp.parent / f"{outp.name}_unit{rank:02d}_idx{idx}.npy"
            pose_path = outp.parent / f"{outp.name}_mid{rank:02d}_idx{idx}.npy"
            np.save(unit_path, clip)
            np.save(pose_path, mid_pose)
            exported_units.append(str(unit_path))
            exported_mid_poses.append(str(pose_path))

    labels = np.asarray(data.get("mobility_label", []))
    score = None
    report_items = []
    for idx in indices:
        item = {"idx": int(idx)}
        if len(labels) > idx:
            item["label"] = str(labels[idx])
        for k in [
            "mobility_score_stationary",
            "mobility_score_mobile",
            "mobility_score_turn",
            "mobility_score_expressive",
            "mobility_metric_root_path",
            "mobility_metric_upper_activity",
            "mobility_metric_torso_activity",
            "mobility_metric_lower_activity",
            "mobility_metric_contact_switch",
            "mobility_metric_jerk",
        ]:
            if k in data:
                item[k] = float(np.asarray(data[k])[idx])
        report_items.append(item)

    report = {
        "indices": indices,
        "mid_poses": exported_mid_poses,
        "unit_paths": exported_units,
        "items": report_items,
    }
    Path(str(out_prefix) + "_selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--intent", default="stationary",
                    choices=["stationary", "stationary_expressive", "turn", "mobile", "landing", "all_safe"])
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--min_gap", type=int, default=10)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    data = load_db(args.db)
    if "mobility_label_id" not in data:
        raise RuntimeError(
            f"{args.db} has no mobility_label_id. Run mobility_unit_labels.py first."
        )

    indices = select_indices(data, args.intent, args.count, args.min_gap)
    report = export_selected(data, indices, args.out_prefix)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
PY

cat > mobility_motion_utils.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion utilities for root-locked / body-centered evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np

ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6


def load_motion(path: str) -> np.ndarray:
    x = np.load(path)
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)


def freeze_root_xz(x: np.ndarray, freeze_y: bool = False) -> np.ndarray:
    y = np.array(x, copy=True)
    if y.ndim != 2:
        raise ValueError(f"Expected [T,D], got {y.shape}")
    y[:, ROOT_X_IDX] = y[0, ROOT_X_IDX]
    y[:, ROOT_Z_IDX] = y[0, ROOT_Z_IDX]
    if freeze_y and y.shape[-1] > ROOT_Y_IDX:
        y[:, ROOT_Y_IDX] = y[0, ROOT_Y_IDX]
    return y


def metrics(x: np.ndarray) -> Dict[str, float]:
    if x.ndim != 2:
        raise ValueError(f"Expected [T,D], got {x.shape}")

    root_xz = x[:, [ROOT_X_IDX, ROOT_Z_IDX]] if x.shape[-1] > ROOT_Z_IDX else np.zeros((len(x), 2))
    root_path = float(np.linalg.norm(np.diff(root_xz, axis=0), axis=-1).sum()) if len(x) > 1 else 0.0
    root_disp = float(np.linalg.norm(root_xz[-1] - root_xz[0])) if len(x) > 1 else 0.0

    pose = x[:, 7:] if x.shape[-1] >= 151 else x
    dpose = np.diff(pose, axis=0)
    frame_energy = np.linalg.norm(dpose, axis=-1) if len(dpose) else np.zeros((1,))

    if x.shape[-1] >= 151:
        # Proxy groups.
        lower = x[:, 7:79]
        torso = x[:, 31:79]
        upper = x[:, 79:151]
        lower_activity = float(np.linalg.norm(np.diff(lower, axis=0), axis=-1).mean())
        torso_activity = float(np.linalg.norm(np.diff(torso, axis=0), axis=-1).mean())
        upper_activity = float(np.linalg.norm(np.diff(upper, axis=0), axis=-1).mean())
    else:
        lower_activity = torso_activity = upper_activity = float("nan")

    jerk = float(np.linalg.norm(np.diff(x, n=3, axis=0), axis=-1).mean()) if len(x) > 4 else 0.0

    return {
        "motion_energy_pose_only": float(frame_energy.mean()),
        "motion_energy_pose_p95": float(np.percentile(frame_energy, 95)),
        "upper_activity_proxy": upper_activity,
        "torso_activity_proxy": torso_activity,
        "lower_activity_proxy": lower_activity,
        "root_path": root_path,
        "root_disp": root_disp,
        "jerk_proxy": jerk,
        "freezing_rate_proxy": float(np.mean(frame_energy < 1e-3)),
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    fz = sub.add_parser("freeze-root")
    fz.add_argument("--input", required=True)
    fz.add_argument("--output", required=True)
    fz.add_argument("--freeze_y", action="store_true")

    ev = sub.add_parser("eval")
    ev.add_argument("--input", required=True)
    ev.add_argument("--output_json", default=None)

    batch = sub.add_parser("batch-eval")
    batch.add_argument("--glob", required=True)
    batch.add_argument("--output_csv", required=True)

    args = ap.parse_args()

    if args.cmd == "freeze-root":
        x = load_motion(args.input)
        y = freeze_root_xz(x, freeze_y=args.freeze_y)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, y)
        print(f"saved {args.output}, shape={y.shape}")

    elif args.cmd == "eval":
        x = load_motion(args.input)
        m_raw = metrics(x)
        x_lock = freeze_root_xz(x)
        m_lock = metrics(x_lock)
        out = {"input": args.input, "raw": m_raw, "root_locked": m_lock}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "batch-eval":
        rows = []
        for p in sorted(Path(".").glob(args.glob)):
            if p.name.endswith("_raw.npy") or "_mid" in p.name or "_unit" in p.name:
                continue
            try:
                x = load_motion(str(p))
                r = metrics(x)
                rl = metrics(freeze_root_xz(x))
                row = {"case": p.stem, "path": str(p)}
                for k, v in r.items():
                    row[f"raw_{k}"] = v
                for k, v in rl.items():
                    row[f"rootlock_{k}"] = v
                rows.append(row)
            except Exception as e:
                rows.append({"case": p.stem, "path": str(p), "error": repr(e)})

        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r.keys()})
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"saved {args.output_csv}")


if __name__ == "__main__":
    main()
PY

cat > generate_mobility_aware.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobility-aware wrapper around generate_controlled_v9.py.

Modes:
  stationary:
    - no --trajectory
    - select stationary / turn-in-place units only
    - optional root-lock after generation

  trajectory:
    - uses --trajectory
    - select mobile / landing units only
    - does NOT allow stationary expressive units to be dragged along the path

This wrapper is deliberately small: it delegates generation to generate_controlled_v9.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from mobility_aware_selector import load_db, select_indices, export_selected
from mobility_motion_utils import load_motion, freeze_root_xz, metrics


def run(cmd: List[str], log_path: Optional[str] = None) -> int:
    print(" ".join(shlex.quote(x) for x in cmd))
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert p.stdout is not None
            for line in p.stdout:
                print(line, end="")
                f.write(line)
            return p.wait()
    return subprocess.call(cmd)


def clear_traj_env():
    for k in list(os.environ.keys()):
        if k.startswith("EDGE_TURN"):
            os.environ.pop(k, None)
    os.environ["EDGE_DYNAMIC_TRAJ_CFG"] = "0"
    os.environ["EDGE_GAIT_PHASE_COND"] = "0"
    os.environ["EDGE_GAIT_CONTACT_LOSS"] = "0"
    os.environ["EDGE_TRAJ_PHYSICS_FEATURES"] = "0"
    os.environ["EDGE_TRAJ_FOURIER_FEATURES"] = "0"
    os.environ["EDGE_TRAJ_SPARSE_WAYPOINT"] = "0"


def select_mid_poses(db_path: str, intent: str, count: int, out_prefix: str) -> List[str]:
    data = load_db(db_path)
    idxs = select_indices(data, intent=intent, count=count, min_gap=10)
    report = export_selected(data, idxs, out_prefix)
    return report.get("mid_poses", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stationary", "trajectory"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--music", required=True)
    ap.add_argument("--start_pose", required=True)
    ap.add_argument("--end_pose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mobility_db", required=True)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--sampler", default="ddim")
    ap.add_argument("--feature_type", default="hybrid")
    ap.add_argument("--endpoint_keyframe_strength", type=float, default=0.3)
    ap.add_argument("--mid_count", type=int, default=1)
    ap.add_argument("--mid_keyframe_strength", type=float, default=0.08)
    ap.add_argument("--beat_weight", type=float, default=0.0)
    ap.add_argument("--energy_scale", type=float, default=0.5)
    ap.add_argument("--context_scale", type=float, default=0.5)
    ap.add_argument("--unit_prior_strength", type=float, default=0.012)
    ap.add_argument("--unit_prior_features", default="upper")
    ap.add_argument("--root_lock_after", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    clear_traj_env()

    os.environ["EDGE_MOBILITY_AWARE"] = "1"
    os.environ["EDGE_MOBILITY_MODE"] = args.mode
    os.environ["EDGE_CHECKPOINT_COMPAT_CPU_MERGE"] = "1"
    os.environ["EDGE_AUDIO_DEVICE"] = "cpu"
    os.environ["EDGE_EXPERIMENT_PROFILE"] = "v10"
    os.environ["EDGE_ENABLE_TEXT_CONTEXT_RAG"] = "1"
    os.environ["EDGE_ENABLE_RAG_SUMMARY_TOKEN"] = "1"

    os.environ["EDGE_AUDIO_ENERGY_AS_COND"] = "1"
    os.environ["EDGE_MUSIC_TENSION_AS_ENERGY"] = "1"
    os.environ["EDGE_ENERGY_CFG_SCALE"] = str(args.energy_scale)

    os.environ["EDGE_CONTEXT_RAG_ENHANCE"] = "1"
    os.environ["EDGE_CONTEXT_RAG_SCALE"] = str(args.context_scale)

    os.environ["EDGE_UNIT_SOFT_PRIOR"] = "1"
    os.environ["EDGE_UNIT_PRIOR_REQUIRED"] = "0"
    os.environ["EDGE_UNIT_PRIOR_TEMPORAL"] = "1"
    os.environ["EDGE_UNIT_PRIOR_DCT"] = "1"
    os.environ["EDGE_UNIT_PRIOR_DCT_DECAY"] = "soft_exp"
    os.environ["EDGE_UNIT_PRIOR_LOW_FREQ_K"] = "4"
    os.environ["EDGE_UNIT_PRIOR_FEATURES"] = args.unit_prior_features
    os.environ["EDGE_UNIT_PRIOR_STRENGTH"] = str(args.unit_prior_strength)

    if args.beat_weight > 0:
        os.environ["EDGE_BEAT_GUIDANCE"] = "1"
        os.environ["EDGE_BEAT_GUIDANCE_WEIGHT"] = str(args.beat_weight)
        os.environ["EDGE_BEAT_GUIDANCE_TARGET"] = "1.35"
        os.environ["EDGE_BEAT_GUIDANCE_FEATURES"] = "all"
    else:
        os.environ["EDGE_BEAT_GUIDANCE"] = "0"
        os.environ["EDGE_BEAT_GUIDANCE_WEIGHT"] = "0"

    intent = "stationary_expressive" if args.mode == "stationary" else "mobile"
    prefix = str(Path(args.out).with_suffix("")) + "_mobility"
    mid_poses = []
    if args.mid_count > 0:
        mid_poses = select_mid_poses(args.mobility_db, intent, args.mid_count, prefix)

    cmd = [
        sys.executable,
        "generate_controlled_v9.py",
        "--checkpoint", args.checkpoint,
        "--music", args.music,
        "--start_pose", args.start_pose,
        "--end_pose", args.end_pose,
        "--out", args.out,
        "--feature_type", args.feature_type,
        "--sampler", args.sampler,
        "--endpoint_keyframe_strength", str(args.endpoint_keyframe_strength),
        "--no_tto",
    ]

    if mid_poses:
        frames = []
        if len(mid_poses) == 1:
            frames = ["75"]
        elif len(mid_poses) == 2:
            frames = ["50", "100"]
        else:
            # evenly spaced excluding endpoints
            frames = [str(int(round((i + 1) * 150 / (len(mid_poses) + 1)))) for i in range(len(mid_poses))]
        cmd += [
            "--mid_poses", ",".join(mid_poses),
            "--mid_pose_frames", ",".join(frames),
            "--mid_keyframe_strength", str(args.mid_keyframe_strength),
        ]

    if args.mode == "trajectory":
        if not args.trajectory:
            raise ValueError("--trajectory is required in trajectory mode")
        cmd += ["--trajectory", args.trajectory]

    status = run(cmd, args.log)
    if status != 0:
        raise SystemExit(status)

    final_out = args.out
    if args.mode == "stationary" and args.root_lock_after:
        x = load_motion(args.out)
        y = freeze_root_xz(x)
        rootlock_out = str(Path(args.out).with_suffix("")) + "_rootlock.npy"
        np.save(rootlock_out, y)
        final_out = rootlock_out
        report = {
            "raw": metrics(x),
            "root_locked": metrics(y),
            "rootlock_motion": rootlock_out,
        }
        Path(str(Path(args.out).with_suffix("")) + "_rootlock_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )

    if args.render:
        audio = args.music
        base = str(Path(final_out).with_suffix(""))
        run([
            sys.executable, "render_from_npy.py",
            "--motion", final_out,
            "--audio", audio,
            "--output", base + "_follow.mp4",
            "--camera_mode", "follow",
        ])
        run([
            sys.executable, "render_from_npy.py",
            "--motion", final_out,
            "--audio", audio,
            "--output", base + "_fixed.mp4",
            "--camera_mode", "fixed",
        ])


if __name__ == "__main__":
    main()
PY

cat > scripts/run_mobility_stationary_pipeline.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

mkdir -p logs
mkdir -p output/mobility_stationary
mkdir -p output/mobility_stationary/videos

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "❌ Please run: conda activate edge"
  exit 1
fi

PY="$CONDA_PREFIX/bin/python"

CKPT="${CKPT:-runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt}"
RAW_DB="${RAW_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz}"
MOB_DB="${MOB_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz}"

START_POSE="${START_POSE:-test_keyframes/demo_dyl002_start.npy}"
END_POSE="${END_POSE:-test_keyframes/demo_dyl002_start.npy}"
MUSIC_NAMES="${MUSIC_NAMES:-dunhuangwu2 dunhuangwu3 dunhuangwu4}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. build mobility labels
if [ ! -f "$MOB_DB" ]; then
  echo "==== Building mobility labels ===="
  "$PY" mobility_unit_labels.py \
    --input "$RAW_DB" \
    --output "$MOB_DB" \
    --report "${MOB_DB%.npz}.report.json" \
    2>&1 | tee logs/build_mobility_labels.log
else
  echo "Mobility DB exists: $MOB_DB"
fi

# 2. stationary root-locked generation
for M in $MUSIC_NAMES; do
  AUDIO="test_music_bank/${M}.wav"
  if [ ! -f "$AUDIO" ]; then
    echo "skip missing music: $AUDIO"
    continue
  fi

  for MID_COUNT in 0 1 2; do
    for PRIOR in 0.000 0.012 0.020; do
      if [ "$MID_COUNT" = "0" ] && [ "$PRIOR" != "0.000" ]; then
        continue
      fi

      CASE="${M}_stationary_mid${MID_COUNT}_prior${PRIOR}"
      OUT="output/mobility_stationary/${CASE}.npy"

      echo ""
      echo "==== Generate $CASE ===="

      "$PY" generate_mobility_aware.py \
        --mode stationary \
        --checkpoint "$CKPT" \
        --music "$AUDIO" \
        --start_pose "$START_POSE" \
        --end_pose "$END_POSE" \
        --out "$OUT" \
        --mobility_db "$MOB_DB" \
        --mid_count "$MID_COUNT" \
        --mid_keyframe_strength 0.08 \
        --endpoint_keyframe_strength 0.25 \
        --energy_scale 0.5 \
        --context_scale 0.5 \
        --unit_prior_strength "$PRIOR" \
        --unit_prior_features upper \
        --root_lock_after \
        --render \
        --log "logs/${CASE}.log" || true

      # move rendered videos to videos folder as convenience symlinks/copies
      ROOTLOCK="${OUT%.npy}_rootlock"
      [ -f "${ROOTLOCK}_follow.mp4" ] && cp -f "${ROOTLOCK}_follow.mp4" "output/mobility_stationary/videos/${CASE}_follow.mp4"
      [ -f "${ROOTLOCK}_fixed.mp4" ] && cp -f "${ROOTLOCK}_fixed.mp4" "output/mobility_stationary/videos/${CASE}_fixed.mp4"
    done
  done

  # Beat only after content exists; still root-locked.
  for BEAT in 0.01 0.03; do
    CASE="${M}_stationary_mid1_prior012_beat${BEAT}"
    OUT="output/mobility_stationary/${CASE}.npy"

    "$PY" generate_mobility_aware.py \
      --mode stationary \
      --checkpoint "$CKPT" \
      --music "$AUDIO" \
      --start_pose "$START_POSE" \
      --end_pose "$END_POSE" \
      --out "$OUT" \
      --mobility_db "$MOB_DB" \
      --mid_count 1 \
      --mid_keyframe_strength 0.08 \
      --endpoint_keyframe_strength 0.25 \
      --beat_weight "$BEAT" \
      --energy_scale 0.5 \
      --context_scale 0.5 \
      --unit_prior_strength 0.012 \
      --unit_prior_features upper \
      --root_lock_after \
      --render \
      --log "logs/${CASE}.log" || true

    ROOTLOCK="${OUT%.npy}_rootlock"
    [ -f "${ROOTLOCK}_follow.mp4" ] && cp -f "${ROOTLOCK}_follow.mp4" "output/mobility_stationary/videos/${CASE}_follow.mp4"
    [ -f "${ROOTLOCK}_fixed.mp4" ] && cp -f "${ROOTLOCK}_fixed.mp4" "output/mobility_stationary/videos/${CASE}_fixed.mp4"
  done
done

# 3. body-centered metrics
"$PY" mobility_motion_utils.py batch-eval \
  --glob "output/mobility_stationary/*_rootlock.npy" \
  --output_csv output/mobility_stationary/body_centered_metrics.csv

echo ""
echo "✅ Done."
echo "Metrics: output/mobility_stationary/body_centered_metrics.csv"
echo "Videos : output/mobility_stationary/videos"
SH

cat > scripts/run_mobility_trajectory_pipeline.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
mkdir -p logs output/mobility_trajectory output/mobility_trajectory/videos

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "❌ Please run: conda activate edge"
  exit 1
fi

PY="$CONDA_PREFIX/bin/python"

CKPT="${CKPT:-runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt}"
RAW_DB="${RAW_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz}"
MOB_DB="${MOB_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz}"

START_POSE="${START_POSE:-test_keyframes/demo_dyl002_start.npy}"
END_POSE="${END_POSE:-test_keyframes/demo_dyl002_end.npy}"
MUSIC="${MUSIC:-test_music_bank/dunhuangwu2.wav}"
TRAJECTORY="${TRAJECTORY:-0,0;0.5,0.7;-0.3,1.2;0,1.6}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$MOB_DB" ]; then
  "$PY" mobility_unit_labels.py \
    --input "$RAW_DB" \
    --output "$MOB_DB" \
    --report "${MOB_DB%.npz}.report.json"
fi

CASE="mobile_units_s_curve"
OUT="output/mobility_trajectory/${CASE}.npy"

"$PY" generate_mobility_aware.py \
  --mode trajectory \
  --checkpoint "$CKPT" \
  --music "$MUSIC" \
  --start_pose "$START_POSE" \
  --end_pose "$END_POSE" \
  --out "$OUT" \
  --mobility_db "$MOB_DB" \
  --trajectory "$TRAJECTORY" \
  --mid_count 2 \
  --mid_keyframe_strength 0.10 \
  --endpoint_keyframe_strength 1.0 \
  --energy_scale 0.5 \
  --context_scale 0.5 \
  --unit_prior_strength 0.012 \
  --unit_prior_features upper+torso \
  --render \
  --log "logs/${CASE}.log"

cp -f "output/mobility_trajectory/${CASE}_follow.mp4" "output/mobility_trajectory/videos/${CASE}_follow.mp4" 2>/dev/null || true
cp -f "output/mobility_trajectory/${CASE}_fixed.mp4" "output/mobility_trajectory/videos/${CASE}_fixed.mp4" 2>/dev/null || true

echo "✅ Mobility-aware trajectory run done."
SH

chmod +x mobility_unit_labels.py mobility_aware_selector.py mobility_motion_utils.py generate_mobility_aware.py
chmod +x scripts/run_mobility_stationary_pipeline.sh scripts/run_mobility_trajectory_pipeline.sh

echo "✅ Installed mobility-aware ChoreoRAG files:"
echo "  mobility_unit_labels.py"
echo "  mobility_aware_selector.py"
echo "  mobility_motion_utils.py"
echo "  generate_mobility_aware.py"
echo "  scripts/run_mobility_stationary_pipeline.sh"
echo "  scripts/run_mobility_trajectory_pipeline.sh"
