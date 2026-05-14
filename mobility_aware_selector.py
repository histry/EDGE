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
