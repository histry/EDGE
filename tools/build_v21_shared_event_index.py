#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a single scalable shared Event-RAG index for arbitrary music queries.

This script does NOT split the database by music. It enriches a style-safe event
JSON with compact motion descriptors, MMR embeddings, endpoint signatures and
family identifiers. The result is one JSON + one NPZ shared by all songs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.v21_common import (
    EVENT_TYPES,
    family_id,
    json_safe,
    load_json_items,
    load_motion,
    motion_descriptor_raw,
    motion_mmr_embedding,
    robust_scale,
)


def get_value(item: Dict[str, Any], key: str, default: float = 0.0) -> float:
    if key in item:
        try:
            return float(item[key])
        except (TypeError, ValueError):
            pass
    desc = item.get("descriptor", {})
    if isinstance(desc, dict) and key in desc:
        try:
            return float(desc[key])
        except (TypeError, ValueError):
            pass
    return float(default)


def choose_style_score(item: Dict[str, Any]) -> float:
    for key in (
        "dunhuang_style_score_v20f3",
        "dunhuang_style_score",
        "dunhuang_style_proxy",
        "prototype_similarity_norm",
        "prototype_similarity",
        "visual_score",
        "quality_score",
    ):
        if key in item:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def balanced_top(items: List[Dict[str, Any]], max_events: int) -> List[Dict[str, Any]]:
    if len(items) <= max_events:
        return items
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get("event_type", "neutral_flow"))].append(item)
    weights = {
        "neutral_flow": 0.25,
        "calm_flow": 0.16,
        "support_shift": 0.16,
        "build_up": 0.12,
        "high_tension": 0.12,
        "arm_flourish": 0.08,
        "release": 0.07,
        "pose_hold": 0.04,
    }
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for event_type, group in groups.items():
        group.sort(
            key=lambda x: (
                float(x["_v21_style"]),
                float(x["_v21_quality"]),
                float(x["_v21_safety"]),
            ),
            reverse=True,
        )
        quota = max(1, int(round(max_events * weights.get(event_type, 0.05))))
        selected.extend(group[:quota])
        leftovers.extend(group[quota:])
    if len(selected) < max_events:
        leftovers.sort(
            key=lambda x: (
                float(x["_v21_style"]),
                float(x["_v21_quality"]),
                float(x["_v21_safety"]),
            ),
            reverse=True,
        )
        selected.extend(leftovers[: max_events - len(selected)])
    selected.sort(
        key=lambda x: (
            float(x["_v21_style"]),
            float(x["_v21_quality"]),
            float(x["_v21_safety"]),
        ),
        reverse=True,
    )
    return selected[:max_events]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_db", required=True)
    ap.add_argument("--output_prefix", required=True, help="e.g. data/.../v21_shared_event_index")
    ap.add_argument("--max_events", type=int, default=7000)
    ap.add_argument("--min_style_percentile", type=float, default=10.0)
    ap.add_argument("--min_quality", type=float, default=0.0)
    ap.add_argument("--min_safety", type=float, default=0.0)
    ap.add_argument("--family_span", type=int, default=600)
    ap.add_argument("--mmr_dim", type=int, default=64)
    args = ap.parse_args()

    source_meta, source_items = load_json_items(args.input_db)
    prepared: List[Dict[str, Any]] = []
    style_values = np.asarray([choose_style_score(x) for x in source_items], dtype=np.float32)
    style_threshold = float(np.percentile(style_values, args.min_style_percentile)) if len(style_values) else 0.0

    for item in source_items:
        pkl = Path(str(item.get("pkl", item.get("path", ""))))
        if not pkl.is_file():
            continue
        style = choose_style_score(item)
        quality = get_value(item, "original_quality_score", get_value(item, "quality_score", 0.0))
        safety = get_value(item, "safety_score", quality)
        if style < style_threshold or quality < args.min_quality or safety < args.min_safety:
            continue
        record = dict(item)
        record["_v21_style"] = float(style)
        record["_v21_quality"] = float(quality)
        record["_v21_safety"] = float(safety)
        record["family_id"] = str(item.get("motion_family_id", family_id(item, args.family_span)))
        prepared.append(record)

    prepared = balanced_top(prepared, max(1, args.max_events))
    if not prepared:
        raise RuntimeError("No events survived the shared-index gate")

    kept_meta: List[Dict[str, Any]] = []
    desc_raw: List[np.ndarray] = []
    mmr_embed: List[np.ndarray] = []
    entry_pose: List[np.ndarray] = []
    exit_pose: List[np.ndarray] = []
    entry_vel: List[np.ndarray] = []
    exit_vel: List[np.ndarray] = []
    lengths: List[int] = []
    style_scores: List[float] = []
    quality_scores: List[float] = []
    safety_scores: List[float] = []

    for idx, item in enumerate(prepared):
        pkl = Path(str(item.get("pkl", item.get("path", ""))))
        try:
            motion = load_motion(pkl)
        except Exception as exc:
            print(f"[SKIP] {pkl}: {exc}", flush=True)
            continue
        if len(motion) < 8:
            continue
        velocity = np.diff(motion, axis=0, prepend=motion[:1])
        desc_raw.append(motion_descriptor_raw(motion))
        mmr_embed.append(motion_mmr_embedding(motion, args.mmr_dim))
        entry_pose.append(motion[0].astype(np.float32))
        exit_pose.append(motion[-1].astype(np.float32))
        entry_vel.append(velocity[: min(4, len(velocity))].mean(axis=0).astype(np.float32))
        exit_vel.append(velocity[-min(4, len(velocity)) :].mean(axis=0).astype(np.float32))
        lengths.append(int(len(motion)))
        style_scores.append(float(item["_v21_style"]))
        quality_scores.append(float(item["_v21_quality"]))
        safety_scores.append(float(item["_v21_safety"]))

        clean = {k: v for k, v in item.items() if not k.startswith("_v21_")}
        clean["v21_index"] = len(kept_meta)
        clean["family_id"] = str(item["family_id"])
        clean["event_type"] = str(item.get("event_type", "neutral_flow"))
        clean["v21_style_score"] = float(item["_v21_style"])
        clean["v21_quality_score"] = float(item["_v21_quality"])
        clean["v21_safety_score"] = float(item["_v21_safety"])
        kept_meta.append(clean)

    if not kept_meta:
        raise RuntimeError("No valid motions could be loaded")

    raw = np.stack(desc_raw).astype(np.float32)
    desc, desc_lo, desc_hi = robust_scale(raw)
    # Duration dimension should represent the actual dynamic-unit length on a stable 0..1 scale.
    desc[:, 11] = np.clip((raw[:, 11] - 24.0) / (72.0 - 24.0), 0.0, 1.0)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")

    np.savez_compressed(
        npz_path,
        motion_desc=desc,
        motion_desc_raw=raw,
        desc_lo=desc_lo,
        desc_hi=desc_hi,
        mmr_embed=np.stack(mmr_embed).astype(np.float32),
        entry_pose=np.stack(entry_pose).astype(np.float32),
        exit_pose=np.stack(exit_pose).astype(np.float32),
        entry_vel=np.stack(entry_vel).astype(np.float32),
        exit_vel=np.stack(exit_vel).astype(np.float32),
        length=np.asarray(lengths, dtype=np.int32),
        style_score=np.asarray(style_scores, dtype=np.float32),
        quality_score=np.asarray(quality_scores, dtype=np.float32),
        safety_score=np.asarray(safety_scores, dtype=np.float32),
    )

    report = {
        "version": "v21_shared_style_event_index",
        "input_db": str(args.input_db),
        "arrays": str(npz_path),
        "num_input": len(source_items),
        "num_prepared": len(prepared),
        "num_indexed": len(kept_meta),
        "style_threshold": style_threshold,
        "family_span": int(args.family_span),
        "event_type_counts": dict(Counter(x["event_type"] for x in kept_meta)),
        "descriptor_dims": [
            "energy", "upper", "torso", "lower", "tension", "calmness",
            "support", "build_up", "release", "accent", "phrase_change", "duration",
        ],
        "items": kept_meta,
    }
    json_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("saved_json:", json_path)
    print("saved_npz:", npz_path)
    print("indexed:", len(kept_meta))
    print("event_types:", report["event_type_counts"])
    print("style_threshold:", style_threshold)


if __name__ == "__main__":
    main()
