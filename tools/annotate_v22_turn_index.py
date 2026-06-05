#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Annotate a V21 shared Event-RAG index with turn-speed metadata.

The original JSON/NPZ is not modified.  All arrays are copied to a new V22
prefix and the following arrays are appended:
    contains_turn, turn_count, turn_angle_deg, turn_path_angle_deg,
    peak_yaw_speed_dps, mean_yaw_speed_dps, turn_duration_frames,
    turn_phase_center.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.v21_common import json_safe, load_motion
from tools.v22_turn_utils import summarize_turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_prefix", required=True)
    ap.add_argument("--output_prefix", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min_peak_dps", type=float, default=35.0)
    args = ap.parse_args()

    in_prefix = Path(args.input_prefix)
    input_json = in_prefix.with_suffix(".json")
    input_npz = in_prefix.with_suffix(".npz")
    if not input_json.is_file() or not input_npz.is_file():
        raise FileNotFoundError(f"Missing V21 index: {input_json} / {input_npz}")

    meta = json.loads(input_json.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = list(meta.get("items", []))
    if not items:
        raise RuntimeError("Index JSON contains no items")
    # V21 NPZ 同时包含两类数组：
    #
    # 1. 逐事件数组，例如：
    #    motion_desc [N,D]、entry_pose [N,151]、length [N]
    #
    # 2. 全局统计/元数据数组，例如：
    #    desc_lo [D]、desc_hi [D]
    #
    # 只有逐事件数组的第一维必须等于 items 数量。
    # desc_lo=(12,) 表示12维描述符归一化下界，不是只有12个事件。
    with np.load(input_npz, allow_pickle=True) as old_arrays:
        arrays: Dict[str, np.ndarray] = {
            key: np.array(old_arrays[key], copy=True)
            for key in old_arrays.files
        }

    required_item_arrays = (
        "motion_desc",
        "mmr_embed",
        "entry_pose",
        "exit_pose",
        "entry_vel",
        "exit_vel",
        "length",
    )

    for key in required_item_arrays:
        if key not in arrays:
            raise RuntimeError(
                f"Required item-aligned array missing from {input_npz}: {key}"
            )

        value = np.asarray(arrays[key])

        if value.ndim == 0 or value.shape[0] != len(items):
            raise RuntimeError(
                f"Required item array mismatch: "
                f"{key} shape={value.shape}, items={len(items)}"
            )

    print("NPZ schema:", flush=True)

    for key, value in arrays.items():
        value = np.asarray(value)

        item_aligned = (
            value.ndim >= 1
            and value.shape[0] == len(items)
        )

        scope = "item" if item_aligned else "global/meta"

        print(
            f"  {key:28s} "
            f"shape={str(value.shape):18s} "
            f"scope={scope}",
            flush=True,
        )

    contains_turn: List[float] = []
    turn_count: List[int] = []
    turn_angle: List[float] = []
    turn_path: List[float] = []
    peak_speed: List[float] = []
    mean_speed: List[float] = []
    duration: List[int] = []
    phase: List[float] = []
    updated_items: List[Dict[str, Any]] = []

    failed = 0
    for idx, item in enumerate(items):
        path = Path(str(item.get("pkl", item.get("path", ""))))
        try:
            motion = load_motion(path)
            stats = summarize_turns(motion, fps=args.fps, min_peak_dps=args.min_peak_dps)
        except Exception as exc:
            failed += 1
            print(f"[WARN] {idx}/{len(items)} {path}: {exc}", flush=True)
            stats = {
                "contains_turn": False,
                "turn_count": 0,
                "turn_angle_deg": 0.0,
                "turn_path_angle_deg": 0.0,
                "peak_yaw_speed_dps": 0.0,
                "mean_yaw_speed_dps": 0.0,
                "turn_duration_frames": 0,
                "turn_phase_center": 0.5,
                "turn_events": [],
            }

        record = dict(item)
        record.update(
            {
                "v22_contains_turn": bool(stats["contains_turn"]),
                "v22_turn_count": int(stats["turn_count"]),
                "v22_turn_angle_deg": float(stats["turn_angle_deg"]),
                "v22_turn_path_angle_deg": float(stats["turn_path_angle_deg"]),
                "v22_peak_yaw_speed_dps": float(stats["peak_yaw_speed_dps"]),
                "v22_mean_yaw_speed_dps": float(stats["mean_yaw_speed_dps"]),
                "v22_turn_duration_frames": int(stats["turn_duration_frames"]),
                "v22_turn_phase_center": float(stats["turn_phase_center"]),
                "v22_turn_events": stats.get("turn_events", []),
            }
        )
        updated_items.append(record)
        contains_turn.append(float(bool(stats["contains_turn"])))
        turn_count.append(int(stats["turn_count"]))
        turn_angle.append(float(stats["turn_angle_deg"]))
        turn_path.append(float(stats["turn_path_angle_deg"]))
        peak_speed.append(float(stats["peak_yaw_speed_dps"]))
        mean_speed.append(float(stats["mean_yaw_speed_dps"]))
        duration.append(int(stats["turn_duration_frames"]))
        phase.append(float(stats["turn_phase_center"]))

        if (idx + 1) % 250 == 0 or idx + 1 == len(items):
            print(f"[{idx + 1}/{len(items)}] annotated", flush=True)

    arrays.update(
        {
            "contains_turn": np.asarray(contains_turn, dtype=np.float32),
            "turn_count": np.asarray(turn_count, dtype=np.int16),
            "turn_angle_deg": np.asarray(turn_angle, dtype=np.float32),
            "turn_path_angle_deg": np.asarray(turn_path, dtype=np.float32),
            "peak_yaw_speed_dps": np.asarray(peak_speed, dtype=np.float32),
            "mean_yaw_speed_dps": np.asarray(mean_speed, dtype=np.float32),
            "turn_duration_frames": np.asarray(duration, dtype=np.int16),
            "turn_phase_center": np.asarray(phase, dtype=np.float32),
        }
    )

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_json = out_prefix.with_suffix(".json")
    output_npz = out_prefix.with_suffix(".npz")
    np.savez_compressed(output_npz, **arrays)

    turn_mask = np.asarray(contains_turn, dtype=bool)
    output_meta = dict(meta)
    output_meta.update(
        {
            "version": "v22_turn_aware_shared_event_index",
            "source_index_json": str(input_json),
            "source_index_npz": str(input_npz),
            "arrays": str(output_npz),
            "turn_annotation": {
                "fps": float(args.fps),
                "min_peak_dps": float(args.min_peak_dps),
                "failed": int(failed),
                "contains_turn": int(turn_mask.sum()),
                "turn_ratio": float(turn_mask.mean()),
                "peak_speed_percentiles": (
                    np.percentile(np.asarray(peak_speed)[turn_mask], [10, 50, 90, 95, 100]).tolist()
                    if turn_mask.any()
                    else [0.0] * 5
                ),
            },
            "items": updated_items,
        }
    )
    output_json.write_text(json.dumps(json_safe(output_meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("saved_json:", output_json)
    print("saved_npz :", output_npz)
    print("events    :", len(items))
    print("turns     :", int(turn_mask.sum()))
    print("turn_ratio:", round(float(turn_mask.mean()), 4))
    print("failed    :", failed)


if __name__ == "__main__":
    main()
