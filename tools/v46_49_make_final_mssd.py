#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a strict final MSSD for V46.49.

Formal experiment:
  --from_v26_report output/.../dunhuangwu2_v26.schedule_report.json

Controlled ablation:
  --from_previous_report output/.../previous_generation.report.json

The tool never silently promotes a weak train_semantic sidecar to a final plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tools.v46_38_music_action_descriptor import (
    MSSD_SCHEMA_VERSION,
    build_descriptor_object,
    normalize_slot,
)


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slots_from_v26(report: dict, fps: float, source: str):
    schedule = report.get("schedule", [])
    if not isinstance(schedule, list) or not schedule:
        raise RuntimeError(f"V26 report contains no schedule: {source}")
    slots = []
    cursor = 0.0
    meta = {
        "usage": "generate_schedule",
        "is_final_schedule": True,
        "slot_source": "v21_router_v26_planner",
        "fps": fps,
        "raw_schedule_json": source,
        "provenance": {
            "builder": "tools/v46_49_make_final_mssd.py",
            "source_type": "v26_schedule_report",
            "source": source,
            "strict_generation_descriptor": True,
        },
    }
    for i, row0 in enumerate(schedule):
        row = dict(row0)
        target = row.get(
            "allocated_phrase_total",
            row.get("music_length", row.get("target_frames")),
        )
        if target is None:
            target = max(1, int(round(float(row.get("duration", 4.0)) * fps)))
        target = int(round(float(target)))
        row["target_frames"] = target
        row["start"] = cursor
        row["end"] = cursor + target / fps
        row["duration"] = target / fps
        row.setdefault("music_alignment_label", row.get("music_event", row.get("motion_event", "lyrical_flow")))
        row.setdefault("music_semantic_top_label", row["music_alignment_label"])
        row.setdefault("slot_source", meta["slot_source"])
        slot, _ = normalize_slot(row, meta, i, fps=fps, source_path=source)
        slot["start"] = slot["start_sec"] = cursor
        slot["end"] = slot["end_sec"] = cursor + target / fps
        slot["duration"] = slot["duration_sec"] = target / fps
        slot["start_frame"] = int(round(cursor * fps))
        slot["end_frame"] = slot["start_frame"] + target
        slot["target_frames"] = target
        slots.append(slot)
        cursor += target / fps
    return slots, meta


def slots_from_previous(report: dict, fps: float, source: str):
    slots = report.get("slots", [])
    if not isinstance(slots, list) or not slots:
        raise RuntimeError(f"Previous generation report contains no slots: {source}")
    slots = [dict(s) for s in slots]
    meta = {
        "usage": "generate_schedule",
        "is_final_schedule": True,
        "slot_source": "v46_49_controlled_previous_schedule",
        "fps": fps,
        "provenance": {
            "builder": "tools/v46_49_make_final_mssd.py",
            "source_type": "previous_generation_report",
            "source": source,
            "controlled_ablation": True,
            "strict_generation_descriptor": True,
        },
    }
    out = []
    cursor = 0.0
    for i, row in enumerate(slots):
        target = int(row.get("target_frames", round(float(row.get("duration", 4.0)) * fps)))
        row["target_frames"] = max(1, target)
        row.setdefault("start", cursor)
        row.setdefault("end", cursor + target / fps)
        row.setdefault("duration", target / fps)
        slot, _ = normalize_slot(row, meta, i, fps=fps, source_path=source)
        out.append(slot)
        cursor = float(slot["end"])
    return out, meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--from_v26_report")
    group.add_argument("--from_previous_report")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(argv)

    if args.from_v26_report:
        source = args.from_v26_report
        slots, meta = slots_from_v26(load_json(source), args.fps, source)
    else:
        source = args.from_previous_report
        slots, meta = slots_from_previous(load_json(source), args.fps, source)

    obj = build_descriptor_object(args.audio, slots, meta)
    obj["descriptor_schema_version"] = MSSD_SCHEMA_VERSION
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "usage": obj.get("usage"),
        "is_final_schedule": obj.get("is_final_schedule"),
        "slot_source": obj.get("slot_source"),
        "num_slots": obj.get("num_slots"),
        "total_target_frames": obj.get("total_target_frames"),
        "last_slot_frames": obj.get("slots", [{}])[-1].get("target_frames"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
