#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export schedule-derived music-event pairs for V30 alignment pretraining.

These pairs are weak/pseudo supervision because they originate from an existing
scheduler. They should be manually reviewed or replaced by expert annotations
for the final publication model.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_glob", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--default_weight", type=float, default=0.45)
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for report_path in sorted(glob.glob(args.report_glob)):
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        audio_value = report.get("audio", report.get("music", ""))
        audio = Path(str(audio_value))
        if not audio.is_file():
            stem = Path(report_path).name.replace("_v26.schedule_report.json", "")
            audio = Path(args.audio_dir) / f"{stem}.wav"
        schedule = report.get("schedule", [])
        for slot, item in enumerate(schedule):
            start = int(item.get("start", item.get("source_start", 0)))
            end = int(item.get("end", item.get("source_end", start + 60)))
            event_index = item.get("event_index", item.get("index", item.get("v21_index")))
            if event_index is None:
                continue
            rows.append({
                "audio": str(audio),
                "start_frame": start,
                "end_frame": end,
                "event_index": int(event_index),
                "group": f"schedule:{audio.stem}",
                "weight": float(args.default_weight),
                "weak_supervision": True,
                "slot": slot,
            })

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    print(json.dumps({
        "version": "v30_schedule_weak_pair_manifest",
        "num_pairs": len(rows),
        "warning": "Review/replace with expert pairs for final publication.",
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
