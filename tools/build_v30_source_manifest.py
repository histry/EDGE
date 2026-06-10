#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit event metadata and build a V30 full-sequence source manifest.

The output does not fabricate full source paths. It resolves only paths that
exist and records unresolved groups so the researcher can repair them.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tools.schedule_v21_multi_music import load_shared_index


def bounds(item: Dict[str, Any], fallback: int):
    start = item.get("source_start", item.get("start", item.get("begin", fallback)))
    end = item.get("source_end", item.get("end", fallback))
    try:
        return int(start), int(end)
    except Exception:
        return fallback, fallback


def group_key(item: Dict[str, Any], fallback: int) -> str:
    for name in ("source_id", "video_id", "source_name", "music_id", "source"):
        value = item.get(name)
        if value not in (None, ""):
            return f"{name}:{value}"
    path = str(item.get("pkl", item.get("path", "")))
    return f"event_path:{Path(path).stem if path else fallback}"


def candidates(item: Dict[str, Any], full_root: Path | None) -> List[Path]:
    keys = (
        "source_motion_path", "source_pkl", "full_motion_path",
        "canonical_source_path", "video_motion_path", "source_path",
    )
    values = [Path(str(item[k])) for k in keys if item.get(k)]
    if full_root is not None:
        for name in ("source_id", "video_id", "source_name", "music_id", "source"):
            if item.get(name):
                stem = str(item[name])
                values.extend([
                    full_root / f"{stem}.pkl",
                    full_root / f"{stem}.npy",
                    full_root / stem / "motion.pkl",
                    full_root / stem / "motion.npy",
                ])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--full_motion_root", default="")
    parser.add_argument("--audio_root", default="")
    args = parser.parse_args()

    _, _, items = load_shared_index(
        Path(args.index_json), Path(args.duration_index_npz)
    )
    root = Path(args.full_motion_root) if args.full_motion_root else None
    audio_root = Path(args.audio_root) if args.audio_root else None
    grouped = defaultdict(list)
    for i, item in enumerate(items):
        grouped[group_key(item, i)].append((i, item))

    sources, events, unresolved = {}, {}, []
    for group, entries in sorted(grouped.items()):
        found = None
        checked = []
        for _, item in entries:
            for path in candidates(item, root):
                checked.append(str(path))
                if path.is_file():
                    found = path
                    break
            if found is not None:
                break
        event_rows = []
        for index, item in entries:
            start, end = bounds(item, index)
            event_rows.append({
                "event_index": int(index),
                "event_id": str(item.get("event_id", index)),
                "start": int(start),
                "end": int(end),
            })
        event_rows.sort(key=lambda row: (row["start"], row["end"]))
        source_key = group.split(":", 1)[-1]
        audio = ""
        if audio_root is not None:
            for candidate in (
                audio_root / f"{source_key}.wav",
                audio_root / f"{Path(source_key).stem}.wav",
                audio_root / source_key / "audio.wav",
            ):
                if candidate.is_file():
                    audio = str(candidate)
                    break
        sources[source_key] = {
            "motion": str(found) if found else "",
            "audio": audio,
            "group": group,
        }
        for event in event_rows:
            events[str(event["event_id"])] = {
                "source_id": source_key,
                "source_motion": str(found) if found else "",
                "source_start": int(event["start"]),
                "source_end": int(event["end"]),
            }
        if found is None:
            unresolved.append({
                "group": group,
                "event_count": len(entries),
                "checked_paths": sorted(set(checked)),
            })

    report = {
        "version": "v30_full_sequence_source_manifest",
        "num_sources": len(sources),
        "resolved_sources": int(sum(bool(x.get("motion")) for x in sources.values())),
        "unresolved_sources": len(unresolved),
        "sources": sources,
        "events": events,
        "unresolved": unresolved,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "version", "num_sources", "resolved_sources", "unresolved_sources"
        )
    }, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
