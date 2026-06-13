#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a contact-back-injected V34 event library without touching V21 files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import CONTACT, json_safe, load_motion
from tools.v33_event_contacts import EventContactCache


def _safe_name(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return result[:80] or "event"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--event_contact_cache", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    index_path = Path(args.index_json)
    output_dir = Path(args.out_dir)
    output_json = Path(args.out_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    meta, arrays, items = load_shared_index(
        index_path, Path(args.duration_index_npz)
    )
    cache = EventContactCache(args.event_contact_cache)
    if len(cache) != len(items):
        raise RuntimeError(
            f"Contact cache events={len(cache)} != index events={len(items)}"
        )

    new_items = []
    rates = []
    confidences = []
    written = 0
    for index, item in enumerate(items):
        row: Dict[str, Any] = dict(item)
        original_path = Path(str(item.get("pkl", item.get("path", ""))))
        motion = load_motion(original_path).astype(np.float32)
        hard, confidence, origin = cache.get(
            index, item, motion, strict=True
        )
        updated = motion.copy()
        # Back-inject contacts only.  All root and rotation channels remain
        # byte-for-byte identical to the original event library.
        updated[:, CONTACT] = hard

        event_id = str(item.get("event_id", index))
        destination = output_dir / (
            f"{index:05d}_{_safe_name(event_id)}.npy"
        )
        if destination.exists() and not bool(args.overwrite):
            existing = load_motion(destination)
            if existing.shape != updated.shape or not np.array_equal(
                existing, updated
            ):
                raise RuntimeError(
                    f"Existing V34 event does not match cache: {destination}"
                )
        else:
            np.save(destination, updated.astype(np.float32))
            written += 1

        row["v34_original_motion_path"] = str(original_path)
        row["v34_contact_origin"] = origin
        row["v34_contact_label_source"] = (
            "event_level_kinematic_pseudo_contact_v33"
        )
        row["v34_contact_confidence_mean"] = (
            confidence.mean(axis=0).astype(float).tolist()
        )
        row["pkl"] = str(destination)
        row["path"] = str(destination)
        new_items.append(row)
        rates.append(hard.mean(axis=0))
        confidences.append(confidence.mean(axis=0))

    output_meta = dict(meta)
    output_meta["items"] = new_items
    output_meta["v34_contact_back_injection"] = {
        "enabled": True,
        "source_index_json": str(index_path),
        "duration_index_npz": str(args.duration_index_npz),
        "event_contact_cache": str(args.event_contact_cache),
        "num_events": len(new_items),
        "written_files": written,
        "contact_rate": np.mean(rates, axis=0).astype(float).tolist(),
        "confidence_mean": np.mean(confidences, axis=0).astype(float).tolist(),
        "original_library_untouched": True,
    }
    output_json.write_text(
        json.dumps(json_safe(output_meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Strict read-back audit.
    _, _, audit_items = load_shared_index(
        output_json, Path(args.duration_index_npz)
    )
    audit_rates = []
    for row in audit_items:
        loaded = load_motion(Path(str(row["pkl"])))
        audit_rates.append(loaded[:, CONTACT].mean(axis=0))
    summary = {
        "version": "v34_contact_back_injected_event_library",
        "out_json": str(output_json),
        "out_dir": str(output_dir),
        "num_events": len(audit_items),
        "contact_rate_readback": np.mean(
            audit_rates, axis=0
        ).astype(float).tolist(),
        "all_paths_exist": all(
            Path(str(row["pkl"])).is_file() for row in audit_items
        ),
    }
    report = output_json.with_suffix(".audit.json")
    report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[SAVED] {output_json}")
    print(f"[SAVED] {report}")


if __name__ == "__main__":
    main()
