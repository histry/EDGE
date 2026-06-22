#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a source-aware balanced JSON+NPZ Event-RAG index.

This tool is intentionally post-hoc: it reads an already built shared index
JSON and its aligned NPZ, enriches items with dancer/repeat/category metadata,
selects a quality-weighted balanced subset, and writes a new aligned JSON+NPZ.

Use it when the motion-only Dunhuang BVH corpus is organized as:

    dancer_repeat_Take_category.bvh

Example:

    dyl_002_Take_003.bvh

means dancer=dyl, repeat=002, category=Take_003.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from tools.v21_common import json_safe, load_json_items
from tools.v34_source_aware_rag import source_aware_select, source_distribution


def _filter_npz_arrays(npz: Any, keep_indices: np.ndarray, n_items: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.ndim >= 1 and arr.shape[0] == n_items:
            out[key] = arr[keep_indices]
        else:
            # Scalars and global calibration arrays are copied unchanged.
            out[key] = arr
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--index_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--audit_json", default="")
    parser.add_argument("--cap_per_source_uid", type=int, default=64)
    parser.add_argument("--category_cap_factor", type=float, default=1.35)
    parser.add_argument("--repeat_cap_factor", type=float, default=1.60)
    parser.add_argument("--dancer_cap_factor", type=float, default=1.50)
    parser.add_argument("--max_events", type=int, default=0)
    args = parser.parse_args()

    meta, items = load_json_items(args.index_json)
    arrays = np.load(Path(args.index_npz), allow_pickle=True)
    n = len(items)
    for key in ("motion_desc", "mmr_embed", "entry_pose", "exit_pose", "entry_vel", "exit_vel", "length"):
        if key in arrays.files and len(arrays[key]) != n:
            raise RuntimeError(
                f"Cannot build source-aware index: NPZ key {key} has {len(arrays[key])}, JSON has {n}"
            )

    selected, balance_report = source_aware_select(
        items,
        cap_per_source_uid=args.cap_per_source_uid,
        category_cap_factor=args.category_cap_factor,
        repeat_cap_factor=args.repeat_cap_factor,
        dancer_cap_factor=args.dancer_cap_factor,
        max_events=args.max_events,
    )
    keep = np.asarray([int(row["_source_aware_original_index"]) for row in selected], dtype=np.int64)

    cleaned = []
    for new_idx, row in enumerate(selected):
        out = {k: v for k, v in row.items() if not k.startswith("_source_aware_")}
        out["v34_source_aware_index"] = int(new_idx)
        out["v21_index"] = int(new_idx)
        cleaned.append(out)

    out_meta = dict(meta)
    out_meta["items"] = cleaned
    out_meta["arrays"] = str(args.out_npz)
    out_meta["source_aware_balancing"] = balance_report
    out_meta["version"] = str(meta.get("version", "shared_event_index")) + "_source_aware_balanced"
    out_meta["source_index_json"] = str(args.index_json)
    out_meta["source_index_npz"] = str(args.index_npz)

    out_json = Path(args.out_json)
    out_npz = Path(args.out_npz)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    filtered = _filter_npz_arrays(arrays, keep, n)
    np.savez_compressed(out_npz, **filtered)
    out_json.write_text(json.dumps(json_safe(out_meta), ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "input_json": str(args.index_json),
        "input_npz": str(args.index_npz),
        "out_json": str(out_json),
        "out_npz": str(out_npz),
        "num_before": int(n),
        "num_after": int(len(cleaned)),
        "before_distribution": source_distribution(items),
        "after_distribution": source_distribution(cleaned),
        "balance_report": balance_report,
        "kept_original_indices_preview": keep[:50].astype(int).tolist(),
    }
    audit_json = Path(args.audit_json) if args.audit_json else out_json.with_suffix(".audit.json")
    audit_json.write_text(json.dumps(json_safe(audit), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(json_safe({
        "saved_json": str(out_json),
        "saved_npz": str(out_npz),
        "audit_json": str(audit_json),
        "num_before": n,
        "num_after": len(cleaned),
        "after": audit["after_distribution"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
