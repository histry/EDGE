#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build optional hierarchical Event-RAG index for V26/V27 graph scheduling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v26_hierarchical_graph_scheduler import build_hierarchy_features


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    meta, arrays, items = load_shared_index(Path(args.index_json), Path(args.duration_index_npz))
    hierarchy = build_hierarchy_features(arrays, items)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **hierarchy)

    summary: Dict[str, Any] = {
        "version": "v26_hierarchical_event_index",
        "source_index_json": str(args.index_json),
        "source_duration_index_npz": str(args.duration_index_npz),
        "num_events": len(items),
        "arrays": {name: list(np.asarray(value).shape) for name, value in hierarchy.items()},
        "body_code_histogram": {
            str(int(code)): int(count)
            for code, count in zip(*np.unique(hierarchy["body_code"], return_counts=True))
        },
        "activity01_percentiles": np.percentile(hierarchy["activity01"], [0, 25, 50, 75, 100]).round(6).tolist(),
        "turn01_percentiles": np.percentile(hierarchy["turn01"], [0, 25, 50, 75, 100]).round(6).tolist(),
        "duration01_percentiles": np.percentile(hierarchy["duration01"], [0, 25, 50, 75, 100]).round(6).tolist(),
        "hierarchy_radius_percentiles": np.percentile(hierarchy["hierarchy_radius"], [0, 25, 50, 75, 100]).round(6).tolist(),
        "hierarchy_embed_norm_percentiles": np.percentile(
            np.linalg.norm(hierarchy["hierarchy_embed"], axis=1),
            [0, 25, 50, 75, 100],
        ).round(6).tolist(),
        "hierarchy_tangent_norm_percentiles": np.percentile(
            np.linalg.norm(hierarchy["hierarchy_tangent"], axis=1),
            [0, 25, 50, 75, 100],
        ).round(6).tolist(),
        "hierarchy_curvature": float(np.asarray(hierarchy["hierarchy_curvature"]).reshape(-1)[0]),
    }
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] hierarchy npz: {out_npz}")
    if args.out_json:
        print(f"[SAVED] hierarchy json: {args.out_json}")


if __name__ == "__main__":
    main()
