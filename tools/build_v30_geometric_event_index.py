#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V30 event index with trainable hyperbolic cross-modal embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.schedule_v21_multi_music import load_shared_index
from tools.v26_hierarchical_graph_scheduler import build_hierarchy_features
from tools.v30_geometric_alignment import (
    encode_motion_numpy,
    load_geometric_aligner,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--alignment_ckpt", required=True)
    parser.add_argument("--hyperbolic_ckpt", default="")
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    meta, arrays, items = load_shared_index(
        Path(args.index_json), Path(args.duration_index_npz)
    )
    hierarchy = build_hierarchy_features(
        arrays, items,
        hyperbolic_ckpt=args.hyperbolic_ckpt or None,
    )
    names = set(arrays.files)
    raw = np.asarray(
        arrays["motion_desc_raw"] if "motion_desc_raw" in names else arrays["motion_desc"],
        np.float32,
    )
    mmr = np.asarray(arrays["mmr_embed"], np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, align_meta = load_geometric_aligner(args.alignment_ckpt, device=device)

    embeddings = []
    for start in range(0, len(items), args.batch_size):
        embeddings.append(
            encode_motion_numpy(
                model,
                raw[start : start + args.batch_size],
                mmr[start : start + args.batch_size],
                device,
            )
        )
    crossmodal = np.concatenate(embeddings, axis=0).astype(np.float32)

    output = {name: np.asarray(value) for name, value in hierarchy.items()}
    output["v30_crossmodal_embed"] = crossmodal
    output["v30_crossmodal_curvature"] = np.asarray(
        [float(model.config.curvature)], np.float32
    )
    output["v30_alignment_checkpoint"] = np.asarray(
        [str(args.alignment_ckpt)], dtype=object
    )
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **output)

    report = {
        "version": "v30_geometric_event_index",
        "num_events": len(items),
        "embed_dim": int(crossmodal.shape[1]),
        "alignment": align_meta,
        "source_index_json": str(args.index_json),
        "source_duration_index_npz": str(args.duration_index_npz),
        "arrays": str(out_npz),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
