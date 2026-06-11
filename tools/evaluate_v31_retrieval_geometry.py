#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Held-out Euclidean-vs-Poincare retrieval audit for V31.

Geometric retrieval should remain disabled unless this audit demonstrates a
repeatable held-out advantage. The tool reports Recall@K and MRR and writes an
explicit enable recommendation; it does not assume that hyperbolic geometry is
better by construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch

from tools.v27_deep_music_features import (
    _try_clap_phrase_embedding,
    phrase_rule_semantic,
)
from tools.v30_geometric_alignment import (
    encode_music_numpy,
    load_geometric_aligner,
    poincare_distance_pairwise,
)


def load_pairs(path: str | Path) -> List[Dict[str, object]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return [
            dict(json.loads(line))
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("pairs", value.get("items", []))
    return [dict(row) for row in value]


def phrase(record: Dict[str, object]) -> SimpleNamespace:
    start = int(record.get("start_frame", record.get("start", 0)))
    end = int(record.get("end_frame", record.get("end", start + 60)))
    return SimpleNamespace(
        start=start,
        end=end,
        length=max(end - start, 8),
        music_event=str(record.get("music_event", "neutral_flow")),
        energy=float(record.get("energy", 0.5)),
        onset=float(record.get("onset", 0.0)),
        beat_density=float(record.get("beat_density", 0.0)),
        tension=float(record.get("tension", 0.0)),
        calmness=float(record.get("calmness", 0.0)),
        boundary_accent_strength=float(
            record.get("boundary_accent_strength", 0.0)
        ),
    )


def metrics(ranks: np.ndarray) -> Dict[str, float]:
    return {
        "recall_at_1": float(np.mean(ranks < 1)),
        "recall_at_5": float(np.mean(ranks < 5)),
        "recall_at_10": float(np.mean(ranks < 10)),
        "mrr": float(np.mean(1.0 / (ranks + 1.0))),
        "median_rank": float(np.median(ranks + 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_manifest", required=True)
    parser.add_argument("--event_index_npz", required=True)
    parser.add_argument("--alignment_ckpt", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--clap_model", default="clap")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model, model_meta = load_geometric_aligner(
        args.alignment_ckpt, device=device
    )
    index = np.load(args.event_index_npz, allow_pickle=True)
    if "v30_crossmodal_embed" not in index.files:
        raise RuntimeError("Event index lacks v30_crossmodal_embed")
    event = np.asarray(index["v30_crossmodal_embed"], np.float32)
    curvature = float(np.asarray(
        index.get("v30_crossmodal_curvature", [model.config.curvature])
    ).reshape(-1)[0])

    ranks_hyperbolic, ranks_euclidean = [], []
    valid_rows = 0
    for record in load_pairs(args.pair_manifest):
        positive = int(record["event_index"])
        if positive < 0 or positive >= len(event):
            continue
        ph = phrase(record)
        embedding, mode = _try_clap_phrase_embedding(
            Path(str(record["audio"])), ph, args.clap_model
        )
        if embedding is None or not np.isfinite(embedding).all():
            continue
        clap = np.zeros((model.config.clap_dim,), np.float32)
        value = np.asarray(embedding, np.float32).reshape(-1)
        clap[: min(len(value), len(clap))] = value[: len(clap)]
        query = encode_music_numpy(
            model,
            phrase_rule_semantic(ph)[None],
            clap[None],
            device,
        )[0]
        with torch.no_grad():
            distance_h = poincare_distance_pairwise(
                torch.from_numpy(query).reshape(1, -1),
                torch.from_numpy(event),
                curvature,
            )[0].cpu().numpy()
        distance_e = np.linalg.norm(event - query[None], axis=1)
        order_h = np.argsort(distance_h)
        order_e = np.argsort(distance_e)
        ranks_hyperbolic.append(int(np.flatnonzero(order_h == positive)[0]))
        ranks_euclidean.append(int(np.flatnonzero(order_e == positive)[0]))
        valid_rows += 1

    if valid_rows < 20:
        raise RuntimeError(
            f"Only {valid_rows} valid held-out pairs; at least 20 required"
        )
    h = np.asarray(ranks_hyperbolic, np.int64)
    e = np.asarray(ranks_euclidean, np.int64)
    rng = np.random.default_rng(args.seed)
    differences = []
    for _ in range(args.bootstrap):
        sample = rng.integers(0, len(h), size=len(h))
        differences.append(
            np.mean(h[sample] < 10) - np.mean(e[sample] < 10)
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    hyperbolic = metrics(h)
    euclidean = metrics(e)
    enable = bool(
        hyperbolic["recall_at_10"] > euclidean["recall_at_10"]
        and low > 0.0
    )
    result = {
        "version": "v31_retrieval_geometry_audit",
        "valid_pairs": valid_rows,
        "hyperbolic": hyperbolic,
        "euclidean_same_embedding": euclidean,
        "recall10_difference": (
            hyperbolic["recall_at_10"] - euclidean["recall_at_10"]
        ),
        "recall10_bootstrap_95ci": [float(low), float(high)],
        "recommend_enable_geometric_retrieval": enable,
        "recommended_weight": 0.25 if enable else 0.0,
        "alignment": model_meta,
        "warning": (
            "This compares metrics on held-out pairs using the same learned "
            "embedding. Final papers should repeat over multiple data splits."
        ),
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
