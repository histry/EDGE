#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V30 music-motion geometric alignment data.

Preferred supervision is an explicit phrase-event pair manifest (JSON/JSONL).
Each record may contain:
  audio, start_frame, end_frame, event_index, negative_event_index, group

The builder extracts a real CLAP phrase embedding when possible and pairs it
with the indexed motion descriptor/MMR embedding.  A legacy V21 router triplet
NPZ can be mixed in as weak supervision, but strict publication runs should
require a minimum number of explicit CLAP-valid pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v26_hierarchical_graph_scheduler import build_hierarchy_features
from tools.v27_deep_music_features import (
    _try_clap_phrase_embedding,
    phrase_rule_semantic,
)


def _load_records(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".jsonl":
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
        return rows
    value = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("pairs", value.get("items", []))
    if not isinstance(value, list):
        raise ValueError("Pair manifest must be JSON list/JSONL")
    return [dict(row) for row in value]


def _pad_or_trim(values: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(x) >= dim:
        return x[:dim].astype(np.float32)
    return np.pad(x, (0, dim - len(x))).astype(np.float32)


def _normalise(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32).reshape(-1)
    return values / max(float(np.linalg.norm(values)), 1e-8)


def _hard_negative(
    positive: int,
    hierarchy_raw: np.ndarray,
    mmr: np.ndarray,
    rng: np.random.Generator,
) -> int:
    body = int(np.argmax(hierarchy_raw[positive, :6]))
    candidates = np.flatnonzero(np.argmax(hierarchy_raw[:, :6], axis=1) == body)
    candidates = candidates[candidates != positive]
    if len(candidates) == 0:
        candidates = np.arange(len(hierarchy_raw))
        candidates = candidates[candidates != positive]
    sample = rng.choice(candidates, size=min(96, len(candidates)), replace=False)
    similarity = mmr[sample] @ _normalise(mmr[positive])
    # A hard negative should share the broad body category but differ at the
    # detailed gesture level. Select a high but non-identical similarity item.
    order = np.argsort(similarity)[::-1]
    return int(sample[order[min(4, len(order) - 1)]])


def _phrase_from_record(record: Mapping[str, Any]) -> Any:
    start = int(record.get("start_frame", record.get("start", 0)))
    end = int(record.get("end_frame", record.get("end", start + 90)))
    return SimpleNamespace(
        start=start,
        end=max(end, start + 8),
        length=max(end - start, 8),
        music_event=str(record.get("music_event", "neutral_flow")),
        energy=float(record.get("energy", 0.5)),
        onset=float(record.get("onset", 0.0)),
        beat_density=float(record.get("beat_density", 0.0)),
        tension=float(record.get("tension", 0.0)),
        calmness=float(record.get("calmness", 0.0)),
        boundary_accent_strength=float(record.get("boundary_accent_strength", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--pair_manifest", default="")
    parser.add_argument("--router_data", default="")
    parser.add_argument("--clap_model", default="clap")
    parser.add_argument("--clap_dim", type=int, default=512)
    parser.add_argument("--router_weight", type=float, default=0.35)
    parser.add_argument("--require_explicit_pairs", type=int, default=0)
    parser.add_argument("--require_clap_valid_pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _, arrays, items = load_shared_index(
        Path(args.index_json), Path(args.duration_index_npz)
    )
    hierarchy = build_hierarchy_features(arrays, items, hyperbolic_ckpt=None)
    hierarchy_raw = np.asarray(hierarchy["hierarchy_raw"], dtype=np.float32)
    target_radius = np.asarray(hierarchy["hierarchy_radius"], dtype=np.float32)
    body = np.asarray(hierarchy["body_code"], dtype=np.int64)
    centre = np.asarray(hierarchy["center_code"], dtype=np.int64)
    gesture = np.asarray(hierarchy["gesture_code"], dtype=np.int64)
    labels = body * 100 + centre * 10 + gesture

    names = set(arrays.files)
    motion_raw = np.asarray(
        arrays["motion_desc_raw"] if "motion_desc_raw" in names else arrays["motion_desc"],
        dtype=np.float32,
    )
    motion_mmr = np.asarray(arrays["mmr_embed"], dtype=np.float32)
    motion_mmr = motion_mmr / np.maximum(
        np.linalg.norm(motion_mmr, axis=1, keepdims=True), 1e-8
    )

    rows: Dict[str, List[Any]] = {
        key: [] for key in (
            "music_rule", "music_clap", "clap_valid",
            "positive_raw", "positive_mmr",
            "negative_raw", "negative_mmr",
            "hierarchy_label", "target_radius", "positive_id",
            "sample_weight", "group", "source_kind",
        )
    }
    explicit_count = 0
    clap_valid_count = 0

    if args.pair_manifest:
        for row_id, record in enumerate(_load_records(args.pair_manifest)):
            event_index = int(record["event_index"])
            if event_index < 0 or event_index >= len(items):
                continue
            phrase = _phrase_from_record(record)
            audio = Path(str(record["audio"]))
            rule = phrase_rule_semantic(phrase)
            clap, mode = _try_clap_phrase_embedding(audio, phrase, args.clap_model)
            valid = int(
                clap is not None
                and np.asarray(clap).size > 0
                and np.isfinite(np.asarray(clap)).all()
            )
            clap_vector = (
                _pad_or_trim(np.asarray(clap, np.float32), args.clap_dim)
                if valid else np.zeros((args.clap_dim,), np.float32)
            )
            negative = record.get("negative_event_index")
            negative_index = (
                int(negative)
                if negative is not None
                else _hard_negative(event_index, hierarchy_raw, motion_mmr, rng)
            )
            if negative_index < 0 or negative_index >= len(items):
                continue

            rows["music_rule"].append(_pad_or_trim(rule, 12))
            rows["music_clap"].append(clap_vector)
            rows["clap_valid"].append(float(valid))
            rows["positive_raw"].append(motion_raw[event_index])
            rows["positive_mmr"].append(motion_mmr[event_index])
            rows["negative_raw"].append(motion_raw[negative_index])
            rows["negative_mmr"].append(motion_mmr[negative_index])
            rows["hierarchy_label"].append(int(labels[event_index]))
            rows["target_radius"].append(float(target_radius[event_index]))
            rows["positive_id"].append(int(event_index))
            rows["sample_weight"].append(float(record.get("weight", 1.0)))
            rows["group"].append(str(record.get("group", audio.stem)))
            rows["source_kind"].append(f"explicit:{mode}")
            explicit_count += 1
            clap_valid_count += valid

    if args.router_data:
        router = np.load(args.router_data, allow_pickle=True)
        music = np.asarray(router["music"], np.float32)
        positive = np.asarray(router["positive"], np.float32)
        negative = np.asarray(router["negative"], np.float32)
        group_values = (
            np.asarray(router["group"], dtype=object)
            if "group" in router.files
            else np.asarray([f"router:{i // 32}" for i in range(len(music))], dtype=object)
        )
        for i in range(len(music)):
            p = _pad_or_trim(positive[i], motion_raw.shape[1])
            n = _pad_or_trim(negative[i], motion_raw.shape[1])
            rows["music_rule"].append(_pad_or_trim(music[i], 12))
            rows["music_clap"].append(np.zeros((args.clap_dim,), np.float32))
            rows["clap_valid"].append(0.0)
            rows["positive_raw"].append(p)
            rows["positive_mmr"].append(np.zeros((motion_mmr.shape[1],), np.float32))
            rows["negative_raw"].append(n)
            rows["negative_mmr"].append(np.zeros((motion_mmr.shape[1],), np.float32))
            rows["hierarchy_label"].append(int(np.argmax(p[:6]) * 100))
            rows["target_radius"].append(float(np.clip(0.2 + 0.6 * np.mean(np.abs(p)), 0.08, 0.92)))
            rows["positive_id"].append(int(10_000_000 + i))
            rows["sample_weight"].append(float(args.router_weight))
            rows["group"].append(str(group_values[i]))
            rows["source_kind"].append("legacy_router_weak")

    if explicit_count < args.require_explicit_pairs:
        raise RuntimeError(
            f"Explicit alignment pairs={explicit_count}, required={args.require_explicit_pairs}"
        )
    if clap_valid_count < args.require_clap_valid_pairs:
        raise RuntimeError(
            f"CLAP-valid explicit pairs={clap_valid_count}, "
            f"required={args.require_clap_valid_pairs}"
        )
    if not rows["music_rule"]:
        raise RuntimeError("No alignment samples were built")

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": "v30_hyperbolic_crossmodal_alignment_dataset",
        "num_samples": len(rows["music_rule"]),
        "explicit_count": explicit_count,
        "clap_valid_explicit_count": clap_valid_count,
        "clap_valid_rate": float(np.mean(rows["clap_valid"])),
        "router_weak_count": int(sum(x == "legacy_router_weak" for x in rows["source_kind"])),
        "motion_raw_dim": int(motion_raw.shape[1]),
        "motion_mmr_dim": int(motion_mmr.shape[1]),
        "clap_dim": int(args.clap_dim),
        "pair_manifest": str(args.pair_manifest),
        "router_data": str(args.router_data),
    }
    np.savez_compressed(
        out,
        music_rule=np.stack(rows["music_rule"]).astype(np.float32),
        music_clap=np.stack(rows["music_clap"]).astype(np.float32),
        clap_valid=np.asarray(rows["clap_valid"], np.float32),
        positive_raw=np.stack(rows["positive_raw"]).astype(np.float32),
        positive_mmr=np.stack(rows["positive_mmr"]).astype(np.float32),
        negative_raw=np.stack(rows["negative_raw"]).astype(np.float32),
        negative_mmr=np.stack(rows["negative_mmr"]).astype(np.float32),
        hierarchy_label=np.asarray(rows["hierarchy_label"], np.int64),
        target_radius=np.asarray(rows["target_radius"], np.float32),
        positive_id=np.asarray(rows["positive_id"], np.int64),
        sample_weight=np.asarray(rows["sample_weight"], np.float32),
        group=np.asarray(rows["group"], dtype=object),
        source_kind=np.asarray(rows["source_kind"], dtype=object),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
