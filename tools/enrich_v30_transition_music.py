#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace transition pseudo-music conditions with real V30 audio embeddings.

Input is the V30 transition dataset containing source group/path and source frame
intervals.  Source manifest entries should provide an audio path:
  "sources": {"source_key": {"motion": "...", "audio": "..."}}

Samples without resolvable real audio receive a zero condition. Strict mode can
require a minimum number and ratio of real CLAP-conditioned samples.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from tools.v27_deep_music_features import (
    _deep_mode_success,
    _try_clap_phrase_embedding,
    phrase_rule_semantic,
)
from tools.v30_geometric_alignment import (
    encode_music_numpy,
    load_geometric_aligner,
)


def _manifest(path: str | Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Source manifest must be an object")
    return value


def _group_key(group: str) -> str:
    return str(group).split(":", 1)[-1]


def _audio_path(
    manifest: Mapping[str, Any],
    group: str,
    source_path: str,
    audio_root: str,
) -> Path | None:
    key = _group_key(group)
    source = manifest.get("sources", {}).get(key, {})
    if isinstance(source, Mapping):
        value = source.get("audio", source.get("audio_path", ""))
        if value and Path(str(value)).is_file():
            return Path(str(value))
    if audio_root:
        root = Path(audio_root)
        candidates = [
            root / f"{key}.wav",
            root / f"{Path(source_path).stem}.wav",
            root / key / "audio.wav",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _phrase(start: int, end: int) -> Any:
    return SimpleNamespace(
        start=int(start),
        end=max(int(end), int(start) + 8),
        length=max(int(end) - int(start), 8),
        music_event="neutral_flow",
        energy=0.5,
        onset=0.0,
        beat_density=0.0,
        tension=0.0,
        calmness=0.0,
        boundary_accent_strength=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--alignment_ckpt", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--audio_root", default="")
    parser.add_argument("--clap_model", default="clap")
    parser.add_argument("--device", default="")
    parser.add_argument("--require_real_music_count", type=int, default=0)
    parser.add_argument("--require_real_music_ratio", type=float, default=0.0)
    args = parser.parse_args()

    z = np.load(args.input_npz, allow_pickle=True)
    manifest = _manifest(args.source_manifest)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model, alignment_meta = load_geometric_aligner(args.alignment_ckpt, device=device)
    group = np.asarray(z["start_group"], dtype=object)
    source_path = np.asarray(z["source_path"], dtype=object)
    start_frame = np.asarray(z["source_start_frame"], dtype=np.int32)
    end_frame = np.asarray(z["source_end_frame"], dtype=np.int32)
    real_target = np.asarray(z["real_target"], dtype=np.bool_)

    semantic = np.zeros((len(group), model.config.embed_dim), dtype=np.float32)
    valid = np.zeros((len(group),), dtype=np.bool_)
    modes = np.asarray(["unresolved"] * len(group), dtype=object)
    cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, str]] = {}

    for index in range(len(group)):
        if start_frame[index] < 0 or end_frame[index] <= start_frame[index]:
            continue
        audio = _audio_path(
            manifest, str(group[index]), str(source_path[index]), args.audio_root
        )
        if audio is None:
            continue
        key = (str(audio), int(start_frame[index]), int(end_frame[index]))
        if key not in cache:
            phrase = _phrase(start_frame[index], end_frame[index])
            rule = phrase_rule_semantic(phrase)
            clap, mode = _try_clap_phrase_embedding(audio, phrase, args.clap_model)
            ok = (
                clap is not None
                and np.asarray(clap).size > 0
                and np.isfinite(np.asarray(clap)).all()
            )
            clap_vector = np.zeros((model.config.clap_dim,), dtype=np.float32)
            if ok:
                value = np.asarray(clap, dtype=np.float32).reshape(-1)
                clap_vector[: min(len(value), len(clap_vector))] = value[: len(clap_vector)]
                embedding = encode_music_numpy(
                    model, rule[None], clap_vector[None], device
                )[0]
            else:
                embedding = np.zeros((model.config.embed_dim,), np.float32)
            cache[key] = (embedding.astype(np.float32), str(mode))
        semantic[index], modes[index] = cache[key]
        valid[index] = _deep_mode_success(str(modes[index]))

    real_music = valid & real_target
    count = int(real_music.sum())
    ratio = float(count / max(int(real_target.sum()), 1))
    if count < args.require_real_music_count:
        raise RuntimeError(
            f"Real CLAP-conditioned transition samples={count}, "
            f"required={args.require_real_music_count}"
        )
    if ratio < args.require_real_music_ratio:
        raise RuntimeError(
            f"Real music ratio={ratio:.4f}, required={args.require_real_music_ratio:.4f}"
        )

    arrays = {name: np.asarray(z[name]) for name in z.files if name != "music" and name != "meta"}
    old_meta = json.loads(str(z["meta"].item()))
    meta = {
        **old_meta,
        "version": "v30_continuous_inr_real_audio_conditioned_dataset",
        "music_dim": int(semantic.shape[1]),
        "real_music_count": count,
        "real_music_ratio_among_real_targets": ratio,
        "all_music_valid_count": int(valid.sum()),
        "alignment": alignment_meta,
        "source_manifest": args.source_manifest,
    }
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **arrays,
        music=semantic,
        music_real_valid=valid,
        music_mode=modes,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
