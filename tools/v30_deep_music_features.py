#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V30 trainable CLAP-to-motion Poincare phrase semantics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

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

_CACHE: Dict[str, Tuple[Any, Dict[str, Any]]] = {}


def _model(path: str, device: str):
    key = f"{path}|{device}"
    if key not in _CACHE:
        _CACHE[key] = load_geometric_aligner(path, device=device)
    return _CACHE[key]


def phrase_geometric_matrix(
    audio_path: str | Path,
    phrases: Sequence[Any],
    enabled: bool = True,
    model_name: str = "clap",
    cache_dir: str | Path | None = None,
    require_deep: bool = True,
    min_deep_success: float = 0.80,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    checkpoint = os.environ.get("V30_ALIGNMENT_CKPT", "").strip()
    if not checkpoint:
        raise RuntimeError("V30_ALIGNMENT_CKPT is required")
    device = os.environ.get(
        "V30_ALIGNMENT_DEVICE",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )
    model, model_meta = _model(checkpoint, device)
    cache_path = None
    if cache_dir:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha1(
            f"{audio_path}|{len(phrases)}|{checkpoint}".encode("utf-8")
        ).hexdigest()[:12]
        cache_path = cache / f"{Path(audio_path).stem}_v30_geo_{signature}.npz"
        if cache_path.is_file():
            z = np.load(cache_path, allow_pickle=True)
            return np.asarray(z["semantic"], np.float32), json.loads(str(z["meta"].item()))

    rules, claps, modes = [], [], []
    clap_dim = int(model.config.clap_dim)
    for phrase in phrases:
        rule = phrase_rule_semantic(phrase)
        embedding, mode = _try_clap_phrase_embedding(
            Path(audio_path), phrase, model_name
        ) if enabled else (None, "disabled")
        valid = (
            embedding is not None
            and np.asarray(embedding).size > 0
            and np.isfinite(np.asarray(embedding)).all()
        )
        clap = np.zeros((clap_dim,), np.float32)
        if valid:
            values = np.asarray(embedding, np.float32).reshape(-1)
            clap[: min(clap_dim, len(values))] = values[:clap_dim]
        rules.append(rule)
        claps.append(clap)
        modes.append(mode)

    semantic = encode_music_numpy(
        model,
        np.stack(rules).astype(np.float32),
        np.stack(claps).astype(np.float32),
        device,
    )
    success = sum(_deep_mode_success(mode) for mode in modes)
    rate = float(success / max(len(modes), 1))
    if require_deep and rate < float(min_deep_success):
        raise RuntimeError(
            f"V30 CLAP success rate={rate:.3f}, required={min_deep_success:.3f}"
        )
    meta = {
        "version": "v30_trainable_hyperbolic_music_semantics",
        "audio": str(audio_path),
        "enabled": bool(enabled),
        "model_name": model_name,
        "num_phrases": len(phrases),
        "deep_success_count": int(success),
        "fallback_count": int(len(phrases) - success),
        "deep_success_rate": rate,
        "modes": [str(x) for x in modes],
        "unique_modes": sorted(set(str(x).split(":")[0] for x in modes)),
        "feature_dim": int(semantic.shape[1]),
        "alignment": model_meta,
        "backend_meta": [{
            "backend": "v30_geometric_alignment",
            "checkpoint": checkpoint,
            "device": device,
            "curvature": float(model.config.curvature),
        }],
    }
    if cache_path:
        np.savez_compressed(
            cache_path,
            semantic=semantic,
            meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
        )
    return semantic, meta
