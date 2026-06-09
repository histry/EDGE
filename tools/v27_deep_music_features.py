#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional deep music semantics for V27/V28 Event-RAG.

Precise boundary detection should still use onset/beat/novelty.  This module is
only used for semantic query shaping inside hierarchical Event-RAG.  If a CLAP
implementation is not installed, it returns a deterministic 12D semantic proxy
from the existing phrase-level acoustic features, so experiments remain
reproducible without new dependencies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    return x / max(float(np.linalg.norm(x)), 1e-8)


def _projection_matrix(in_dim: int, out_dim: int = 12) -> np.ndarray:
    seed = int(hashlib.sha1(f"v27_music_semantic_{in_dim}_{out_dim}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
    mat /= np.maximum(np.linalg.norm(mat, axis=0, keepdims=True), 1e-8)
    return mat


def phrase_rule_semantic(phrase: Any) -> np.ndarray:
    """Return a 12D semantic proxy aligned with hierarchy raw features."""
    music_event = str(getattr(phrase, "music_event", "neutral_flow"))
    group_map = {
        "calm_flow": 1,
        "release": 1,
        "neutral_flow": 2,
        "build_up": 3,
        "climax": 3,
        "accent": 4,
        "section_change": 5,
    }
    group = int(group_map.get(music_event, 2))
    coarse = np.zeros((6,), dtype=np.float32)
    coarse[np.clip(group, 0, 5)] = 1.0
    energy = float(getattr(phrase, "energy", 0.5))
    onset = float(getattr(phrase, "onset", 0.0))
    beat = float(getattr(phrase, "beat_density", 0.0))
    tension = float(getattr(phrase, "tension", 0.0))
    calm = float(getattr(phrase, "calmness", 0.0))
    boundary = float(getattr(phrase, "boundary_accent_strength", 0.0))
    activity = np.clip(0.42 * energy + 0.24 * beat + 0.18 * onset + 0.16 * tension - 0.22 * calm, 0.0, 1.0)
    turn = np.clip(0.44 * tension + 0.26 * boundary + 0.18 * beat + (0.18 if music_event in {"climax", "section_change"} else 0.0), 0.0, 1.0)
    duration = np.clip((float(getattr(phrase, "length", 60)) - 24.0) / 120.0, 0.0, 1.0)
    style = np.clip(0.50 + 0.30 * calm + 0.20 * tension, 0.0, 1.0)
    quality = np.clip(0.55 + 0.25 * beat + 0.20 * boundary, 0.0, 1.0)
    safety = np.clip(0.68 + 0.20 * calm - 0.15 * onset, 0.0, 1.0)
    return _normalize(np.concatenate([coarse, np.asarray([activity, turn, duration, style, quality, safety], dtype=np.float32)]))


def _try_clap_phrase_embedding(audio_path: Path, phrase: Any, model_name: str) -> Tuple[np.ndarray | None, str]:
    """Best-effort CLAP/MSCLAP phrase embedding.

    This deliberately avoids hard dependency imports at module load.  Supported
    installations differ across labs; when unavailable, callers receive None and
    fall back to rule semantics.
    """
    try:
        import librosa  # type: ignore
    except Exception as exc:
        return None, f"librosa_unavailable:{exc}"

    start_sec = float(getattr(phrase, "start", 0)) / 30.0
    end_sec = max(start_sec + 0.25, float(getattr(phrase, "end", getattr(phrase, "start", 0) + 30)) / 30.0)
    try:
        y, sr = librosa.load(str(audio_path), sr=48000, mono=True, offset=start_sec, duration=end_sec - start_sec)
    except Exception as exc:
        return None, f"audio_load_failed:{exc}"
    if y.size < 256:
        return None, "audio_too_short"

    if model_name.lower() in {"laion_clap", "clap"}:
        try:
            import laion_clap  # type: ignore

            model = laion_clap.CLAP_Module(enable_fusion=False)
            emb = model.get_audio_embedding_from_data(x=y[None, :], use_tensor=False)
            return np.asarray(emb, dtype=np.float32).reshape(-1), "laion_clap"
        except Exception as exc:
            return None, f"laion_clap_failed:{exc}"
    if model_name.lower() in {"msclap", "microsoft_clap"}:
        try:
            from msclap import CLAP  # type: ignore

            model = CLAP(version="2023", use_cuda=False)
            # Some msclap versions only expose file-based inference.
            emb = model.get_audio_embeddings([str(audio_path)])
            return np.asarray(emb, dtype=np.float32).reshape(-1), "msclap_file"
        except Exception as exc:
            return None, f"msclap_failed:{exc}"
    return None, f"unsupported_model:{model_name}"


def phrase_semantic_matrix(
    audio_path: str | Path,
    phrases: Sequence[Any],
    enabled: bool = False,
    model_name: str = "clap",
    cache_dir: str | Path | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    audio = Path(audio_path)
    cache_path = None
    if cache_dir:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        cache_path = cache / f"{audio.stem}_v27_semantic_{model_name}_{len(phrases)}.npz"
        if cache_path.is_file():
            data = np.load(cache_path, allow_pickle=True)
            return np.asarray(data["semantic"], dtype=np.float32), json.loads(str(data["meta"].item()))

    rows = []
    modes = []
    for phrase in phrases:
        rule = phrase_rule_semantic(phrase)
        if enabled:
            emb, mode = _try_clap_phrase_embedding(audio, phrase, model_name)
            if emb is not None and emb.size > 0:
                proj = _projection_matrix(int(emb.size), 12)
                deep = _normalize(np.asarray(emb, dtype=np.float32).reshape(1, -1) @ proj)[0]
                rows.append(_normalize(0.55 * rule + 0.45 * deep))
                modes.append(mode)
                continue
            modes.append(mode)
        else:
            modes.append("disabled_rule_proxy")
        rows.append(rule)
    semantic = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 12), dtype=np.float32)
    meta = {
        "audio": str(audio_path),
        "enabled": bool(enabled),
        "model_name": str(model_name),
        "num_phrases": int(len(phrases)),
        "modes": modes,
        "feature_dim": int(semantic.shape[1]) if semantic.ndim == 2 else 0,
    }
    if cache_path is not None:
        np.savez_compressed(cache_path, semantic=semantic, meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object))
    return semantic, meta
