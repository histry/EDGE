#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that V27 CLAP semantic extraction really works.

This tool runs the same phrase_semantic_matrix path used by the scheduler and
writes an auditable JSON report.  It exits non-zero in strict mode if CLAP is
not installed, the checkpoint cannot be loaded, or embeddings fall back to rule
features.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from tools.v27_deep_music_features import phrase_semantic_matrix


@dataclass
class DummyPhrase:
    start: int
    end: int
    length: int
    music_event: str = "climax"
    energy: float = 0.75
    onset: float = 0.55
    beat_density: float = 0.65
    tension: float = 0.70
    calmness: float = 0.15
    boundary_accent_strength: float = 0.55


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _runtime_meta() -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        import torch

        meta["torch_version"] = str(torch.__version__)
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            meta["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        meta["torch_error"] = str(exc)
    for package in ("laion_clap", "msclap", "librosa", "soundfile"):
        try:
            module = __import__(package)
            meta[f"{package}_import"] = True
            meta[f"{package}_version"] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            meta[f"{package}_import"] = False
            meta[f"{package}_error"] = f"{type(exc).__name__}:{exc}"
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="clap")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--cache_dir", default="data/v27_clap_check_cache")
    parser.add_argument("--strict", type=int, default=1)
    parser.add_argument("--min_success", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start_seconds", type=float, default=0.0)
    parser.add_argument("--duration_seconds", type=float, default=6.0)
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        raise RuntimeError(f"Audio file does not exist: {audio}")
    start = int(round(float(args.start_seconds) * float(args.fps)))
    length = max(8, int(round(float(args.duration_seconds) * float(args.fps))))
    phrases = [DummyPhrase(start=start, end=start + length, length=length)]
    semantic, meta = phrase_semantic_matrix(
        audio,
        phrases,
        enabled=True,
        model_name=args.model,
        cache_dir=args.cache_dir,
        require_deep=bool(args.strict),
        min_deep_success=float(args.min_success),
    )
    result = {
        "ok": bool(float(meta.get("deep_success_rate", 0.0)) >= float(args.min_success)),
        "audio": str(audio),
        "semantic_shape": list(semantic.shape),
        "semantic_norm": float(np.linalg.norm(semantic[0])) if len(semantic) else 0.0,
        "semantic_first_values": semantic[0, : min(6, semantic.shape[1])].round(6).tolist() if len(semantic) else [],
        "semantic_meta": meta,
        "runtime": _runtime_meta(),
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    if bool(args.strict) and not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
