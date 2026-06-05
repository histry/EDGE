#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract V20 music event stream.

This wrapper first tries to reuse the existing tools.extract_music_emotion_features
module.  It then augments the feature stream with local deltas, next-beat distance
and a frame-wise event type.  Output .npy is numeric [T,12]:
  0 energy
  1 onset
  2 beat
  3 tempo_proxy
  4 arousal
  5 delta_arousal
  6 tension
  7 calmness
  8 delta_tension
  9 next_beat_distance_norm
 10 phrase_boundary_score
 11 event_type_id
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.v20_motion_utils import json_safe, minmax, moving_average, write_json

EVENT_TO_ID = {
    "calm_flow": 0,
    "accent": 1,
    "climax": 2,
    "build_up": 3,
    "release": 4,
    "section_change": 5,
    "neutral": 6,
}
ID_TO_EVENT = {v: k for k, v in EVENT_TO_ID.items()}


def _fallback_audio_features(audio: str, num_frames: int, fps: float = 30.0) -> np.ndarray:
    # Deterministic fallback: create a smooth neutral feature stream if librosa/existing extractor fails.
    x = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    energy = 0.5 + 0.2 * np.sin(2 * np.pi * x)
    onset = np.zeros_like(x)
    beat = np.zeros_like(x)
    beat[::max(1, int(fps // 2))] = 1.0
    tempo = np.ones_like(x) * 0.5
    arousal = minmax(energy)
    tension = minmax(0.7 * energy + 0.3 * moving_average(beat, 5))
    calm = 1.0 - minmax(energy)
    return np.stack([energy, onset, beat, tempo, arousal, np.zeros_like(x), tension, calm], axis=-1).astype(np.float32)


def extract_base_features(audio: str, num_frames: int, fps: float) -> np.ndarray:
    try:
        from tools.extract_music_emotion_features import extract_music_emotion_features
        feat, _summary = extract_music_emotion_features(audio, num_frames=num_frames, fps=fps)
        feat = np.asarray(feat, dtype=np.float32)
    except Exception as exc:
        print(f"WARN: existing emotion extractor failed ({exc}); using deterministic fallback.")
        feat = _fallback_audio_features(audio, num_frames=num_frames, fps=fps)
    if feat.ndim == 1:
        feat = feat[:, None]
    if feat.shape[0] != num_frames:
        from tools.v20_motion_utils import resize_feature
        feat = resize_feature(feat, num_frames)
    if feat.shape[1] < 8:
        pad = np.zeros((num_frames, 8 - feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=-1)
    return feat.astype(np.float32)


def next_beat_distance(beat: np.ndarray, max_dist: int = 45) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32)
    T = len(beat)
    idx = np.where(beat > max(0.3, float(np.percentile(beat, 75))))[0]
    if len(idx) == 0:
        return np.ones((T,), dtype=np.float32)
    out = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        future = idx[idx >= t]
        d = int(future[0] - t) if len(future) else max_dist
        out[t] = min(float(d), float(max_dist)) / float(max_dist)
    return out


def classify_music_event(row: np.ndarray, prev: np.ndarray | None = None) -> str:
    energy, onset, beat, tempo, arousal, da, tension, calmness, dt, nbd, phrase, _ = row
    if phrase > 0.65 or abs(da) > 0.18 or abs(dt) > 0.18:
        return "section_change"
    if beat > 0.65 or onset > 0.65:
        return "accent"
    if tension > 0.68 or arousal > 0.70:
        return "climax"
    if da > 0.08 or dt > 0.08:
        return "build_up"
    if da < -0.08 or dt < -0.08:
        return "release"
    if calmness > 0.62 and tension < 0.48:
        return "calm_flow"
    return "neutral"


def build_music_event_stream(audio: str, num_frames: int, fps: float) -> tuple[np.ndarray, Dict]:
    base = extract_base_features(audio, num_frames=num_frames, fps=fps)
    energy = minmax(base[:, 0]) if base.shape[1] > 0 else np.zeros(num_frames, dtype=np.float32)
    onset = minmax(base[:, 1]) if base.shape[1] > 1 else np.zeros(num_frames, dtype=np.float32)
    beat = minmax(base[:, 2]) if base.shape[1] > 2 else np.zeros(num_frames, dtype=np.float32)
    tempo = minmax(base[:, 3]) if base.shape[1] > 3 else np.ones(num_frames, dtype=np.float32) * 0.5
    arousal = minmax(base[:, 4]) if base.shape[1] > 4 else energy
    tension = minmax(base[:, 6]) if base.shape[1] > 6 else minmax(0.6 * energy + 0.4 * onset)
    calmness = minmax(base[:, 7]) if base.shape[1] > 7 else 1.0 - energy

    arousal_s = moving_average(arousal, 7)
    tension_s = moving_average(tension, 7)
    da = np.zeros_like(arousal_s)
    dt = np.zeros_like(tension_s)
    da[1:] = arousal_s[1:] - arousal_s[:-1]
    dt[1:] = tension_s[1:] - tension_s[:-1]
    nbd = next_beat_distance(beat)
    phrase = minmax(np.abs(da) + np.abs(dt) + 0.5 * onset)

    out = np.stack([energy, onset, beat, tempo, arousal_s, da, tension_s, calmness, dt, nbd, phrase, np.zeros(num_frames)], axis=-1).astype(np.float32)
    events = []
    for t in range(num_frames):
        name = classify_music_event(out[t])
        out[t, 11] = EVENT_TO_ID[name]
        events.append(name)
    counts: Dict[str, int] = {}
    for e in events:
        counts[e] = counts.get(e, 0) + 1
    summary = {
        "audio": audio,
        "num_frames": int(num_frames),
        "fps": float(fps),
        "event_counts": counts,
        "feature_dim": int(out.shape[1]),
        "columns": [
            "energy", "onset", "beat", "tempo_proxy", "arousal", "delta_arousal",
            "tension", "calmness", "delta_tension", "next_beat_distance_norm",
            "phrase_boundary_score", "event_type_id",
        ],
        "id_to_event": ID_TO_EVENT,
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    stream, summary = build_music_event_stream(args.audio, args.num_frames, args.fps)
    Path(args.out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, stream.astype(np.float32))
    out_json = args.out_json or str(Path(args.out_npy).with_suffix(".json"))
    write_json(summary, out_json)
    print(f"saved_npy: {args.out_npy}")
    print(f"saved_json: {out_json}")
    print(f"event_counts: {summary['event_counts']}")


if __name__ == "__main__":
    main()
