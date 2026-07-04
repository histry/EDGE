#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.12 classical-music semantic sidecar proxy.

This is a schema adapter / fallback, not a claim of a trained classifier.  If a
trained classical-music model exists, replace this command with the real model
but keep the same JSON schema:

{
  "audio": "...wav",
  "labels": ["calm_meditative", ...],
  "slots": [
    {"slot_id": 0, "start_sec": 0.0, "end_sec": 4.0,
     "top_label": "lyrical_flow", "probs": {"lyrical_flow": 0.6, ...}}
  ]
}
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

LABELS = [
    "calm_meditative", "lyrical_flow", "pose_hold", "instrument_phrase",
    "percussive_accent", "turning_climax", "footwork_flow",
]

NAME_HINTS = {
    "pipa": "instrument_phrase",
    "guzhen": "lyrical_flow",
    "guzheng": "lyrical_flow",
    "xiao": "calm_meditative",
    "gu": "percussive_accent",
    "drum": "percussive_accent",
    "luo": "percussive_accent",
    "gong": "percussive_accent",
}

def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.max(x))
    y = np.exp(x)
    return y / max(float(y.sum()), 1e-8)

def duration_sec(path: Path) -> float:
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return float(len(y) / sr) if sr else 0.0
    except Exception:
        try:
            import wave
            with wave.open(str(path), "rb") as wf:
                return float(wf.getnframes() / wf.getframerate())
        except Exception:
            return 4.0

def extract_features(path: Path, n_slots: int) -> np.ndarray:
    try:
        from tools.extract_v21_music_features import extract_audio_features
        feat, _ = extract_audio_features(path, num_frames=n_slots)
        return np.asarray(feat, dtype=np.float32)
    except Exception:
        return np.zeros((n_slots, 12), dtype=np.float32)

def stem_hint(path: Path) -> str | None:
    stem = path.stem.lower()
    for key, label in NAME_HINTS.items():
        if key in stem:
            return label
    return None

def logits_from_feature(row, hint=None):
    energy = float(row[0]) if len(row) > 0 else 0.4
    onset = float(row[1]) if len(row) > 1 else 0.2
    beat = float(row[2]) if len(row) > 2 else 0.0
    arousal = float(row[4]) if len(row) > 4 else energy
    tension = float(row[6]) if len(row) > 6 else onset
    calm = float(row[7]) if len(row) > 7 else 1.0 - energy
    novelty = float(row[8]) if len(row) > 8 else 0.0
    accent = float(row[11]) if len(row) > 11 else onset
    scores = {
        "calm_meditative": 1.8 * calm - 0.9 * onset - 0.6 * arousal,
        "pose_hold": 0.9 * calm + 0.4 * (1.0 - novelty) - 0.4 * onset,
        "lyrical_flow": 0.6 * energy + 0.4 * tension + 0.5 * novelty - 0.3 * accent,
        "instrument_phrase": 0.5 * tension + 0.6 * novelty + 0.3 * onset,
        "percussive_accent": 1.2 * accent + 0.9 * onset + 0.3 * beat,
        "turning_climax": 1.0 * arousal + 0.8 * tension + 0.4 * novelty,
        "footwork_flow": 0.6 * beat + 0.5 * energy + 0.3 * novelty,
    }
    if hint in scores:
        scores[hint] += 1.4
    return np.asarray([scores[k] for k in LABELS], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--slot_seconds", type=float, default=4.0)
    args = ap.parse_args()
    audio = Path(args.audio)
    total = max(0.1, duration_sec(audio))
    n = max(1, int(math.ceil(total / max(args.slot_seconds, 1e-3))))
    feat = extract_features(audio, n)
    hint = stem_hint(audio)
    slots = []
    for i in range(n):
        probs = softmax(logits_from_feature(feat[i], hint))
        d = {LABELS[j]: float(probs[j]) for j in range(len(LABELS))}
        top = LABELS[int(np.argmax(probs))]
        slots.append({
            "slot_id": i,
            "start_sec": float(i * args.slot_seconds),
            "end_sec": float(min(total, (i + 1) * args.slot_seconds)),
            "top_label": top,
            "probs": d,
        })
    out = {"version": "v46_12_classical_music_semantic_proxy", "audio": str(audio), "labels": LABELS, "slots": slots}
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(outp)

if __name__ == "__main__":
    main()
