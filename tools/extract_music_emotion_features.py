#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _patch_scipy_hann() -> None:
    """
    修复 librosa 旧版本 + scipy 新版本不兼容问题：
    librosa 旧代码调用 scipy.signal.hann；
    新 scipy 中 hann 在 scipy.signal.windows.hann。
    """
    try:
        import scipy.signal
        if not hasattr(scipy.signal, "hann"):
            scipy.signal.hann = scipy.signal.windows.hann
    except Exception:
        pass


def _minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def _safe_interp(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    if len(x) == 0:
        return np.zeros((length,), dtype=np.float32)
    if len(x) == length:
        return x.astype(np.float32)
    src = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def _moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if k <= 1 or len(x) <= 2:
        return x
    k = min(int(k), len(x))
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xp, kernel, mode="valid")
    return y[: len(x)].astype(np.float32)


def _safe_beat_track(librosa, onset_env: np.ndarray, sr: int, hop_length: int) -> Tuple[float, np.ndarray]:
    """
    兼容不同 librosa/scipy 版本。
    优先使用 beat_track(trim=False)；
    如果仍失败，则退化为 onset 峰值 pseudo-beat。
    """
    _patch_scipy_hann()

    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            trim=False,
        )
        if isinstance(tempo, (list, tuple, np.ndarray)):
            tempo = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
        else:
            tempo = float(tempo)
        return tempo, np.asarray(beat_frames, dtype=np.int64)
    except Exception as e:
        print("[WARN] librosa beat_track failed, fallback to onset peaks:", repr(e))

        onset = np.asarray(onset_env, dtype=np.float32)
        if len(onset) < 3:
            return 0.0, np.zeros((0,), dtype=np.int64)

        threshold = float(onset.mean() + 0.5 * onset.std())
        peaks = []
        last = -999
        min_gap = max(3, len(onset) // 32)

        for i in range(1, len(onset) - 1):
            if onset[i] > threshold and onset[i] >= onset[i - 1] and onset[i] >= onset[i + 1]:
                if i - last >= min_gap:
                    peaks.append(i)
                    last = i

        if len(peaks) >= 2:
            dt = np.diff(np.asarray(peaks, dtype=np.float32)) * hop_length / float(sr)
            dt = dt[dt > 1e-6]
            tempo = float(60.0 / np.median(dt)) if len(dt) else 0.0
        else:
            tempo = 0.0

        return tempo, np.asarray(peaks, dtype=np.int64)


def extract_music_emotion_features(
    audio: str,
    num_frames: int = 150,
    fps: float = 30.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> Tuple[np.ndarray, Dict]:
    _patch_scipy_hann()

    try:
        import librosa
    except Exception as exc:
        raise RuntimeError("需要安装 librosa：pip install librosa") from exc

    y, sr = librosa.load(audio, sr=sr, mono=True)

    if len(y) == 0:
        feats = np.zeros((num_frames, 8), dtype=np.float32)
        return feats, {"audio": audio, "warning": "empty audio"}

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    tempo, beat_frames = _safe_beat_track(librosa, onset_env, sr=sr, hop_length=hop_length)

    energy = _moving_average(_minmax(_safe_interp(rms, num_frames)), 3)
    onset = _moving_average(_minmax(_safe_interp(onset_env, num_frames)), 3)
    brightness = _minmax(_safe_interp(centroid, num_frames))
    bw = _minmax(_safe_interp(bandwidth, num_frames))
    zcr_f = _minmax(_safe_interp(zcr, num_frames))

    beat = np.zeros((num_frames,), dtype=np.float32)
    if len(beat_frames):
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        duration = len(y) / float(sr)
        for t in beat_times:
            idx = int(round((t / max(duration, 1e-8)) * (num_frames - 1)))
            if 0 <= idx < num_frames:
                beat[idx] = 1.0
        beat = _moving_average(beat, 3)

    tempo_norm = np.clip((float(tempo) - 40.0) / 160.0, 0.0, 1.0)
    tempo_curve = np.full((num_frames,), tempo_norm, dtype=np.float32)

    # 弱情感语义代理，不作为强情感识别 claim。
    arousal = _minmax(0.55 * energy + 0.35 * onset + 0.10 * tempo_curve)
    valence = _minmax(0.65 * brightness + 0.20 * bw - 0.15 * zcr_f)
    tension = _minmax(0.45 * onset + 0.25 * energy + 0.20 * bw + 0.10 * tempo_curve)
    calmness = _minmax(1.0 - 0.60 * arousal - 0.25 * onset + 0.15 * (1.0 - bw))

    feats = np.stack(
        [
            energy,
            onset,
            beat,
            tempo_curve,
            arousal,
            valence,
            tension,
            calmness,
        ],
        axis=-1,
    ).astype(np.float32)

    summary = {
        "audio": str(audio),
        "num_frames": int(num_frames),
        "fps": float(fps),
        "sr": int(sr),
        "hop_length": int(hop_length),
        "tempo": float(tempo),
        "feature_dim": 8,
        "feature_names": [
            "energy",
            "onset",
            "beat",
            "tempo_norm",
            "arousal_proxy",
            "valence_brightness_proxy",
            "tension_proxy",
            "calmness_proxy",
        ],
        "global": {
            "energy": float(np.mean(energy)),
            "onset": float(np.mean(onset)),
            "arousal": float(np.mean(arousal)),
            "valence": float(np.mean(valence)),
            "tension": float(np.mean(tension)),
            "calmness": float(np.mean(calmness)),
            "beat_count": int(np.sum(beat > 0.25)),
        },
        "compat": {
            "scipy_signal_hann_patch": True,
            "beat_track_trim": False,
        },
    }

    return feats, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop_length", type=int, default=512)
    args = ap.parse_args()

    feats, summary = extract_music_emotion_features(
        args.audio,
        num_frames=args.num_frames,
        fps=args.fps,
        sr=args.sr,
        hop_length=args.hop_length,
    )

    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, feats)

    out_json = Path(args.out_json) if args.out_json else out_npy.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_npy: {out_npy}")
    print(f"saved_json: {out_json}")


if __name__ == "__main__":
    main()
