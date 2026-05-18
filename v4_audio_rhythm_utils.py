from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _read_wav_basic(path: str | Path) -> Tuple[np.ndarray, int]:
    path = Path(path)
    try:
        from scipy.io import wavfile
        sr, y = wavfile.read(str(path))
        y = np.asarray(y)
    except Exception:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if width == 1:
            dtype = np.uint8
            y = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            y = (y - 128.0) / 128.0
        elif width == 2:
            dtype = np.int16
            y = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            y = y / 32768.0
        elif width == 4:
            dtype = np.int32
            y = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            y = y / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {width}")

        if channels > 1:
            y = y.reshape(-1, channels).mean(axis=1)

        return y.astype(np.float32), int(sr)

    if y.ndim > 1:
        y = y.mean(axis=1)

    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        y = y.astype(np.float32) / max(abs(info.min), abs(info.max))
    else:
        y = y.astype(np.float32)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.max(np.abs(y)) + 1e-8
    if peak > 1.5:
        y = y / peak

    return y.astype(np.float32), int(sr)


def _resample_1d(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    if len(x) == length:
        return x.astype(np.float32)
    if len(x) <= 1:
        return np.full((length,), float(x[0]) if len(x) else 0.0, dtype=np.float32)

    old = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.interp(new, old, x).astype(np.float32)


def _smooth(x: np.ndarray, radius: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    radius = int(max(0, radius))
    if radius <= 0 or len(x) <= 2:
        return x
    k = np.arange(-radius, radius + 1, dtype=np.float32)
    sigma = max(1.0, radius / 1.5)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w = w / np.sum(w)
    pad = np.pad(x, (radius, radius), mode="edge")
    return np.convolve(pad, w, mode="valid").astype(np.float32)


def _norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.min(x)
    mx = np.max(x)
    if mx < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / mx).astype(np.float32)


def audio_rhythm_curve(
    audio_path: str | Path,
    target_len: int,
    fps: float = 30.0,
    rms_weight: float = 0.55,
    flux_weight: float = 0.45,
    smooth_radius: int = 2,
) -> Dict[str, np.ndarray | float | int | str]:
    """
    Build a simple dependency-light rhythm curve from WAV.

    Outputs:
      rhythm_curve: [target_len], normalized 0..1
      speed_curve:  [target_len], normalized 0..1, used for temporal phase speed
    """
    y, sr = _read_wav_basic(audio_path)

    if len(y) < sr * 0.1:
        y = np.pad(y, (0, int(sr * 0.1) - len(y)), mode="constant")

    frame = max(128, int(round(sr * 0.050)))  # 50 ms
    hop = max(64, int(round(sr / max(1e-6, fps) / 2.0)))

    n_frames = max(1, 1 + max(0, len(y) - frame) // hop)
    rms = []
    flux = []

    prev_mag = None
    win = np.hanning(frame).astype(np.float32)

    for i in range(n_frames):
        start = i * hop
        seg = y[start:start + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)), mode="constant")
        seg = seg.astype(np.float32)

        rms.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))

        mag = np.abs(np.fft.rfft(seg * win))
        mag = mag / (np.linalg.norm(mag) + 1e-8)
        if prev_mag is None:
            flux.append(0.0)
        else:
            flux.append(float(np.maximum(mag - prev_mag, 0.0).sum()))
        prev_mag = mag

    rms = _norm01(np.asarray(rms, dtype=np.float32))
    flux = _norm01(np.asarray(flux, dtype=np.float32))

    env = float(rms_weight) * rms + float(flux_weight) * flux
    env = _norm01(env)
    env = _smooth(env, radius=smooth_radius)
    env = _norm01(env)

    rhythm = _resample_1d(env, target_len)
    rhythm = _smooth(rhythm, radius=smooth_radius)
    rhythm = _norm01(rhythm)

    # speed curve is more conservative than raw rhythm.
    speed_curve = 0.25 + 0.75 * rhythm
    speed_curve = speed_curve / np.mean(speed_curve).clip(1e-6)

    return {
        "audio_path": str(audio_path),
        "sample_rate": int(sr),
        "audio_num_samples": int(len(y)),
        "target_len": int(target_len),
        "rhythm_curve": rhythm.astype(np.float32),
        "speed_curve": speed_curve.astype(np.float32),
    }


def warp_motion_by_speed_curve(
    motion: np.ndarray,
    speed_curve: np.ndarray,
    warp_strength: float = 1.0,
    min_speed: float = 0.55,
    max_speed: float = 1.85,
    preserve_ends: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nonlinear time-warp a [T,C] motion according to speed_curve [T_out].

    High speed_curve -> phase advances faster -> faster motion progression.
    Low speed_curve  -> phase advances slower  -> slower motion progression.
    """
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2:
        raise ValueError(f"motion must be [T,C], got {motion.shape}")

    T_src, C = motion.shape
    T_out = len(speed_curve)

    speed = np.asarray(speed_curve, dtype=np.float32).reshape(-1)
    speed = _norm01(speed)

    # Blend with uniform speed. warp_strength=0 -> linear playback.
    adaptive = float(min_speed) + (float(max_speed) - float(min_speed)) * speed
    adaptive = adaptive / np.mean(adaptive).clip(1e-6)
    final_speed = (1.0 - float(warp_strength)) * np.ones_like(adaptive) + float(warp_strength) * adaptive
    final_speed = np.maximum(final_speed, 1e-4)

    phase = np.cumsum(final_speed)
    phase = (phase - phase[0]) / max(phase[-1] - phase[0], 1e-8)

    if preserve_ends:
        phase[0] = 0.0
        phase[-1] = 1.0

    src_x = np.linspace(0.0, 1.0, T_src, dtype=np.float32)
    warped = np.empty((T_out, C), dtype=np.float32)
    for c in range(C):
        warped[:, c] = np.interp(phase, src_x, motion[:, c]).astype(np.float32)

    return warped, phase.astype(np.float32)


def save_rhythm_diagnostics(path: str | Path, payload: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.astype(float).tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        else:
            serializable[k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
