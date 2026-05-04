"""Energy-speed coherence helpers for ChoreoRAG / TEA-MotionAdapter."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    x = x - float(x.min())
    m = float(x.max())
    if m <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x / m).astype(np.float32)


def smooth1d(x: np.ndarray, window: int = 7) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    window = int(max(1, window))
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)


def onset_from_audio(audio_feature: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_feature, dtype=np.float32)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 2 or len(audio) == 0:
        return np.zeros((0,), dtype=np.float32)

    T, C = audio.shape
    if C > 768:
        onset = np.maximum(audio[:, 768], 0.0)
    elif C >= 35:
        onset = np.maximum(audio[:, 0] + 0.5 * audio[:, -2] + 0.5 * audio[:, -1], 0.0)
    elif T > 1:
        onset = np.zeros((T,), dtype=np.float32)
        onset[1:] = np.linalg.norm(audio[1:] - audio[:-1], axis=-1)
        onset[0] = onset[1]
    else:
        onset = np.zeros((T,), dtype=np.float32)
    return normalize01(onset)


def trajectory_speed_and_curvature(traj_xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    traj = np.asarray(traj_xz, dtype=np.float32)
    if traj.ndim == 3 and traj.shape[0] == 1:
        traj = traj[0]
    traj = traj[:, :2]
    T = len(traj)
    speed = np.zeros((T,), dtype=np.float32)
    if T > 1:
        speed[1:] = np.linalg.norm(traj[1:] - traj[:-1], axis=-1)
        speed[0] = speed[1]

    curv = np.zeros((T,), dtype=np.float32)
    if T > 2:
        v1 = traj[1:-1] - traj[:-2]
        v2 = traj[2:] - traj[1:-1]
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        cos = np.sum(v1 * v2, axis=-1) / np.clip(n1 * n2, 1e-8, None)
        curv[1:-1] = 1.0 - np.clip(cos, -1.0, 1.0)

    return normalize01(speed), normalize01(curv)


def build_energy_curve_from_audio_traj(audio_feature: np.ndarray, traj_xz: np.ndarray, base_level: float | None = None) -> np.ndarray:
    audio = np.asarray(audio_feature, dtype=np.float32)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    T = audio.shape[0] if audio.ndim == 2 else len(np.asarray(traj_xz))

    base = _env_float("EDGE_ENERGY_LEVEL", 0.55) if base_level is None else float(base_level)
    e_min = _env_float("EDGE_ENERGY_MIN", 0.20)
    e_max = _env_float("EDGE_ENERGY_MAX", 0.85)

    w_speed = _env_float("EDGE_ENERGY_TRAJ_SPEED_WEIGHT", 0.55)
    w_audio = _env_float("EDGE_ENERGY_AUDIO_WEIGHT", 0.30)
    w_curv = _env_float("EDGE_ENERGY_CURVATURE_WEIGHT", 0.15)
    smooth = _env_int("EDGE_ENERGY_SMOOTH", 7)

    onset = onset_from_audio(audio)
    speed, curv = trajectory_speed_and_curvature(traj_xz)

    def resize(x):
        x = np.asarray(x, dtype=np.float32)
        if len(x) == T:
            return x
        if len(x) == 0:
            return np.zeros((T,), dtype=np.float32)
        return np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(x)), x).astype(np.float32)

    onset = resize(onset)
    speed = resize(speed)
    curv = resize(curv)

    total_w = max(w_speed + w_audio + w_curv, 1e-8)
    dynamic = (w_speed * speed + w_audio * onset + w_curv * curv) / total_w
    dynamic = smooth1d(normalize01(dynamic), smooth)

    curve = base * 0.35 + dynamic * 0.65
    curve = np.clip(curve, e_min, e_max).astype(np.float32)
    return curve.reshape(T, 1)


def energy_curve_summary(curve: np.ndarray) -> Dict[str, float]:
    c = np.asarray(curve, dtype=np.float32).reshape(-1)
    if c.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {"min": float(c.min()), "max": float(c.max()), "mean": float(c.mean()), "std": float(c.std())}
