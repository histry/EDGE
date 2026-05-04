"""PACE-ChoreoRAG trajectory pacing and elastic anchoring utilities.

This module is inference-only and disabled by default.  It is designed to be
called from generate_controlled.py inside build_control_trajectory():

    progress = build_pace_progress(audio_feature, num_frames, base_progress=progress)
    traj_physical = apply_pace_choreorag_to_trajectory(traj_physical, audio_feature)

Main switches:
    EDGE_TRAJ_BEAT_PACING=1       non-uniform progress from onset/energy
    EDGE_TRAJ_AUTO_SCALE=1        root-speed budget / trajectory scale cap
    EDGE_TRAJ_ELASTIC_ANCHOR=1    sparse-anchor reinterpolation

Recommended current config:
    export EDGE_TRAJ_BEAT_PACING=1
    export EDGE_TRAJ_AUTO_SCALE=1
    export EDGE_TRAJ_TARGET_ROOT_SPEED=0.012
    export EDGE_TRAJ_MAX_ROOT_SPEED=0.016
    export EDGE_TRAJ_MIN_SCALE=0.25
    export EDGE_TRAJ_MAX_SCALE=0.50
    export EDGE_TRAJ_ELASTIC_ANCHOR=1
    export EDGE_TRAJ_ANCHOR_STRIDE=15
    export EDGE_TRAJ_ANCHOR_BLEND=1.0
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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


def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - float(x.min())
    m = float(x.max())
    return np.zeros_like(x, dtype=np.float32) if m <= 1e-8 else (x / m).astype(np.float32)


def _smooth_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    window = max(1, int(window))
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)


def onset_curve(audio_feature: np.ndarray) -> np.ndarray:
    """Feature-layout tolerant onset proxy."""
    feat = np.asarray(audio_feature, dtype=np.float32)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    if feat.ndim != 2 or len(feat) == 0:
        return np.zeros((0,), dtype=np.float32)

    if feat.shape[1] > 768:
        x = np.maximum(feat[:, 768], 0.0)
    elif feat.shape[1] >= 35:
        # baseline_features: envelope + mfcc + chroma + peak/beat style tails
        x = np.maximum(feat[:, 0], 0.0) + 0.5 * np.maximum(feat[:, -2], 0.0) + 0.5 * np.maximum(feat[:, -1], 0.0)
    elif feat.shape[1] > 1 and len(feat) > 1:
        x = np.zeros((len(feat),), dtype=np.float32)
        x[1:] = np.linalg.norm(feat[1:] - feat[:-1], axis=-1)
        x[0] = x[1]
    else:
        x = np.maximum(feat[:, 0], 0.0)
    return _safe_norm01(_smooth_1d(x, _env_int("EDGE_TRAJ_ONSET_SMOOTH", 5)))


def energy_curve(audio_feature: np.ndarray) -> np.ndarray:
    feat = np.asarray(audio_feature, dtype=np.float32)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    if feat.ndim != 2 or len(feat) == 0:
        return np.zeros((0,), dtype=np.float32)
    return _safe_norm01(_smooth_1d(np.sqrt(np.mean(feat ** 2, axis=-1)), _env_int("EDGE_TRAJ_ENERGY_SMOOTH", 9)))


def _resample_1d(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if len(x) == n:
        return x.astype(np.float32)
    if len(x) == 0:
        return np.zeros((n,), dtype=np.float32)
    if len(x) == 1:
        return np.full((n,), float(x[0]), dtype=np.float32)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)


def build_pace_progress(audio_feature: np.ndarray, num_frames: int, base_progress: Optional[np.ndarray] = None) -> np.ndarray:
    """Return a monotonic [0,1] trajectory progress curve.

    If EDGE_TRAJ_BEAT_PACING is disabled, this returns base_progress or linear
    progress.  If enabled, movement speed follows onset/energy while respecting
    a minimum speed bias, so the path never stalls completely.
    """
    n = int(num_frames)
    if base_progress is None or len(base_progress) != n:
        base = np.linspace(0.0, 1.0, n, dtype=np.float32)
    else:
        base = np.asarray(base_progress, dtype=np.float32)

    if not _env_bool("EDGE_TRAJ_BEAT_PACING", False):
        return base.astype(np.float32)

    onset = _resample_1d(onset_curve(audio_feature), n)
    energy = _resample_1d(energy_curve(audio_feature), n)
    onset_w = _env_float("EDGE_TRAJ_PACING_ONSET_WEIGHT", 0.70)
    energy_w = _env_float("EDGE_TRAJ_PACING_ENERGY_WEIGHT", 0.30)
    min_bias = _env_float("EDGE_TRAJ_PACING_MIN_BIAS", 0.20)
    speed = min_bias + onset_w * onset + energy_w * energy
    speed = np.maximum(speed, 1e-6)
    prog = np.cumsum(speed)
    prog = prog - prog[0]
    prog = prog / max(float(prog[-1]), 1e-8)
    blend = float(np.clip(_env_float("EDGE_TRAJ_PACING_BLEND", 1.0), 0.0, 1.0))
    out = (1.0 - blend) * base + blend * prog.astype(np.float32)
    out = np.maximum.accumulate(out)
    out = (out - out[0]) / max(float(out[-1] - out[0]), 1e-8)
    return out.astype(np.float32)


def _trajectory_stats(traj: np.ndarray) -> Dict[str, float]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim == 3 and traj.shape[0] == 1:
        traj = traj[0]
    traj = traj[:, :2]
    if len(traj) < 2:
        return dict(root_speed_mean=0.0, root_speed_max=0.0, spatial_range=0.0, path_length=0.0)
    step = np.linalg.norm(traj[1:] - traj[:-1], axis=-1)
    return dict(
        root_speed_mean=float(step.mean()),
        root_speed_max=float(step.max()),
        spatial_range=float(np.linalg.norm(traj.max(axis=0) - traj.min(axis=0))),
        path_length=float(step.sum()),
    )


def _resample_traj_by_progress(traj: np.ndarray, progress: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim == 3 and traj.shape[0] == 1:
        traj = traj[0]
    xy = traj[:, :2]
    n = len(xy)
    if n <= 1:
        return xy.astype(np.float32)
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(step)]).astype(np.float32)
    if float(arc[-1]) <= 1e-8:
        return np.repeat(xy[:1], len(progress), axis=0).astype(np.float32)
    arc = arc / arc[-1]
    progress = np.asarray(progress, dtype=np.float32)
    progress = np.clip(progress, 0.0, 1.0)
    out_x = np.interp(progress, arc, xy[:, 0])
    out_z = np.interp(progress, arc, xy[:, 1])
    return np.stack([out_x, out_z], axis=-1).astype(np.float32)


def _apply_scale_cap(traj: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    traj = np.asarray(traj, dtype=np.float32).copy()
    stats0 = _trajectory_stats(traj)
    rel = traj - traj[:1]

    scale = _env_float("EDGE_TRAJ_SCALE", 1.0)
    if _env_bool("EDGE_TRAJ_AUTO_SCALE", False):
        target_mean = _env_float("EDGE_TRAJ_TARGET_ROOT_SPEED", 0.012)
        max_step = _env_float("EDGE_TRAJ_MAX_ROOT_SPEED", 0.016)
        min_scale = _env_float("EDGE_TRAJ_MIN_SCALE", 0.25)
        max_scale = _env_float("EDGE_TRAJ_MAX_SCALE", 0.55)
        scale_candidates = [max_scale]
        if stats0["root_speed_mean"] > 1e-8:
            scale_candidates.append(target_mean / stats0["root_speed_mean"])
        if stats0["root_speed_max"] > 1e-8:
            scale_candidates.append(max_step / stats0["root_speed_max"])
        scale = float(np.clip(min(scale_candidates), min_scale, max_scale))

    out = traj[:1] + rel * float(scale)
    stats1 = _trajectory_stats(out)
    meta = {f"before_{k}": v for k, v in stats0.items()}
    meta.update({f"after_scale_{k}": v for k, v in stats1.items()})
    meta["scale"] = float(scale)
    return out.astype(np.float32), meta


def _apply_elastic_anchors(traj: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    traj = np.asarray(traj, dtype=np.float32)
    if not _env_bool("EDGE_TRAJ_ELASTIC_ANCHOR", False) or len(traj) < 3:
        return traj.astype(np.float32), {"elastic_anchor_enabled": 0.0}

    stride = max(2, _env_int("EDGE_TRAJ_ANCHOR_STRIDE", 15))
    blend = float(np.clip(_env_float("EDGE_TRAJ_ANCHOR_BLEND", 1.0), 0.0, 1.0))
    n = len(traj)
    anchors = list(range(0, n, stride))
    if anchors[-1] != n - 1:
        anchors.append(n - 1)
    anchors = np.asarray(sorted(set(anchors)), dtype=np.int64)
    x = np.interp(np.arange(n), anchors, traj[anchors, 0])
    z = np.interp(np.arange(n), anchors, traj[anchors, 1])
    sparse = np.stack([x, z], axis=-1).astype(np.float32)
    out = (1.0 - blend) * traj + blend * sparse
    return out.astype(np.float32), {
        "elastic_anchor_enabled": 1.0,
        "anchor_stride": float(stride),
        "anchor_count": float(len(anchors)),
        "anchor_blend": float(blend),
    }


def apply_pace_choreorag_to_trajectory(traj_physical: np.ndarray, audio_feature: Optional[np.ndarray] = None, save_meta_path: str = "") -> np.ndarray:
    """Apply PACE-ChoreoRAG root-speed budgeting and elastic anchoring.

    This function preserves the initial point. It assumes traj_physical is a
    physical X/Z target trajectory after any keep_absolute normalization.
    """
    traj = np.asarray(traj_physical, dtype=np.float32)
    if traj.ndim == 3 and traj.shape[0] == 1:
        traj = traj[0]
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"traj_physical must be [T,2+], got {traj.shape}")
    traj = traj[:, :2].copy()

    meta: Dict[str, float] = {}
    meta.update({f"input_{k}": v for k, v in _trajectory_stats(traj).items()})

    if _env_bool("EDGE_TRAJ_RETIME_AFTER_SCALE", False) and audio_feature is not None:
        progress = build_pace_progress(audio_feature, len(traj), base_progress=np.linspace(0, 1, len(traj), dtype=np.float32))
        traj = _resample_traj_by_progress(traj, progress)
        meta.update({f"after_retime_{k}": v for k, v in _trajectory_stats(traj).items()})

    traj, scale_meta = _apply_scale_cap(traj)
    meta.update(scale_meta)
    traj, anchor_meta = _apply_elastic_anchors(traj)
    meta.update(anchor_meta)
    meta.update({f"output_{k}": v for k, v in _trajectory_stats(traj).items()})

    if _env_bool("EDGE_TRAJ_VERBOSE", True) and (
        _env_bool("EDGE_TRAJ_AUTO_SCALE", False) or _env_bool("EDGE_TRAJ_ELASTIC_ANCHOR", False)
    ):
        print(
            "✅ PACE trajectory: "
            f"scale={meta.get('scale', 1.0):.3f}, "
            f"root_speed={meta.get('input_root_speed_mean', 0.0):.5f}->{meta.get('output_root_speed_mean', 0.0):.5f}, "
            f"range={meta.get('input_spatial_range', 0.0):.3f}->{meta.get('output_spatial_range', 0.0):.3f}, "
            f"anchors={int(meta.get('anchor_count', 0.0)) if meta.get('elastic_anchor_enabled', 0.0) else 0}"
        )

    if save_meta_path:
        try:
            Path(save_meta_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"⚠️ failed to save PACE trajectory meta: {exc}")
    return traj.astype(np.float32)


def summarize_trajectory_stats(traj_physical: np.ndarray) -> Dict[str, float]:
    return _trajectory_stats(np.asarray(traj_physical, dtype=np.float32))
