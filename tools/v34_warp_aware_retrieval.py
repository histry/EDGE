#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V34 warp-aware Event-RAG beam search.

Unlike V26, duration feasibility is evaluated after the candidate-specific
transition length is known.  In strict mode, an event is never admitted to the
beam when its exact locked-slot warp lies outside the allowed interval.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

import tools.schedule_v26_whole_song as scheduler
from tools.v34_source_aware_rag import identity_lists

try:
    from tools.v34_gpu_candidate_cache import build_v34_gpu_candidate_cache
except Exception:  # pragma: no cover - keeps old CPU path usable everywhere.
    build_v34_gpu_candidate_cache = None

try:
    from tools.v34_boundary_compatibility import evaluate_boundary_compatibility
except Exception:  # pragma: no cover - keeps baseline retrieval importable.
    evaluate_boundary_compatibility = None


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _array_or(
    source: Any,
    key: str,
    fallback: np.ndarray,
    dtype=np.float32,
) -> np.ndarray:
    if source is None:
        return np.asarray(fallback, dtype=dtype)
    try:
        if key in source:
            return np.asarray(source[key], dtype=dtype)
    except Exception:
        pass
    return np.asarray(fallback, dtype=dtype)


def _excess_ratio(value: float, limit: float) -> float:
    return float(max(0.0, float(value) / max(float(limit), 1e-8) - 1.0))


def _csv_set(name: str, default: str) -> set:
    raw = os.getenv(name, default)
    return {part.strip() for part in raw.split(",") if part.strip()}


def _unique_indices(*groups: Sequence[int]) -> np.ndarray:
    seen = set()
    out: List[int] = []
    for group in groups:
        for value in group:
            idx = int(value)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return np.asarray(out, dtype=np.int64)


_FK_PARENTS = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
], dtype=np.int64)

_FK_OFFSETS = np.array([
    [0.00, 0.00, 0.00], [-0.10, -0.10, 0.00],
    [0.10, -0.10, 0.00], [0.00, 0.13, 0.00],
    [0.00, -0.42, 0.00], [0.00, -0.42, 0.00],
    [0.00, 0.14, 0.00], [0.00, -0.40, 0.00],
    [0.00, -0.40, 0.00], [0.00, 0.14, 0.00],
    [0.00, -0.08, 0.12], [0.00, -0.08, 0.12],
    [0.00, 0.14, 0.00], [-0.10, 0.08, 0.00],
    [0.10, 0.08, 0.00], [0.00, 0.16, 0.00],
    [-0.18, 0.00, 0.00], [0.18, 0.00, 0.00],
    [-0.28, 0.00, 0.00], [0.28, 0.00, 0.00],
    [-0.25, 0.00, 0.00], [0.25, 0.00, 0.00],
    [-0.08, 0.00, 0.00], [0.08, 0.00, 0.00],
], dtype=np.float32)

_COLLISION_PAIRS = (
    (18, 3), (19, 3), (20, 3), (21, 3), (22, 3), (23, 3),
    (18, 6), (19, 6), (20, 6), (21, 6), (22, 6), (23, 6),
    (18, 9), (19, 9), (20, 9), (21, 9), (22, 9), (23, 9),
    (20, 12), (21, 12), (22, 12), (23, 12),
    (20, 21), (22, 23),
)


def _rot6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _fk_collision_risk(motion: np.ndarray) -> Dict[str, float]:
    try:
        x = np.asarray(motion, dtype=np.float32)
        if x.ndim == 3:
            x = x[0]
        if x.ndim != 2 or x.shape[1] < 151 or len(x) < 2:
            return {"self_collision_risk": 0.0, "self_collision_min_distance": 1.0}
        t = x.shape[0]
        root = x[:, [4, 5, 6]]
        local_r = _rot6d_to_matrix_np(x[:, 7:151].reshape(t, 24, 6))
        joints = np.zeros((t, 24, 3), dtype=np.float32)
        global_r = np.zeros((t, 24, 3, 3), dtype=np.float32)
        joints[:, 0] = root
        global_r[:, 0] = local_r[:, 0]
        for j in range(1, 24):
            p = int(_FK_PARENTS[j])
            global_r[:, j] = np.matmul(global_r[:, p], local_r[:, j])
            off = _FK_OFFSETS[j][None, :, None]
            joints[:, j] = joints[:, p] + np.matmul(global_r[:, p], off)[..., 0]
        radius = _env_float("V34_SELF_COLLISION_RADIUS", 0.16)
        dists = []
        for a, b in _COLLISION_PAIRS:
            dists.append(np.linalg.norm(joints[:, a] - joints[:, b], axis=-1))
        dist = np.stack(dists, axis=1)
        penetration = np.maximum(0.0, radius - dist) / max(radius, 1e-8)
        return {
            "self_collision_risk": float(np.mean(np.square(penetration))),
            "self_collision_min_distance": float(np.min(dist)),
        }
    except Exception:
        return {"self_collision_risk": 0.0, "self_collision_min_distance": 1.0}


def _motion_rhythm_features(
    motion: Any,
    *,
    fps: int = 30,
) -> Dict[str, float]:
    """Compact rhythm-density descriptors for one snippet.

    These features are intentionally computed from the already-loaded motion
    tensors, so rhythm repair is a pure retrieval-time policy and does not
    require rebuilding the event database.
    """
    try:
        x = np.asarray(motion, dtype=np.float32)
    except Exception:
        x = np.zeros((0, 151), dtype=np.float32)
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2 or len(x) < 3:
        return {
            "mean_energy": 0.0,
            "p90_energy": 0.0,
            "max_energy": 0.0,
            "first1s_energy_ratio": 0.0,
            "tail_mean_energy": 0.0,
            "late_energy_ratio": 0.0,
            "tail_to_mean_energy": 0.0,
            "low_energy_fraction": 1.0,
            "self_collision_risk": 0.0,
            "self_collision_min_distance": 1.0,
        }

    rot = x[:, 7:151] if x.shape[1] >= 151 else x
    if x.shape[1] > 6:
        root = x[:, [4, 5, 6]] * 3.0
        feat = np.concatenate([rot, root], axis=1)
    else:
        feat = rot
    energy = np.linalg.norm(np.diff(feat, axis=0), axis=1)
    if len(energy) == 0:
        return {
            "mean_energy": 0.0,
            "p90_energy": 0.0,
            "max_energy": 0.0,
            "first1s_energy_ratio": 0.0,
            "tail_mean_energy": 0.0,
            "late_energy_ratio": 0.0,
            "tail_to_mean_energy": 0.0,
            "low_energy_fraction": 1.0,
            "self_collision_risk": 0.0,
            "self_collision_min_distance": 1.0,
        }

    first = energy[: min(int(fps), len(energy))]
    tail = energy[len(energy) // 2:]
    total = float(np.sum(energy) + 1e-8)
    mean_energy = float(np.mean(energy))
    late_start = min(len(energy), max(0, int(round(0.40 * len(energy)))))
    late = energy[late_start:]
    low_threshold = _env_float("V34_LOW_ENERGY_FRAME_THRESHOLD", 0.010)
    collision = _fk_collision_risk(x) if _enabled("V34_SELF_COLLISION_RETRIEVAL", "1") else {
        "self_collision_risk": 0.0,
        "self_collision_min_distance": 1.0,
    }
    return {
        "mean_energy": mean_energy,
        "p90_energy": float(np.percentile(energy, 90)),
        "max_energy": float(np.max(energy)),
        "first1s_energy_ratio": float(np.sum(first) / total),
        "tail_mean_energy": float(np.mean(tail)) if len(tail) else 0.0,
        "late_energy_ratio": float(np.sum(late) / total) if len(late) else 0.0,
        "tail_to_mean_energy": float(
            (np.mean(tail) if len(tail) else 0.0) / max(mean_energy, 1e-8)
        ),
        "low_energy_fraction": float(np.mean(energy < low_threshold)),
        **collision,
    }



def _native_floor_penalty_arrays(
    items: Sequence[Mapping[str, Any]],
    motions: Sequence[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Penalise native floor-pathological RAG snippets before beam expansion.

    V40 uses this to stop the retrieval layer from repeatedly selecting snippets
    whose own FK chain already contains deep foot-floor penetration.  If the
    JSON item has fields produced by tools/v40_native_floor_audit.py they are
    used directly; otherwise the value is computed on the fly from the loaded
    motion.  The penalty is deliberately soft, so a snippet can still survive
    when it is musically essential, but it must pay an explicit physical prior.
    """
    n = len(motions)
    zeros = np.zeros((n,), dtype=np.float32)
    if not _enabled("V40_NATIVE_FLOOR_PENALTY", "1") or n == 0:
        return zeros, {"enabled": False}
    tolerance = _env_float("V40_NATIVE_FLOOR_TOLERANCE_M", 0.04)
    weight = _env_float("V40_NATIVE_FLOOR_PENALTY_WEIGHT", 8.0)
    q = float(np.clip(_env_float("V40_NATIVE_FLOOR_QUANTILE", 0.05), 0.005, 0.45))
    margin = _env_float("V40_NATIVE_FLOOR_MARGIN", 0.006)
    native_pen = np.zeros((n,), dtype=np.float32)
    min_foot_y = np.zeros((n,), dtype=np.float32)
    floor_y = np.zeros((n,), dtype=np.float32)
    source = []
    for i in range(n):
        item = items[i] if i < len(items) else {}
        if isinstance(item, Mapping) and "native_floor_penetration_m" in item:
            pen = float(item.get("native_floor_penetration_m", 0.0) or 0.0)
            native_pen[i] = max(0.0, pen)
            min_foot_y[i] = float(item.get("native_min_foot_y", 0.0) or 0.0)
            floor_y[i] = float(item.get("native_floor_y", 0.0) or 0.0)
            source.append("json")
            continue
        try:
            x = np.asarray(motions[i], dtype=np.float32)
            if x.ndim == 3:
                x = x[0]
            if x.ndim != 2 or x.shape[1] < 151 or len(x) < 2:
                source.append("missing")
                continue
            t = x.shape[0]
            root = x[:, [4, 5, 6]]
            local_r = _rot6d_to_matrix_np(x[:, 7:151].reshape(t, 24, 6))
            joints = np.zeros((t, 24, 3), dtype=np.float32)
            global_r = np.zeros((t, 24, 3, 3), dtype=np.float32)
            joints[:, 0] = root
            global_r[:, 0] = local_r[:, 0]
            for j in range(1, 24):
                p = int(_FK_PARENTS[j])
                global_r[:, j] = np.matmul(global_r[:, p], local_r[:, j])
                joints[:, j] = joints[:, p] + np.matmul(global_r[:, p], _FK_OFFSETS[j][None, :, None])[..., 0]
            foot_y = joints[:, [7, 8, 10, 11], 1]
            fy = float(np.quantile(foot_y.reshape(-1), q))
            mf = float(np.min(foot_y))
            floor_y[i] = fy
            min_foot_y[i] = mf
            native_pen[i] = max(0.0, fy + margin - mf)
            source.append("computed")
        except Exception:
            source.append("failed")
            native_pen[i] = 0.0
    excess = np.maximum(0.0, native_pen - float(tolerance))
    penalty = weight * np.square(excess).astype(np.float32)
    return penalty.astype(np.float32), {
        "enabled": True,
        "weight": float(weight),
        "tolerance_m": float(tolerance),
        "quantile": float(q),
        "margin": float(margin),
        "max_native_penetration_m": float(np.max(native_pen)) if len(native_pen) else 0.0,
        "mean_native_penetration_m": float(np.mean(native_pen)) if len(native_pen) else 0.0,
        "num_over_tolerance": int(np.sum(native_pen > float(tolerance))),
        "source_counts": {str(k): int(source.count(k)) for k in sorted(set(source))},
    }

def _build_rhythm_feature_arrays(
    motions: Sequence[np.ndarray],
    *,
    fps: int = 30,
) -> Dict[str, np.ndarray]:
    rows = [_motion_rhythm_features(motion, fps=fps) for motion in motions]
    if not rows:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "mean_energy": empty,
            "p90_energy": empty,
            "max_energy": empty,
            "first1s_energy_ratio": empty,
            "tail_mean_energy": empty,
            "late_energy_ratio": empty,
            "tail_to_mean_energy": empty,
            "low_energy_fraction": empty,
            "self_collision_risk": empty,
            "self_collision_min_distance": empty,
        }
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        for key in rows[0].keys()
    }


def _music_motion_pressure(phrase: Any) -> float:
    return float(np.clip(
        0.45 * float(getattr(phrase, "energy", 0.0))
        + 0.25 * float(getattr(phrase, "beat_density", 0.0))
        + 0.20 * float(getattr(phrase, "onset", 0.0))
        + 0.10 * float(getattr(phrase, "tension", 0.0))
        - 0.25 * float(getattr(phrase, "calmness", 0.0)),
        0.0,
        1.0,
    ))


def _is_musical_silence(phrase: Any) -> bool:
    event = str(getattr(phrase, "music_event", "neutral_flow"))
    silence_events = _csv_set(
        "V34_RHYTHM_SILENCE_EVENTS",
        "silence,rest,ending,cadence,settle,section_end",
    )
    if event in silence_events:
        return True
    return bool(
        float(getattr(phrase, "energy", 0.0))
        <= _env_float("V34_RHYTHM_SILENCE_ENERGY_MAX", 0.12)
        and float(getattr(phrase, "beat_density", 0.0))
        <= _env_float("V34_RHYTHM_SILENCE_BEAT_MAX", 0.10)
        and float(getattr(phrase, "calmness", 0.0))
        >= _env_float("V34_RHYTHM_SILENCE_CALM_MIN", 0.72)
    )


def _rhythm_prior_penalty_arrays(
    *,
    rhythm_features: Dict[str, np.ndarray],
    phrase: Any,
    slot_duration_sec: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """Candidate-local motion-density penalties used before top-k pruning."""
    mean_e = rhythm_features["mean_energy"]
    first_ratio = rhythm_features["first1s_energy_ratio"]
    tail_e = rhythm_features["tail_mean_energy"]
    late_ratio = rhythm_features.get(
        "late_energy_ratio",
        np.zeros_like(mean_e, dtype=np.float32),
    )
    tail_to_mean = rhythm_features.get(
        "tail_to_mean_energy",
        np.zeros_like(mean_e, dtype=np.float32),
    )
    low_energy_fraction = rhythm_features.get(
        "low_energy_fraction",
        np.ones_like(mean_e, dtype=np.float32),
    )
    self_collision_risk = rhythm_features.get(
        "self_collision_risk",
        np.zeros_like(mean_e, dtype=np.float32),
    )
    zeros = np.zeros_like(mean_e, dtype=np.float32)
    if not _enabled("V34_RHYTHM_DEGRADATION_PENALTY", "0"):
        return zeros, {
            "hold": zeros,
            "density": zeros,
            "frontload": zeros,
            "tail_ratio": zeros,
            "coverage": zeros,
            "low_energy": zeros,
            "self_collision": zeros,
        }, {
            "enabled": 0.0,
            "music_motion_pressure": 0.0,
            "musical_silence": 0.0,
        }

    is_silence = _is_musical_silence(phrase)
    if is_silence:
        return zeros, {
            "hold": zeros,
            "density": zeros,
            "frontload": zeros,
            "tail_ratio": zeros,
            "coverage": zeros,
            "low_energy": zeros,
            "self_collision": zeros,
        }, {
            "enabled": 1.0,
            "music_motion_pressure": 0.0,
            "musical_silence": 1.0,
        }

    pressure = max(_env_float("V34_RHYTHM_MIN_PRESSURE", 0.45), _music_motion_pressure(phrase))
    ratio_limit = _env_float("V34_HOLD_FIRST1S_RATIO_LIMIT", 0.55)
    tail_limit = _env_float("V34_HOLD_TAIL_ENERGY_LIMIT", 0.024)
    min_mean = _env_float("V34_MIN_SLOT_MEAN_ENERGY", 0.018)
    late_ratio_min = _env_float("V34_LATE_ENERGY_RATIO_MIN", 0.34)
    tail_mean_ratio_min = _env_float("V34_TAIL_MEAN_RATIO_MIN", 0.55)
    low_energy_fraction_max = _env_float("V34_LOW_ENERGY_FRACTION_MAX", 0.62)
    duration_min = _env_float("V34_DENSITY_SLOT_DURATION_MIN_SEC", 1.5)
    hold_weight = _env_float("V34_HOLD_PENALTY_WEIGHT", 4.00)
    density_weight = _env_float("V34_DENSITY_PENALTY_WEIGHT", 4.50)
    frontload_weight = _env_float("V34_FRONTLOAD_PENALTY_WEIGHT", 4.50)
    tail_ratio_weight = _env_float("V34_TAIL_RATIO_PENALTY_WEIGHT", 3.00)
    coverage_weight = _env_float("V34_COVERAGE_PENALTY_WEIGHT", 2.75)
    low_energy_weight = _env_float("V34_LOW_ENERGY_PENALTY_WEIGHT", 2.00)
    self_collision_weight = _env_float("V34_SELF_COLLISION_PENALTY_WEIGHT", 1.75)
    global_weight = _env_float("V34_RHYTHM_WEIGHT", 1.20)

    ratio_excess = np.maximum(0.0, (first_ratio - ratio_limit) / max(1.0 - ratio_limit, 1e-8))
    frontload_penalty = frontload_weight * np.square(ratio_excess) * pressure
    tail_deficit = np.maximum(0.0, (tail_limit - tail_e) / max(tail_limit, 1e-8))
    hold_penalty = hold_weight * ratio_excess * tail_deficit * pressure

    if float(slot_duration_sec) > duration_min:
        density_deficit = np.maximum(0.0, (min_mean - mean_e) / max(min_mean, 1e-8))
        density_penalty = density_weight * np.square(density_deficit) * pressure
        tail_ratio_deficit = np.maximum(
            0.0,
            (tail_mean_ratio_min - tail_to_mean) / max(tail_mean_ratio_min, 1e-8),
        )
        tail_ratio_penalty = tail_ratio_weight * np.square(tail_ratio_deficit) * pressure
        coverage_deficit = np.maximum(
            0.0,
            (late_ratio_min - late_ratio) / max(late_ratio_min, 1e-8),
        )
        coverage_penalty = coverage_weight * np.square(coverage_deficit) * pressure
        low_energy_excess = np.maximum(
            0.0,
            (low_energy_fraction - low_energy_fraction_max)
            / max(1.0 - low_energy_fraction_max, 1e-8),
        )
        low_energy_penalty = low_energy_weight * np.square(low_energy_excess) * pressure
    else:
        density_penalty = zeros
        tail_ratio_penalty = zeros
        coverage_penalty = zeros
        low_energy_penalty = zeros
    self_collision_penalty = self_collision_weight * self_collision_risk

    total = global_weight * (
        hold_penalty
        + density_penalty
        + frontload_penalty
        + tail_ratio_penalty
        + coverage_penalty
        + low_energy_penalty
        + self_collision_penalty
    )
    return (
        np.asarray(total, dtype=np.float32),
        {
            "hold": np.asarray(global_weight * hold_penalty, dtype=np.float32),
            "density": np.asarray(global_weight * density_penalty, dtype=np.float32),
            "frontload": np.asarray(global_weight * frontload_penalty, dtype=np.float32),
            "tail_ratio": np.asarray(global_weight * tail_ratio_penalty, dtype=np.float32),
            "coverage": np.asarray(global_weight * coverage_penalty, dtype=np.float32),
            "low_energy": np.asarray(global_weight * low_energy_penalty, dtype=np.float32),
            "self_collision": np.asarray(global_weight * self_collision_penalty, dtype=np.float32),
        },
        {
            "enabled": 1.0,
            "music_motion_pressure": float(pressure),
            "musical_silence": 0.0,
            "first1s_ratio_limit": float(ratio_limit),
            "tail_energy_limit": float(tail_limit),
            "min_mean_energy": float(min_mean),
            "late_energy_ratio_min": float(late_ratio_min),
            "tail_mean_ratio_min": float(tail_mean_ratio_min),
            "low_energy_fraction_max": float(low_energy_fraction_max),
        },
    )


def _edge_hub_candidates(
    *,
    feasible_indices: np.ndarray,
    base: np.ndarray,
    event_types: Sequence[str],
    natural: np.ndarray,
    rhythm_features: Dict[str, np.ndarray],
    source_prior_penalty: np.ndarray,
    slot_duration_sec: float,
    fps: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Preserve low-node-score connector candidates before beam expansion.

    A bridge snippet can have mediocre music score while being valuable for
    physical/semantic continuity.  This keeps a small, quality-gated union of
    neutral transition-like events in the shortlist, preventing Node Top-K from
    deleting every useful connector before edge costs are evaluated.
    """
    if not _enabled("V34_EDGE_HUB_RESCUE", "1") or len(feasible_indices) == 0:
        return np.zeros((0,), dtype=np.int64), {"enabled": False, "count": 0}
    top_k = max(0, _env_int("V34_EDGE_HUB_TOP_K", 128))
    if top_k <= 0:
        return np.zeros((0,), dtype=np.int64), {"enabled": True, "count": 0}

    hub_tags = _csv_set(
        "V34_EDGE_HUB_EVENT_TAGS",
        "neutral_flow,calm_flow,build_up,transition,turn,section_change",
    )
    max_seconds = _env_float("V34_EDGE_HUB_MAX_SECONDS", 3.20)
    min_tail_ratio = _env_float("V34_EDGE_HUB_MIN_TAIL_RATIO", 0.42)
    max_first_ratio = _env_float("V34_EDGE_HUB_MAX_FIRST1S_RATIO", 0.82)
    max_collision = _env_float("V34_EDGE_HUB_MAX_COLLISION_RISK", 0.20)

    idx = np.asarray(feasible_indices, dtype=np.int64)
    event_mask = np.asarray(
        [str(event_types[int(i)]) in hub_tags for i in idx],
        dtype=bool,
    )
    duration_mask = natural[idx] <= max_seconds * max(float(fps), 1.0)
    tail_to_mean = rhythm_features.get(
        "tail_to_mean_energy",
        np.zeros_like(natural, dtype=np.float32),
    )
    first_ratio = rhythm_features.get(
        "first1s_energy_ratio",
        np.ones_like(natural, dtype=np.float32),
    )
    collision = rhythm_features.get(
        "self_collision_risk",
        np.zeros_like(natural, dtype=np.float32),
    )
    rhythm_mask = (
        (tail_to_mean[idx] >= min_tail_ratio)
        & (first_ratio[idx] <= max_first_ratio)
        & (collision[idx] <= max_collision)
    )
    mask = event_mask & duration_mask & rhythm_mask
    if not np.any(mask):
        return np.zeros((0,), dtype=np.int64), {
            "enabled": True,
            "count": 0,
            "reason": "no_hub_candidate_passed_gate",
            "event_tags": sorted(hub_tags),
        }

    candidates = idx[mask]
    source_pen = (
        source_prior_penalty[candidates]
        if len(source_prior_penalty) == len(base)
        else np.zeros_like(candidates, dtype=np.float32)
    )
    # Prefer connector candidates with residual tail motion, low front-loading,
    # low collision risk, and reasonable node score.
    hub_score = (
        0.35 * base[candidates]
        + 0.90 * tail_to_mean[candidates]
        - 0.70 * first_ratio[candidates]
        - 1.25 * collision[candidates]
        - 0.45 * source_pen
        - 0.05 * np.abs(
            natural[candidates] / max(slot_duration_sec * max(float(fps), 1.0), 1.0)
            - 1.0
        )
    )
    order = np.argsort(hub_score)[::-1]
    selected = candidates[order[: min(top_k, len(order))]]
    return np.asarray(selected, dtype=np.int64), {
        "enabled": True,
        "count": int(len(selected)),
        "pool_size": int(len(candidates)),
        "top_k": int(top_k),
        "event_tags": sorted(hub_tags),
        "max_seconds": float(max_seconds),
        "min_tail_ratio": float(min_tail_ratio),
        "max_first1s_ratio": float(max_first_ratio),
    }


def _memory_indices(
    selected: Sequence[int],
    selected_parts: Sequence[Mapping[str, Any]] | None,
) -> List[int]:
    if (
        not _enabled("V34_MEMORY_IGNORE_RELAXED", "1")
        or not selected_parts
        or len(selected_parts) != len(selected)
    ):
        return [int(x) for x in selected]
    kept: List[int] = []
    for idx, part in zip(selected, selected_parts):
        meta = dict(part.get("transition_meta", {}) or {})
        relaxation = dict(meta.get("constraint_relaxation", {}) or {})
        relaxed = bool(
            part.get("constraint_relaxation_used", False)
            or part.get("constraint_relaxed", False)
            or relaxation.get("active", False)
            or relaxation.get("used_due_to_empty_strict", False)
        )
        if not relaxed:
            kept.append(int(idx))
    return kept


def _rhythm_streak_penalty(
    *,
    selected: Sequence[int],
    candidate: int,
    event_types: Sequence[str],
) -> Tuple[float, Dict[str, Any]]:
    if not _enabled("V34_RHYTHM_DEGRADATION_PENALTY", "0"):
        return 0.0, {"enabled": False, "score": 0.0}
    static_tags = _csv_set(
        "V34_STATIC_EVENT_TAGS",
        "pose_hold,calm_flow,neutral_flow",
    )
    current_event = str(event_types[int(candidate)])
    streak_count = 1 if current_event in static_tags else 0
    if streak_count:
        for previous in reversed(selected):
            if str(event_types[int(previous)]) in static_tags:
                streak_count += 1
            else:
                break

    allowed = max(1, _env_int("V34_STATIC_STREAK_ALLOW", 1))
    weight = _env_float("V34_STREAK_PENALTY_WEIGHT", 3.00)
    score = 0.0
    if streak_count > allowed:
        score = weight * float((streak_count - allowed) ** 2)
    return float(score), {
        "enabled": True,
        "score": float(score),
        "current_event": current_event,
        "static_tags": sorted(static_tags),
        "streak_count": int(streak_count),
        "allowed_streak": int(allowed),
        "weight": float(weight),
    }


def _source_prior_penalty_arrays(
    *,
    source_ids: Sequence[str],
    dancer_ids: Sequence[str],
    repeat_ids: Sequence[str],
    category_ids: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(source_ids)
    zeros = np.zeros((n,), dtype=np.float32)
    if not _enabled("V34_SOURCE_AWARE_RAG", "1") or n == 0:
        return zeros, {"enabled": False}

    def over_rep_penalty(values: Sequence[str], weight_name: str, default: float) -> np.ndarray:
        weight = _env_float(weight_name, default)
        if weight <= 0:
            return zeros.copy()
        counts: Dict[str, int] = {}
        for value in values:
            counts[str(value)] = counts.get(str(value), 0) + 1
        mean = float(n / max(len(counts), 1))
        out = np.zeros((n,), dtype=np.float32)
        for i, value in enumerate(values):
            ratio = float(counts.get(str(value), 0)) / max(mean, 1e-8)
            out[i] = float(weight * max(0.0, np.log(max(ratio, 1e-8))))
        return out

    category_pen = over_rep_penalty(category_ids, "V34_CATEGORY_PRIOR_BALANCE_WEIGHT", 0.20)
    dancer_pen = over_rep_penalty(dancer_ids, "V34_DANCER_PRIOR_BALANCE_WEIGHT", 0.08)
    repeat_pen = over_rep_penalty(repeat_ids, "V34_REPEAT_PRIOR_BALANCE_WEIGHT", 0.06)
    total = category_pen + dancer_pen + repeat_pen
    return np.asarray(total, dtype=np.float32), {
        "enabled": True,
        "category_prior_weight": _env_float("V34_CATEGORY_PRIOR_BALANCE_WEIGHT", 0.20),
        "dancer_prior_weight": _env_float("V34_DANCER_PRIOR_BALANCE_WEIGHT", 0.08),
        "repeat_prior_weight": _env_float("V34_REPEAT_PRIOR_BALANCE_WEIGHT", 0.06),
    }


def _source_aware_transition_penalty(
    *,
    selected: Sequence[int],
    candidate: int,
    phrase: Any,
    source_uids: Sequence[str],
    dancer_ids: Sequence[str],
    repeat_ids: Sequence[str],
    category_ids: Sequence[str],
    dancer_category_groups: Sequence[str],
) -> Tuple[float, Dict[str, Any]]:
    if not _enabled("V34_SOURCE_AWARE_RAG", "1"):
        return 0.0, {"enabled": False, "score": 0.0}
    if not selected:
        return 0.0, {"enabled": True, "score": 0.0, "window": 0}

    window = max(1, _env_int("V34_SOURCE_AWARE_WINDOW", 8))
    recent = [int(x) for x in selected[-window:]]
    cand = int(candidate)

    source = str(source_uids[cand])
    dancer = str(dancer_ids[cand])
    repeat = str(repeat_ids[cand])
    category = str(category_ids[cand])
    dancer_category = str(dancer_category_groups[cand])

    same_source = sum(1 for x in recent if str(source_uids[int(x)]) == source)
    same_dancer = sum(1 for x in recent if str(dancer_ids[int(x)]) == dancer)
    same_repeat = sum(1 for x in recent if str(repeat_ids[int(x)]) == repeat)
    same_category = sum(1 for x in recent if str(category_ids[int(x)]) == category)
    same_dancer_category = sum(
        1 for x in recent
        if str(dancer_category_groups[int(x)]) == dancer_category
    )

    reset_allow = _slot_reset_allow(phrase)
    # Structural changes may intentionally revisit a category; exact same
    # source and dancer-category repeats remain expensive.
    category_relief = 1.0 - 0.55 * float(reset_allow)
    dancer_relief = 1.0 - 0.35 * float(reset_allow)

    score = (
        _env_float("V34_SOURCE_UID_REPEAT_WEIGHT", 2.50) * float(same_source)
        + _env_float("V34_DANCER_CATEGORY_REPEAT_WEIGHT", 1.20) * float(same_dancer_category)
        + _env_float("V34_CATEGORY_REPEAT_WEIGHT", 0.35) * category_relief * float(same_category)
        + _env_float("V34_DANCER_REPEAT_WEIGHT", 0.25) * dancer_relief * float(same_dancer)
        + _env_float("V34_REPEAT_ID_REPEAT_WEIGHT", 0.12) * float(same_repeat)
    )
    score *= _env_float("V34_SOURCE_AWARE_WEIGHT", 1.0)
    return float(score), {
        "enabled": True,
        "score": float(score),
        "window": int(window),
        "reset_allow": float(reset_allow),
        "candidate": {
            "source_uid": source,
            "dancer_id": dancer,
            "repeat_id": repeat,
            "category_id": category,
            "dancer_category_group": dancer_category,
        },
        "recent_counts": {
            "same_source_uid": int(same_source),
            "same_dancer": int(same_dancer),
            "same_repeat": int(same_repeat),
            "same_category": int(same_category),
            "same_dancer_category": int(same_dancer_category),
        },
        "weights": {
            "source_uid": _env_float("V34_SOURCE_UID_REPEAT_WEIGHT", 2.50),
            "dancer_category": _env_float("V34_DANCER_CATEGORY_REPEAT_WEIGHT", 1.20),
            "category": _env_float("V34_CATEGORY_REPEAT_WEIGHT", 0.35),
            "dancer": _env_float("V34_DANCER_REPEAT_WEIGHT", 0.25),
            "repeat": _env_float("V34_REPEAT_ID_REPEAT_WEIGHT", 0.12),
            "global": _env_float("V34_SOURCE_AWARE_WEIGHT", 1.0),
        },
    }


def _slot_reset_allow(phrase: Any) -> float:
    boundary_strength = float(getattr(phrase, "boundary_accent_strength", 0.0))
    music_event = str(getattr(phrase, "music_event", "neutral_flow"))
    tension = float(getattr(phrase, "tension", 0.0))
    calm = float(getattr(phrase, "calmness", 0.0))
    return float(np.clip(
        0.12 + 0.55 * boundary_strength + 0.20 * tension - 0.12 * calm
        + (0.18 if music_event == "section_change" else 0.0),
        0.0,
        0.85,
    ))


def _semantic_continuity_penalty(
    *,
    selected: Sequence[int],
    selected_parts: Sequence[Mapping[str, Any]] | None = None,
    previous: int,
    candidate: int,
    phrase: Any,
    event_types: Sequence[str],
    families: Sequence[str],
    natural: np.ndarray,
    body_code: np.ndarray,
    activity01: np.ndarray,
    turn01: np.ndarray,
) -> Dict[str, Any]:
    if not _enabled("V34_SEMANTIC_EDGE", "1"):
        return {"enabled": False, "score": 0.0, "hard_reject": False}

    reset_allow = _slot_reset_allow(phrase)
    prev_body = float(body_code[int(previous)])
    next_body = float(body_code[int(candidate)])
    body_jump = abs(next_body - prev_body) / 5.0
    activity_jump = abs(float(activity01[int(candidate)]) - float(activity01[int(previous)]))
    turn_jump = abs(float(turn01[int(candidate)]) - float(turn01[int(previous)]))
    prev_event = str(event_types[int(previous)])
    next_event = str(event_types[int(candidate)])
    event_jump = 1.0 - float(scheduler.event_compatibility(prev_event, next_event))
    duration_jump = abs(float(np.log(
        max(float(natural[int(candidate)]), 1.0)
        / max(float(natural[int(previous)]), 1.0)
    )))

    memory_window = max(1, _env_int("V34_MOTIF_MEMORY_WINDOW", 4))
    memory_selected = _memory_indices(selected, selected_parts)
    recent = [int(x) for x in memory_selected[-memory_window:]]
    if recent:
        memory_activity = abs(float(activity01[int(candidate)]) - float(np.mean(activity01[recent])))
        memory_body = min(abs(float(body_code[int(candidate)]) - float(body_code[int(x)])) / 5.0 for x in recent)
        recent_family_repeat = sum(1 for x in recent if str(families[int(x)]) == str(families[int(candidate)]))
    else:
        memory_activity = 0.0
        memory_body = 0.0
        recent_family_repeat = 0

    body_allow = _env_float("V34_SEMANTIC_MAX_BODY_JUMP", 0.24) + 0.48 * reset_allow
    activity_allow = _env_float("V34_SEMANTIC_MAX_ACTIVITY_JUMP", 0.18) + 0.42 * reset_allow
    turn_allow = _env_float("V34_SEMANTIC_MAX_TURN_JUMP", 0.22) + 0.40 * reset_allow
    event_allow = _env_float("V34_SEMANTIC_MAX_EVENT_JUMP", 0.48) + 0.35 * reset_allow
    duration_allow = _env_float("V34_SEMANTIC_MAX_DURATION_LOG_JUMP", 0.42) + 0.35 * reset_allow
    memory_activity_allow = _env_float("V34_MOTIF_MAX_MEMORY_ACTIVITY_JUMP", 0.26) + 0.35 * reset_allow
    memory_body_allow = _env_float("V34_MOTIF_MAX_MEMORY_BODY_JUMP", 0.36) + 0.35 * reset_allow

    terms = {
        "body": _excess_ratio(body_jump, body_allow),
        "activity": _excess_ratio(activity_jump, activity_allow),
        "turn": _excess_ratio(turn_jump, turn_allow),
        "event": _excess_ratio(event_jump, event_allow),
        "duration": _excess_ratio(duration_jump, duration_allow),
        "memory_activity": _excess_ratio(memory_activity, memory_activity_allow),
        "memory_body": _excess_ratio(memory_body, memory_body_allow),
    }
    score = (
        1.15 * terms["body"]
        + 1.05 * terms["activity"]
        + 0.55 * terms["turn"]
        + 0.90 * terms["event"]
        + 0.55 * terms["duration"]
        + _env_float("V34_MOTIF_MEMORY_WEIGHT", 0.65)
        * (0.65 * terms["memory_activity"] + 0.35 * terms["memory_body"])
    )
    hard_checks = {
        "body": body_jump <= body_allow,
        "activity": activity_jump <= activity_allow,
        "turn": turn_jump <= turn_allow,
        "event": event_jump <= event_allow,
        "duration": duration_jump <= duration_allow,
    }
    if _enabled("V34_MOTIF_MEMORY_HARD_PRUNE", "0"):
        hard_checks.update({
            "memory_activity": memory_activity <= memory_activity_allow,
            "memory_body": memory_body <= memory_body_allow,
        })

    return {
        "enabled": True,
        "score": float(score),
        "hard_reject": bool(
            _enabled("V34_SEMANTIC_EDGE_HARD_PRUNE", "1")
            and not all(bool(x) for x in hard_checks.values())
        ),
        "checks": hard_checks,
        "terms": {key: float(value) for key, value in terms.items()},
        "metrics": {
            "body_jump": float(body_jump),
            "activity_jump": float(activity_jump),
            "turn_jump": float(turn_jump),
            "event_jump": float(event_jump),
            "duration_log_jump": float(duration_jump),
            "memory_activity_jump": float(memory_activity),
            "memory_body_jump": float(memory_body),
            "recent_family_repeat": int(recent_family_repeat),
        },
        "limits": {
            "body_jump": float(body_allow),
            "activity_jump": float(activity_allow),
            "turn_jump": float(turn_allow),
            "event_jump": float(event_allow),
            "duration_log_jump": float(duration_allow),
            "memory_activity_jump": float(memory_activity_allow),
            "memory_body_jump": float(memory_body_allow),
        },
        "context": {
            "reset_allow": float(reset_allow),
            "previous_event": prev_event,
            "candidate_event": next_event,
            "memory_ignore_relaxed": bool(_enabled("V34_MEMORY_IGNORE_RELAXED", "1")),
            "memory_effective_count": int(len(memory_selected)),
        },
    }


def _broad_feasible_mask(
    natural: np.ndarray,
    *,
    slot_length: int,
    first_slot: bool,
    min_content_frames: int,
    transition_min_frames: int,
    transition_max_frames: int,
    minimum_warp: float,
    maximum_warp: float,
    tolerance: float,
) -> np.ndarray:
    """Vectorized version of _integer_content_interval(... ) is not None."""
    natural = np.maximum(np.asarray(natural, dtype=np.float32), 1.0)
    minimum = max(0.0, float(minimum_warp) - float(tolerance))
    maximum = max(minimum, float(maximum_warp) + float(tolerance))
    warp_low = np.ceil(minimum * natural - 1e-7).astype(np.int32)
    warp_high = np.floor(maximum * natural + 1e-7).astype(np.int32)
    slot_length = int(slot_length)
    if first_slot:
        return (warp_low <= slot_length) & (slot_length <= warp_high)

    transition_high = min(int(transition_max_frames), slot_length - int(min_content_frames))
    transition_low = max(0, int(transition_min_frames))
    if transition_high < transition_low:
        return np.zeros_like(natural, dtype=bool)
    content_low = np.maximum.reduce([
        np.full_like(warp_low, int(min_content_frames)),
        np.full_like(warp_low, slot_length - transition_high),
        warp_low,
    ])
    content_high = np.minimum(
        np.full_like(warp_high, slot_length - transition_low),
        warp_high,
    )
    return content_low <= content_high




def _integer_content_interval(
    *,
    natural_duration: float,
    slot_length: int,
    first_slot: bool,
    min_content_frames: int,
    transition_min_frames: int,
    transition_max_frames: int,
    minimum_warp: float,
    maximum_warp: float,
    tolerance: float,
) -> tuple[int, int] | None:
    """Return the integer content-length interval satisfying all hard bounds.

    The music boundary remains locked. For non-first slots, content and transition
    exactly partition the slot. This helper therefore converts the warp interval
    into a candidate-specific legal transition budget instead of rejecting an
    event only after a heuristic transition length has consumed the slot.
    """
    slot_length = int(slot_length)
    natural = max(float(natural_duration), 1.0)
    minimum = max(0.0, float(minimum_warp) - float(tolerance))
    maximum = max(minimum, float(maximum_warp) + float(tolerance))

    warp_low = int(np.ceil(minimum * natural - 1e-7))
    warp_high = int(np.floor(maximum * natural + 1e-7))

    if first_slot:
        return (slot_length, slot_length) if warp_low <= slot_length <= warp_high else None

    transition_high = min(
        int(transition_max_frames),
        slot_length - int(min_content_frames),
    )
    transition_low = max(0, int(transition_min_frames))
    if transition_high < transition_low:
        return None

    content_low = max(
        int(min_content_frames),
        slot_length - transition_high,
        warp_low,
    )
    content_high = min(
        slot_length - transition_low,
        warp_high,
    )
    if content_low > content_high:
        return None
    return int(content_low), int(content_high)


def _negotiate_transition_budget(
    *,
    natural_duration: float,
    slot_length: int,
    desired_transition: int,
    first_slot: bool,
    min_content_frames: int,
    transition_min_frames: int,
    transition_max_frames: int,
    minimum_warp: float,
    maximum_warp: float,
    tolerance: float,
) -> dict[str, float | int | bool] | None:
    """Project a desired transition onto the strict warp-feasible budget.

    The old V34 path treated the heuristic physical transition estimate as a
    hard precondition. With small locked music slots that estimate frequently
    saturates at the slot cap, leaving only ``min_content_frames`` and making
    every real event fail the warp gate. Here the heuristic remains the desired
    budget, while exact warp feasibility is hard. Final physical validity is
    still enforced by the post-generation cross-boundary absolute gate.
    """
    interval = _integer_content_interval(
        natural_duration=natural_duration,
        slot_length=slot_length,
        first_slot=first_slot,
        min_content_frames=min_content_frames,
        transition_min_frames=transition_min_frames,
        transition_max_frames=transition_max_frames,
        minimum_warp=minimum_warp,
        maximum_warp=maximum_warp,
        tolerance=tolerance,
    )
    if interval is None:
        return None

    low, high = interval
    if first_slot:
        content = int(slot_length)
        transition = 0
    else:
        desired_content = int(slot_length) - int(desired_transition)
        content = int(np.clip(desired_content, low, high))
        transition = int(slot_length) - content

    warp_ratio = float(content / max(float(natural_duration), 1.0))
    return {
        "content": int(content),
        "transition": int(transition),
        "warp_ratio": warp_ratio,
        "content_low": int(low),
        "content_high": int(high),
        "desired_transition": int(desired_transition),
        "adjustment_frames": int(transition - int(desired_transition)),
        "adjusted": bool(transition != int(desired_transition)),
    }


def choose_events_v34(
    phrases: Sequence[Any],
    phrase_semantics: np.ndarray,
    predictions: Dict[str, np.ndarray],
    arrays,
    hierarchy,
    items: List[Dict[str, Any]],
    router,
    motions: Sequence[np.ndarray],
    transition_bundle,
    device: torch.device,
    args,
):
    motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    mmr_embed = np.asarray(arrays["mmr_embed"], dtype=np.float32)
    style = np.asarray(arrays["style_score"], dtype=np.float32)
    quality = np.asarray(arrays["quality_score"], dtype=np.float32)
    safety = np.asarray(arrays["safety_score"], dtype=np.float32)
    natural = np.asarray(arrays["natural_duration"], dtype=np.float32)
    names = set(arrays.files) if hasattr(arrays, "files") else set(arrays.keys())
    turn_peak_dps = (
        np.asarray(arrays["turn_peak_dps"], dtype=np.float32)
        if "turn_peak_dps" in names else np.zeros_like(natural)
    )
    turn_angle_deg = (
        np.asarray(arrays["turn_angle_deg"], dtype=np.float32)
        if "turn_angle_deg" in names else np.zeros_like(natural)
    )
    hierarchy_body_fallback = np.full((len(natural),), 2.0, dtype=np.float32)
    hierarchy_activity_fallback = (
        motion_desc[:, 0].astype(np.float32)
        if motion_desc.ndim == 2 and motion_desc.shape[1] > 0
        else np.full((len(natural),), 0.5, dtype=np.float32)
    )
    body_code = _array_or(hierarchy, "body_code", hierarchy_body_fallback, dtype=np.float32)
    activity01 = _array_or(hierarchy, "activity01", hierarchy_activity_fallback, dtype=np.float32)
    turn01 = _array_or(hierarchy, "turn01", np.zeros((len(natural),), dtype=np.float32), dtype=np.float32)
    if len(body_code) != len(natural):
        body_code = hierarchy_body_fallback
    if len(activity01) != len(natural):
        activity01 = hierarchy_activity_fallback
    if len(turn01) != len(natural):
        turn01 = np.zeros((len(natural),), dtype=np.float32)
    entry_pose = np.asarray(arrays["entry_pose"], dtype=np.float32)
    exit_pose = np.asarray(arrays["exit_pose"], dtype=np.float32)
    entry_vel = np.asarray(arrays["entry_vel"], dtype=np.float32)
    exit_vel = np.asarray(arrays["exit_vel"], dtype=np.float32)
    event_types = [
        str(item.get("event_type", "neutral_flow")) for item in items
    ]
    families = [str(item.get("family_id", "")) for item in items]
    source_identity = identity_lists(items)
    dancer_ids = source_identity["dancer_id"]
    repeat_ids = source_identity["repeat_id"]
    category_ids = source_identity["category_id"]
    source_uids = source_identity["source_uid"]
    dancer_category_groups = source_identity["dancer_category_group"]
    source_prior_penalty, source_prior_meta = _source_prior_penalty_arrays(
        source_ids=source_uids,
        dancer_ids=dancer_ids,
        repeat_ids=repeat_ids,
        category_ids=category_ids,
    )
    native_floor_penalty, native_floor_meta = _native_floor_penalty_arrays(items, motions)
    queries = [np.asarray(phrase.query, np.float32) for phrase in phrases]
    similarities = scheduler.precompute_music_similarity(
        router, queries, motion_desc, device
    )
    transition_choices = scheduler.planner_bundle_lengths(args.planner_ckpt)
    gpu_cache = None
    if build_v34_gpu_candidate_cache is not None:
        gpu_cache = build_v34_gpu_candidate_cache(arrays, motions, device)
    rhythm_features = _build_rhythm_feature_arrays(
        motions,
        fps=int(getattr(args, "fps", 30)),
    )
    if len(rhythm_features["mean_energy"]) != len(natural):
        rhythm_features = {
            "mean_energy": np.zeros_like(natural, dtype=np.float32),
            "p90_energy": np.zeros_like(natural, dtype=np.float32),
            "max_energy": np.zeros_like(natural, dtype=np.float32),
            "first1s_energy_ratio": np.zeros_like(natural, dtype=np.float32),
            "tail_mean_energy": np.zeros_like(natural, dtype=np.float32),
            "late_energy_ratio": np.zeros_like(natural, dtype=np.float32),
            "tail_to_mean_energy": np.zeros_like(natural, dtype=np.float32),
            "low_energy_fraction": np.ones_like(natural, dtype=np.float32),
            "self_collision_risk": np.zeros_like(natural, dtype=np.float32),
            "self_collision_min_distance": np.ones_like(natural, dtype=np.float32),
        }

    minimum = float(os.getenv("V34_WARP_MIN", str(args.min_time_warp)))
    maximum = float(os.getenv("V34_WARP_MAX", str(args.max_time_warp)))
    relaxed_minimum = float(os.getenv("V34_WARP_RELAX_MIN", str(args.min_time_warp)))
    relaxed_maximum = float(os.getenv("V34_WARP_RELAX_MAX", str(args.max_time_warp)))
    tolerance = float(os.getenv("V34_WARP_TOLERANCE", "0.0"))
    hard_prune = _enabled("V34_WARP_HARD_PRUNE", "1")
    warp_weight = float(os.getenv("V34_WARP_PENALTY_WEIGHT", "1.25"))
    requested_top_k = int(os.getenv("V34_WARP_PREFILTER_TOP_K", "512"))
    compat_enabled = (
        _enabled("V34_BOUNDARY_COMPAT", "1")
        and evaluate_boundary_compatibility is not None
    )
    compat_hard_prune = _enabled("V34_COMPAT_HARD_PRUNE", "1")
    compat_weight = float(os.getenv("V34_BOUNDARY_COMPAT_WEIGHT", "1.20"))
    semantic_edge_weight = float(os.getenv("V34_SEMANTIC_EDGE_WEIGHT", "1.10"))
    relax_constraints_on_empty = _enabled("V34_RELAX_CONSTRAINTS_ON_EMPTY", "1")
    relax_compat_on_empty = (
        relax_constraints_on_empty
        and _enabled("V34_RELAX_COMPAT_ON_EMPTY", "1")
    )
    relax_semantic_on_empty = (
        relax_constraints_on_empty
        and _enabled("V34_RELAX_SEMANTIC_ON_EMPTY", "1")
    )
    compat_relax_penalty_weight = _env_float(
        "V34_RELAX_COMPAT_PENALTY_WEIGHT",
        max(5.0, 3.0 * compat_weight),
    )
    semantic_relax_penalty_weight = _env_float(
        "V34_RELAX_SEMANTIC_PENALTY_WEIGHT",
        max(4.5, 3.0 * semantic_edge_weight),
    )
    contact_relax_penalty_weight = _env_float(
        "V34_RELAX_CONTACT_PENALTY_WEIGHT",
        2.0,
    )
    relax_rescue_top_k = max(1, _env_int("V34_RELAX_RESCUE_TOP_K", 768))

    beam = [scheduler.CandidateState(0.0, [], [], [])]
    for slot, phrase in enumerate(phrases):
        slot_duration_sec = float(phrase.length) / max(float(getattr(args, "fps", 30)), 1.0)
        predicted_event = scheduler.EVENT_TYPES[
            int(predictions["event_ids"][slot])
        ]
        predicted_duration = float(predictions["durations"][slot])
        desired_activity = float(predictions["activity"][slot])
        compat = np.asarray([
            0.60 * scheduler.event_compatibility(phrase.music_event, event)
            + 0.40 * (
                1.0 if event == predicted_event else
                scheduler.event_compatibility(predicted_event, event)
            )
            for event in event_types
        ], dtype=np.float32)

        transition_guess = 0 if slot == 0 else int(phrase.transition_base_frames)
        slot_content_target = max(
            float(args.min_content_frames),
            float(
                phrase.length
                - min(
                    transition_guess,
                    max(0, phrase.length - args.min_content_frames),
                )
            ),
        )
        target_natural = max(
            float(args.min_content_frames),
            slot_content_target * max(float(phrase.speed_factor), 1e-6),
        )
        duration_match = 1.0 - np.minimum(
            np.abs(natural - target_natural) / max(target_natural, 1.0), 1.0
        )
        planner_duration_match = 1.0 - np.minimum(
            np.abs(natural - predicted_duration) / max(predicted_duration, 1.0),
            1.0,
        )
        activity_match = 1.0 - np.minimum(
            np.abs(motion_desc[:, 0] - desired_activity), 1.0
        )
        low_activity = np.clip(
            (float(args.anti_static_activity_threshold) - motion_desc[:, 0])
            / max(float(args.anti_static_activity_threshold), 1e-6),
            0.0, 1.0,
        )
        long_slot_pressure = np.clip(
            (slot_content_target - float(args.anti_static_min_content_frames))
            / max(
                float(args.max_single_event_seconds * args.fps)
                - float(args.anti_static_min_content_frames),
                1.0,
            ),
            0.0, 1.0,
        )
        music_motion_need = np.clip(
            0.42 * float(phrase.energy)
            + 0.26 * float(phrase.beat_density)
            + 0.20 * float(phrase.onset)
            + 0.12 * float(phrase.tension)
            - 0.22 * float(phrase.calmness),
            0.0, 1.0,
        )
        anti_static_penalty = low_activity * max(
            float(long_slot_pressure), float(music_motion_need)
        )
        turn_soft = float(args.turn_peak_soft_dps)
        turn_hard = max(float(args.turn_peak_hard_dps), turn_soft + 1.0)
        turn_over = np.clip(
            (turn_peak_dps - turn_soft) / (turn_hard - turn_soft), 0.0, 1.0
        )
        turn_angle_over = np.clip(
            (turn_angle_deg - args.turn_angle_soft_deg)
            / max(args.turn_angle_hard_deg - args.turn_angle_soft_deg, 1.0),
            0.0, 1.0,
        )
        turn_penalty = 0.75 * turn_over + 0.25 * turn_angle_over

        hierarchy_score = np.zeros_like(style, dtype=np.float32)
        hierarchy_components: Dict[str, np.ndarray] = {}
        hierarchy_query: Dict[str, Any] = {}
        if args.hierarchical_retrieval:
            hierarchy_query = scheduler.build_slot_query(
                phrase,
                predicted_event=predicted_event,
                target_natural=target_natural,
                desired_activity=desired_activity,
                music_semantic=(
                    phrase_semantics[slot]
                    if len(phrase_semantics) > slot else None
                ),
                deep_music_weight=(
                    args.deep_music_weight if args.deep_music_features else 0.0
                ),
            )
            hierarchy_score, hierarchy_components = (
                scheduler.hierarchical_node_scores(hierarchy, hierarchy_query)
            )

        # Approximate warp is a soft ranking signal only.  Exact feasibility is
        # checked after the candidate-specific transition length is computed.
        approximate_warp = slot_content_target / np.maximum(natural, 1.0)
        approximate_warp_penalty = np.abs(np.log(np.maximum(approximate_warp, 1e-6)))
        base = (
            args.style_weight * style
            + args.quality_weight * quality
            + args.safety_weight * safety
            + args.music_weight * similarities[slot]
            + args.event_weight * compat
            + args.duration_weight * duration_match
            + args.planner_duration_weight * planner_duration_match
            + args.activity_weight * activity_match
            + args.hierarchy_weight * hierarchy_score
            - args.anti_static_weight * anti_static_penalty
            - args.turn_peak_penalty_weight * turn_penalty
            - 0.35 * warp_weight * approximate_warp_penalty
        )
        rhythm_prior_penalty, rhythm_prior_terms, rhythm_slot_context = (
            _rhythm_prior_penalty_arrays(
                rhythm_features=rhythm_features,
                phrase=phrase,
                slot_duration_sec=slot_duration_sec,
            )
        )
        base = base - rhythm_prior_penalty
        if len(source_prior_penalty) == len(base):
            base = base - source_prior_penalty
        if len(native_floor_penalty) == len(base):
            base = base - native_floor_penalty

        node_top_k = min(
            int(args.candidate_top_k),
            max(int(args.graph_node_top_k), requested_top_k),
        )

        # Build the shortlist *inside* the broad hard-feasible duration set.
        # The previous implementation ranked all 4,225 events first and only
        # then tested the top-K. Short events needed by a small slot could be
        # absent from that top-K even when they existed in the database.
        slot_minimum = minimum
        slot_maximum = maximum
        warp_relaxed = False
        broad_feasible = _broad_feasible_mask(
            natural,
            slot_length=int(phrase.length),
            first_slot=(slot == 0),
            min_content_frames=int(args.min_content_frames),
            transition_min_frames=int(args.transition_min_frames),
            transition_max_frames=int(args.transition_max_frames),
            minimum_warp=minimum,
            maximum_warp=maximum,
            tolerance=tolerance,
        )
        feasible_indices = np.flatnonzero(broad_feasible)
        if (
            len(feasible_indices) == 0
            and _enabled("V34_WARP_RELAX_ON_EMPTY", "1")
            and (relaxed_minimum < minimum or relaxed_maximum > maximum)
        ):
            relaxed_mask = _broad_feasible_mask(
                natural,
                slot_length=int(phrase.length),
                first_slot=(slot == 0),
                min_content_frames=int(args.min_content_frames),
                transition_min_frames=int(args.transition_min_frames),
                transition_max_frames=int(args.transition_max_frames),
                minimum_warp=relaxed_minimum,
                maximum_warp=relaxed_maximum,
                tolerance=tolerance,
            )
            relaxed_indices = np.flatnonzero(relaxed_mask)
            if len(relaxed_indices) > 0:
                broad_feasible = relaxed_mask
                feasible_indices = relaxed_indices
                slot_minimum = relaxed_minimum
                slot_maximum = relaxed_maximum
                warp_relaxed = True
        if len(feasible_indices) == 0:
            natural_min = float(np.min(natural)) if len(natural) else float("nan")
            natural_max = float(np.max(natural)) if len(natural) else float("nan")
            raise RuntimeError(
                f"V34 slot has no globally warp-feasible event: slot={slot}, "
                f"music_length={phrase.length}, bounds=[{minimum},{maximum}], "
                f"relaxed_bounds=[{relaxed_minimum},{relaxed_maximum}], "
                f"natural_range=[{natural_min},{natural_max}], "
                f"min_content={args.min_content_frames}, "
                f"transition_range=[{args.transition_min_frames},"
                f"{args.transition_max_frames}]. Merge/repartition the music "
                "slot; making it shorter cannot restore feasibility."
            )
        ranked_feasible = feasible_indices[
            np.argsort(base[feasible_indices])[::-1]
        ]
        node_shortlist = ranked_feasible[: min(node_top_k, len(ranked_feasible))]
        edge_hub_shortlist, edge_hub_meta = _edge_hub_candidates(
            feasible_indices=feasible_indices,
            base=base,
            event_types=event_types,
            natural=natural,
            rhythm_features=rhythm_features,
            source_prior_penalty=source_prior_penalty,
            slot_duration_sec=slot_duration_sec,
            fps=int(getattr(args, "fps", 30)),
        )
        shortlist = _unique_indices(node_shortlist, edge_hub_shortlist)
        strict_expanded: List[Any] = []
        relaxed_expanded: List[Any] = []
        expanded: List[Any] = []
        rejected_warp = 0
        compat_rejected = 0
        semantic_rejected = 0
        compat_deferred = 0
        semantic_deferred = 0
        contact_deferred = 0
        negotiated_count = 0
        budget_penalty_weight = float(
            os.getenv("V34_TRANSITION_BUDGET_PENALTY_WEIGHT", "0.035")
        )
        slot_boundary_cache = None
        if gpu_cache is not None and slot > 0:
            previous_indices = [
                int(state.selected[-1]) for state in beam if state.selected
            ]
            try:
                slot_boundary_cache = gpu_cache.compute_slot(
                    previous_indices,
                    shortlist,
                    phrase,
                    args,
                )
            except Exception as exc:
                if _enabled("V34_GPU_STRICT", "0"):
                    raise
                if _enabled("V34_GPU_RETRIEVAL_VERBOSE", "1"):
                    print(f"[V34-GPU] slot={slot} fallback to CPU: {exc}")

        for state in beam:
            for raw_idx in shortlist:
                idx = int(raw_idx)
                if idx in state.selected:
                    continue
                family = families[idx]
                same_family = sum(
                    1 for previous in state.selected
                    if families[previous] == family
                )
                same_source = sum(
                    1 for previous in state.selected
                    if str(source_uids[int(previous)]) == str(source_uids[idx])
                )
                if args.hard_family_unique and same_family > 0:
                    continue

                transition_len = 0
                transition_cost = 0.0
                boundary_velocity_penalty = 0.0
                boundary_acceleration_penalty = 0.0
                graph_edge_cost = 0.0
                graph_edge_meta: Dict[str, Any] = {}
                transition_meta: Dict[str, Any] = {}
                gpu_boundary_cache_hit = False
                boundary_compat_score = 0.0
                boundary_compat_meta: Dict[str, Any] = {"enabled": False}
                semantic_edge_score = 0.0
                semantic_edge_meta: Dict[str, Any] = {"enabled": False}
                constraint_relaxed = False
                compat_relaxed = False
                semantic_relaxed = False
                contact_relaxed = False
                relaxation_reasons: List[str] = []
                relaxation_penalty = 0.0
                if state.selected:
                    previous = state.selected[-1]
                    cached_boundary = (
                        slot_boundary_cache.get(previous, idx, args)
                        if slot_boundary_cache is not None else None
                    )
                    if cached_boundary is not None:
                        gpu_boundary_cache_hit = True
                        transition_cost = float(cached_boundary["transition_cost"])
                        candidate_boundary = dict(cached_boundary["candidate_boundary"])
                        boundary_velocity_penalty = float(
                            cached_boundary["boundary_velocity_penalty"]
                        )
                        boundary_acceleration_penalty = float(
                            cached_boundary["boundary_acceleration_penalty"]
                        )
                        if args.music_dominant_timing:
                            transition_len = int(cached_boundary["transition_len"])
                            transition_meta = {
                                **dict(cached_boundary["transition_meta"]),
                                "candidate_boundary": candidate_boundary,
                            }
                    if not gpu_boundary_cache_hit:
                        transition_cost = scheduler.transition_cost_from_arrays(
                            exit_pose[previous], exit_vel[previous],
                            entry_pose[idx], entry_vel[idx],
                        )
                        candidate_boundary = scheduler.boundary_metrics(
                            motions[previous], motions[idx]
                        )
                        boundary_velocity_penalty = min(
                            candidate_boundary["velocity_jump"]
                            / max(args.velocity_jump_reference, 1e-6),
                            args.boundary_penalty_cap,
                        )
                        boundary_acceleration_penalty = min(
                            candidate_boundary["acceleration_jump"]
                            / max(args.acceleration_jump_reference, 1e-6),
                            args.boundary_penalty_cap,
                        )
                        if args.music_dominant_timing:
                            transition_len, transition_meta = (
                                scheduler.dynamic_transition_len(
                                    motions[previous], motions[idx], phrase, args
                                )
                            )
                            transition_meta = {
                                **transition_meta,
                                "candidate_boundary": candidate_boundary,
                            }
                    if not args.music_dominant_timing:
                        class_index = int(predictions["transition_class"][slot])
                        transition_len = int(
                            transition_choices[
                                min(class_index, len(transition_choices) - 1)
                            ]
                        )
                        transition_meta = {
                            "chosen_transition_frames": transition_len,
                            "dominant_reason": "planner_class",
                        }
                    if args.graph_scheduler:
                        prev_prev = (
                            state.selected[-2] if len(state.selected) >= 2 else None
                        )
                        graph_edge_cost, graph_edge_meta = (
                            scheduler.hierarchical_graph_edge_penalty(
                                hierarchy,
                                previous,
                                idx,
                                phrase,
                                prev_prev_idx=prev_prev,
                            )
                        )
                        if (
                            args.graph_hard_prune
                            and graph_edge_cost > args.graph_hard_prune_threshold
                        ):
                            continue
                    if compat_enabled:
                        boundary_compat_meta = evaluate_boundary_compatibility(
                            previous_index=int(previous),
                            candidate_index=int(idx),
                            candidate_boundary=candidate_boundary,
                            transition_cost=float(transition_cost),
                            phrase=phrase,
                            args=args,
                            hierarchy=hierarchy,
                        )
                        boundary_compat_score = float(
                            boundary_compat_meta.get("score", 0.0)
                        )
                        if (
                            compat_hard_prune
                            and bool(boundary_compat_meta.get("hard_reject", False))
                        ):
                            compat_rejected += 1
                            if not relax_compat_on_empty:
                                continue
                            constraint_relaxed = True
                            compat_relaxed = True
                            compat_deferred += 1
                            relaxation_reasons.append("boundary_compatibility")
                            failed_checks = [
                                str(key)
                                for key, ok in dict(
                                    boundary_compat_meta.get("checks", {})
                                ).items()
                                if not bool(ok)
                            ]
                            contact_failed = any(
                                key in {
                                    "contact",
                                    "contact_binary",
                                    "support_count",
                                    "aerial_planted",
                                    "stance_flip",
                                }
                                for key in failed_checks
                            )
                            if contact_failed:
                                contact_relaxed = True
                                contact_deferred += 1
                                relaxation_reasons.append("contact_state")
                            relaxation_penalty += (
                                compat_relax_penalty_weight
                                * (1.0 + boundary_compat_score)
                            )
                            if contact_failed:
                                relaxation_penalty += contact_relax_penalty_weight
                        transition_meta = dict(transition_meta)
                        transition_meta["boundary_compatibility"] = (
                            boundary_compat_meta
                        )
                    semantic_edge_meta = _semantic_continuity_penalty(
                        selected=state.selected,
                        selected_parts=state.parts,
                        previous=int(previous),
                        candidate=int(idx),
                        phrase=phrase,
                        event_types=event_types,
                        families=families,
                        natural=natural,
                        body_code=body_code,
                        activity01=activity01,
                        turn01=turn01,
                    )
                    semantic_edge_score = float(semantic_edge_meta.get("score", 0.0))
                    if bool(semantic_edge_meta.get("hard_reject", False)):
                        semantic_rejected += 1
                        if not relax_semantic_on_empty:
                            continue
                        constraint_relaxed = True
                        semantic_relaxed = True
                        semantic_deferred += 1
                        relaxation_reasons.append("semantic_continuity")
                        relaxation_penalty += (
                            semantic_relax_penalty_weight
                            * (1.0 + semantic_edge_score)
                        )
                    transition_meta = dict(transition_meta)
                    transition_meta["semantic_continuity"] = semantic_edge_meta

                desired_transition_len = int(transition_len)
                negotiated = _negotiate_transition_budget(
                    natural_duration=float(natural[idx]),
                    slot_length=int(phrase.length),
                    desired_transition=desired_transition_len,
                    first_slot=(slot == 0),
                    min_content_frames=int(args.min_content_frames),
                    transition_min_frames=int(args.transition_min_frames),
                    transition_max_frames=int(args.transition_max_frames),
                    minimum_warp=slot_minimum,
                    maximum_warp=slot_maximum,
                    tolerance=tolerance,
                )
                if negotiated is None:
                    rejected_warp += 1
                    if hard_prune:
                        continue
                    exact_content = max(
                        int(args.min_content_frames),
                        int(phrase.length) - desired_transition_len,
                    )
                    transition_len = desired_transition_len
                    warp_ratio = float(
                        exact_content / max(float(natural[idx]), 1.0)
                    )
                    feasible = False
                    budget_adjustment = 0
                    feasible_content_interval = None
                else:
                    exact_content = int(negotiated["content"])
                    transition_len = int(negotiated["transition"])
                    warp_ratio = float(negotiated["warp_ratio"])
                    feasible = True
                    budget_adjustment = int(negotiated["adjustment_frames"])
                    feasible_content_interval = [
                        int(negotiated["content_low"]),
                        int(negotiated["content_high"]),
                    ]
                    if bool(negotiated["adjusted"]):
                        negotiated_count += 1

                warp_penalty = abs(float(np.log(max(warp_ratio, 1e-6))))
                transition_budget_penalty = (
                    budget_penalty_weight * abs(float(budget_adjustment))
                )
                transition_meta = dict(transition_meta)
                transition_meta["pre_warp_negotiation_frames"] = int(
                    desired_transition_len
                )
                transition_meta["chosen_transition_frames"] = int(
                    transition_len
                )
                transition_meta["warp_budget_negotiated"] = bool(
                    budget_adjustment != 0
                )
                transition_meta["warp_budget_adjustment_frames"] = int(
                    budget_adjustment
                )
                transition_meta["feasible_content_interval"] = (
                    feasible_content_interval
                )
                if relaxation_reasons:
                    # Preserve order while removing duplicates.
                    relaxation_reasons = list(dict.fromkeys(relaxation_reasons))
                transition_meta["constraint_relaxation"] = {
                    "enabled": bool(relax_constraints_on_empty),
                    "active": bool(constraint_relaxed),
                    "used_due_to_empty_strict": False,
                    "compat_relaxed": bool(compat_relaxed),
                    "semantic_relaxed": bool(semantic_relaxed),
                    "contact_relaxed": bool(contact_relaxed),
                    "reasons": relaxation_reasons,
                    "penalty": float(relaxation_penalty),
                    "penalty_weights": {
                        "compat": float(compat_relax_penalty_weight),
                        "semantic": float(semantic_relax_penalty_weight),
                        "contact": float(contact_relax_penalty_weight),
                    },
                }
                transition_meta["compat_relaxed"] = bool(compat_relaxed)
                transition_meta["semantic_relaxed"] = bool(semantic_relaxed)
                transition_meta["contact_relaxed"] = bool(contact_relaxed)

                mmr = 0.0
                if state.selected:
                    mmr = max(
                        float(mmr_embed[idx] @ mmr_embed[previous])
                        for previous in state.selected
                    )
                rhythm_streak_score, rhythm_streak_meta = _rhythm_streak_penalty(
                    selected=state.selected,
                    candidate=idx,
                    event_types=event_types,
                )
                source_aware_penalty, source_aware_meta = (
                    _source_aware_transition_penalty(
                        selected=state.selected,
                        candidate=idx,
                        phrase=phrase,
                        source_uids=source_uids,
                        dancer_ids=dancer_ids,
                        repeat_ids=repeat_ids,
                        category_ids=category_ids,
                        dancer_category_groups=dancer_category_groups,
                    )
                )
                rhythm_penalty = float(rhythm_prior_penalty[idx]) + float(
                    rhythm_streak_score
                )
                rhythm_meta = {
                    "enabled": bool(
                        _enabled("V34_RHYTHM_DEGRADATION_PENALTY", "0")
                    ),
                    "score": float(rhythm_penalty),
                    "prior_penalty": float(rhythm_prior_penalty[idx]),
                    "hold_penalty": float(rhythm_prior_terms["hold"][idx]),
                    "density_penalty": float(rhythm_prior_terms["density"][idx]),
                    "frontload_penalty": float(rhythm_prior_terms["frontload"][idx]),
                    "tail_ratio_penalty": float(rhythm_prior_terms["tail_ratio"][idx]),
                    "coverage_penalty": float(rhythm_prior_terms["coverage"][idx]),
                    "low_energy_penalty": float(rhythm_prior_terms["low_energy"][idx]),
                    "self_collision_penalty": float(
                        rhythm_prior_terms["self_collision"][idx]
                    ),
                    "streak_penalty": float(rhythm_streak_score),
                    "streak": rhythm_streak_meta,
                    "features": {
                        "mean_energy": float(rhythm_features["mean_energy"][idx]),
                        "p90_energy": float(rhythm_features["p90_energy"][idx]),
                        "max_energy": float(rhythm_features["max_energy"][idx]),
                        "first1s_energy_ratio": float(
                            rhythm_features["first1s_energy_ratio"][idx]
                        ),
                        "tail_mean_energy": float(
                            rhythm_features["tail_mean_energy"][idx]
                        ),
                        "late_energy_ratio": float(
                            rhythm_features["late_energy_ratio"][idx]
                        ),
                        "tail_to_mean_energy": float(
                            rhythm_features["tail_to_mean_energy"][idx]
                        ),
                        "low_energy_fraction": float(
                            rhythm_features["low_energy_fraction"][idx]
                        ),
                        "self_collision_risk": float(
                            rhythm_features["self_collision_risk"][idx]
                        ),
                        "self_collision_min_distance": float(
                            rhythm_features["self_collision_min_distance"][idx]
                        ),
                    },
                    "slot_context": {
                        key: (
                            bool(value)
                            if key == "musical_silence"
                            else float(value)
                        )
                        for key, value in rhythm_slot_context.items()
                    },
                }
                transition_meta = dict(transition_meta)
                transition_meta["rhythm_degradation"] = rhythm_meta
                transition_meta["source_aware_rag"] = source_aware_meta
                score = (
                    state.score
                    + float(base[idx])
                    - args.transition_weight * transition_cost
                    - args.boundary_velocity_penalty_weight
                    * boundary_velocity_penalty
                    - args.boundary_acceleration_penalty_weight
                    * boundary_acceleration_penalty
                    - args.graph_edge_weight * graph_edge_cost
                    - compat_weight * boundary_compat_score
                    - semantic_edge_weight * semantic_edge_score
                    - args.mmr_weight * mmr
                    - args.family_repeat_weight * same_family
                    - args.source_repeat_weight * same_source
                    - warp_weight * warp_penalty
                    - transition_budget_penalty
                    - relaxation_penalty
                    - rhythm_streak_score
                    - source_aware_penalty
                )
                part = {
                    "slot": slot,
                    "music_start": phrase.start,
                    "music_end": phrase.end,
                    "music_length": phrase.length,
                    "music_event": phrase.music_event,
                    "music_speed_factor": float(phrase.speed_factor),
                    "music_transition_profile": phrase.transition_profile,
                    "boundary_accent_strength": float(
                        phrase.boundary_accent_strength
                    ),
                    "predicted_motion_event": predicted_event,
                    "predicted_duration": predicted_duration,
                    "event_index": idx,
                    "event_id": str(items[idx].get("event_id", idx)),
                    "family_id": family,
                    "source_uid": str(source_uids[idx]),
                    "dancer_id": str(dancer_ids[idx]),
                    "repeat_id": str(repeat_ids[idx]),
                    "category_id": str(category_ids[idx]),
                    "dancer_category_group": str(dancer_category_groups[idx]),
                    "motion_event": event_types[idx],
                    "natural_duration": float(natural[idx]),
                    "slot_content_target": float(slot_content_target),
                    "exact_content_target": int(exact_content),
                    "target_natural_duration": float(target_natural),
                    "desired_transition_len": int(desired_transition_len),
                    "negotiated_transition_len": int(transition_len),
                    "transition_budget_adjustment_frames": int(
                        budget_adjustment
                    ),
                    "transition_budget_penalty": float(
                        transition_budget_penalty
                    ),
                    "feasible_content_interval": feasible_content_interval,
                    "warp_ratio_at_retrieval": warp_ratio,
                    "warp_feasible": bool(feasible),
                    "warp_bounds": [minimum, maximum],
                    "effective_warp_bounds": [slot_minimum, slot_maximum],
                    "warp_relaxed": bool(warp_relaxed),
                    "warp_penalty": float(warp_penalty),
                    "transition_len": int(transition_len),
                    "transition_meta": transition_meta,
                    "constraint_relaxed": bool(constraint_relaxed),
                    "compat_relaxed": bool(compat_relaxed),
                    "semantic_relaxed": bool(semantic_relaxed),
                    "contact_relaxed": bool(contact_relaxed),
                    "constraint_relaxation_reasons": relaxation_reasons,
                    "constraint_relaxation_penalty": float(relaxation_penalty),
                    "constraint_relaxation_used": False,
                    "relax_constraints_on_empty": bool(relax_constraints_on_empty),
                    "style": float(style[idx]),
                    "quality": float(quality[idx]),
                    "safety": float(safety[idx]),
                    "music_similarity": float(similarities[slot, idx]),
                    "event_compatibility": float(compat[idx]),
                    "duration_match": float(duration_match[idx]),
                    "planner_duration_match": float(
                        planner_duration_match[idx]
                    ),
                    "activity_match": float(activity_match[idx]),
                    "anti_static_penalty": float(anti_static_penalty[idx]),
                    "turn_peak_dps": float(turn_peak_dps[idx]),
                    "turn_angle_deg": float(turn_angle_deg[idx]),
                    "turn_penalty": float(turn_penalty[idx]),
                    "rhythm_degradation_enabled": bool(
                        _enabled("V34_RHYTHM_DEGRADATION_PENALTY", "0")
                    ),
                    "rhythm_degradation_penalty": float(rhythm_penalty),
                    "rhythm_prior_penalty": float(rhythm_prior_penalty[idx]),
                    "rhythm_hold_penalty": float(
                        rhythm_prior_terms["hold"][idx]
                    ),
                    "rhythm_density_penalty": float(
                        rhythm_prior_terms["density"][idx]
                    ),
                    "rhythm_frontload_penalty": float(
                        rhythm_prior_terms["frontload"][idx]
                    ),
                    "rhythm_tail_ratio_penalty": float(
                        rhythm_prior_terms["tail_ratio"][idx]
                    ),
                    "rhythm_coverage_penalty": float(
                        rhythm_prior_terms["coverage"][idx]
                    ),
                    "rhythm_low_energy_penalty": float(
                        rhythm_prior_terms["low_energy"][idx]
                    ),
                    "rhythm_self_collision_penalty": float(
                        rhythm_prior_terms["self_collision"][idx]
                    ),
                    "rhythm_streak_penalty": float(rhythm_streak_score),
                    "rhythm_mean_energy": float(
                        rhythm_features["mean_energy"][idx]
                    ),
                    "rhythm_p90_energy": float(
                        rhythm_features["p90_energy"][idx]
                    ),
                    "rhythm_first1s_energy_ratio": float(
                        rhythm_features["first1s_energy_ratio"][idx]
                    ),
                    "rhythm_tail_mean_energy": float(
                        rhythm_features["tail_mean_energy"][idx]
                    ),
                    "rhythm_late_energy_ratio": float(
                        rhythm_features["late_energy_ratio"][idx]
                    ),
                    "rhythm_tail_to_mean_energy": float(
                        rhythm_features["tail_to_mean_energy"][idx]
                    ),
                    "rhythm_low_energy_fraction": float(
                        rhythm_features["low_energy_fraction"][idx]
                    ),
                    "rhythm_self_collision_risk": float(
                        rhythm_features["self_collision_risk"][idx]
                    ),
                    "rhythm_self_collision_min_distance": float(
                        rhythm_features["self_collision_min_distance"][idx]
                    ),
                    "rhythm_meta": rhythm_meta,
                    "source_aware_rag_enabled": bool(
                        _enabled("V34_SOURCE_AWARE_RAG", "1")
                    ),
                    "source_aware_prior_penalty": float(
                        source_prior_penalty[idx]
                        if len(source_prior_penalty) == len(base) else 0.0
                    ),
                    "source_aware_transition_penalty": float(
                        source_aware_penalty
                    ),
                    "source_aware_meta": source_aware_meta,
                    "source_aware_prior_meta": source_prior_meta,
                    "v40_native_floor_prior_meta": native_floor_meta,
                    "v40_native_floor_penalty": float(native_floor_penalty[idx]) if len(native_floor_penalty) == len(base) else 0.0,
                    "candidate_top_k": int(args.candidate_top_k),
                    "graph_node_top_k": int(node_top_k),
                    "node_shortlist_size": int(len(node_shortlist)),
                    "edge_hub_shortlist_size": int(len(edge_hub_shortlist)),
                    "edge_hub_rescue_meta": edge_hub_meta,
                    "effective_shortlist_size": int(len(shortlist)),
                    "hierarchy_enabled": bool(args.hierarchical_retrieval),
                    "hierarchy_query_group": int(
                        hierarchy_query.get("group", -1)
                    ) if hierarchy_query else -1,
                    "hierarchy_score": float(hierarchy_score[idx])
                    if args.hierarchical_retrieval else 0.0,
                    "hierarchy_hyper_score": float(
                        hierarchy_components.get(
                            "hierarchy_hyper_score", np.zeros_like(style)
                        )[idx]
                    ) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_coarse_score": float(
                        hierarchy_components.get(
                            "hierarchy_coarse_score", np.zeros_like(style)
                        )[idx]
                    ) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_activity_score": float(
                        hierarchy_components.get(
                            "hierarchy_activity_score", np.zeros_like(style)
                        )[idx]
                    ) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_turn_score": float(
                        hierarchy_components.get(
                            "hierarchy_turn_score", np.zeros_like(style)
                        )[idx]
                    ) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_semantic_score": float(
                        hierarchy_components.get(
                            "hierarchy_semantic_score", np.zeros_like(style)
                        )[idx]
                    ) if args.hierarchical_retrieval else 0.0,
                    "transition_cost": float(transition_cost),
                    "boundary_velocity_penalty": float(
                        boundary_velocity_penalty
                    ),
                    "boundary_acceleration_penalty": float(
                        boundary_acceleration_penalty
                    ),
                    "graph_scheduler_enabled": bool(args.graph_scheduler),
                    "graph_edge_cost": float(graph_edge_cost),
                    "graph_edge_meta": graph_edge_meta,
                    "boundary_compat_enabled": bool(compat_enabled),
                    "boundary_compat_hard_prune": bool(compat_hard_prune),
                    "boundary_compat_score": float(boundary_compat_score),
                    "boundary_compat_meta": boundary_compat_meta,
                    "semantic_edge_weight": float(semantic_edge_weight),
                    "semantic_edge_score": float(semantic_edge_score),
                    "semantic_edge_meta": semantic_edge_meta,
                    "semantic_edge_hard_prune": bool(
                        _enabled("V34_SEMANTIC_EDGE_HARD_PRUNE", "1")
                    ),
                    "gpu_boundary_cache": bool(gpu_boundary_cache_hit),
                    "mmr_penalty": float(mmr),
                    "score": float(score),
                }
                state_out = scheduler.CandidateState(
                    score=score,
                    selected=state.selected + [idx],
                    transition_lengths=state.transition_lengths
                    + [transition_len],
                    parts=state.parts + [part],
                )
                if constraint_relaxed:
                    relaxed_expanded.append(state_out)
                else:
                    strict_expanded.append(state_out)

        if strict_expanded:
            expanded = strict_expanded
        elif relaxed_expanded and relax_constraints_on_empty:
            relaxed_pool_size = len(relaxed_expanded)
            relaxed_expanded.sort(
                key=lambda state: (
                    float(state.parts[-1].get("constraint_relaxation_penalty", 0.0))
                    if state.parts else 0.0,
                    -float(state.score),
                )
            )
            expanded = relaxed_expanded[: min(relax_rescue_top_k, relaxed_pool_size)]
            print(
                "[V34-RELAX] "
                f"slot={slot} strict feasible set empty; "
                f"using {len(expanded)}/{relaxed_pool_size} relaxed candidates "
                f"after minimum-violation rescue top-k={relax_rescue_top_k} "
                f"(compat_deferred={compat_deferred}, "
                f"semantic_deferred={semantic_deferred}, "
                f"contact_deferred={contact_deferred})."
            )
            for rescue_rank, relaxed_state in enumerate(expanded):
                if not relaxed_state.parts:
                    continue
                relaxed_part = relaxed_state.parts[-1]
                relaxed_part["constraint_relaxation_used"] = True
                relaxed_part["constraint_relaxation_rescue_rank"] = int(rescue_rank)
                relaxed_part["constraint_relaxation_rescue_pool_size"] = int(relaxed_pool_size)
                relaxed_part["constraint_relaxation_rescue_top_k"] = int(relax_rescue_top_k)
                relaxed_transition_meta = dict(
                    relaxed_part.get("transition_meta", {})
                )
                relax_meta = dict(
                    relaxed_transition_meta.get("constraint_relaxation", {})
                )
                relax_meta["used_due_to_empty_strict"] = True
                relax_meta["rescue_rank"] = int(rescue_rank)
                relax_meta["rescue_pool_size_before_top_k"] = int(relaxed_pool_size)
                relax_meta["rescue_top_k"] = int(relax_rescue_top_k)
                relaxed_transition_meta["constraint_relaxation"] = relax_meta
                relaxed_transition_meta["semantic_relaxed"] = bool(
                    relaxed_part.get("semantic_relaxed", False)
                )
                relaxed_transition_meta["compat_relaxed"] = bool(
                    relaxed_part.get("compat_relaxed", False)
                )
                relaxed_transition_meta["contact_relaxed"] = bool(
                    relaxed_part.get("contact_relaxed", False)
                )
                relaxed_part["transition_meta"] = relaxed_transition_meta

        if not expanded:
            raise RuntimeError(
                f"V34 found no warp-feasible candidate for slot={slot}, "
                f"music_length={phrase.length}, bounds=[{minimum},{maximum}], "
                f"warp_rejected={rejected_warp}, "
                f"compat_rejected={compat_rejected}, "
                f"semantic_rejected={semantic_rejected}, "
                f"compat_deferred={compat_deferred}, "
                f"semantic_deferred={semantic_deferred}, "
                f"contact_deferred={contact_deferred}, "
                f"globally_feasible={len(feasible_indices)}, "
                f"negotiated={negotiated_count}, "
                f"warp_relaxed={warp_relaxed}, "
                f"constraint_relax_on_empty={relax_constraints_on_empty}. "
                "Even after adaptive semantic/contact relaxation, the graph is "
                "deadlocked by non-relaxable constraints such as duplicate, "
                "family, graph, or shortlist limits."
            )
        expanded.sort(key=lambda state: state.score, reverse=True)
        beam = expanded[: int(args.beam_size)]
    return beam[0]
