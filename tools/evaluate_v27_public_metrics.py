#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 public-style metrics using SO(3) and FK kinematics.

The historical filename is retained for experiment compatibility.  BAS now
uses joint-space kinetic energy rather than raw 6D coordinate differences.
The transparent FGD-style descriptor is built from SO(3) angular dynamics,
joint-space dynamics, root height and contacts, avoiding representation
artifacts caused by invalid or differently parameterized 6D rotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import load_motion
from tools.v26_music_phrase_segmentation import whole_song_features
from tools.v29_motion_geometry import (
    CONTACT,
    ROOT,
    ROOT_Y,
    angular_velocity_np,
    motion_to_joint_positions_np,
)


def _load_generated(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    if x.ndim == 3:
        x = x[0]
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    return x


def _local_peaks(
    values: np.ndarray,
    percentile: float,
    gap: int,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(x) < 3:
        return np.zeros((0,), dtype=np.int32)
    threshold = float(np.percentile(x, percentile))
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] >= threshold and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            if not peaks or i - peaks[-1] >= gap:
                peaks.append(i)
    return np.asarray(peaks, dtype=np.int32)


def beat_alignment_score(
    motion: np.ndarray,
    audio: str | Path,
    fps: float,
    feature_dir: str,
) -> Dict[str, float]:
    features, _ = whole_song_features(
        audio, fps=fps, cache_dir=feature_dir or None
    )
    music_accent = (
        0.55 * features[:, 1]
        + 0.35 * features[:, 2]
        + 0.10 * features[:, 0]
    )
    music_peaks = _local_peaks(
        music_accent,
        percentile=72.0,
        gap=max(4, int(round(0.18 * fps))),
    )

    joints = motion_to_joint_positions_np(motion)
    joint_velocity = np.diff(joints, axis=0, prepend=joints[:1]) * fps
    # Remove whole-body translation and use robust mean joint kinetic energy.
    root_velocity = joint_velocity[:, :1]
    local_velocity = joint_velocity - root_velocity
    kinetic = np.mean(np.linalg.norm(local_velocity, axis=-1), axis=-1)
    motion_peaks = _local_peaks(
        kinetic,
        percentile=78.0,
        gap=max(6, int(round(0.22 * fps))),
    )
    if len(motion_peaks) == 0 or len(music_peaks) == 0:
        return {
            "bas": 0.0,
            "hit_6f": 0.0,
            "hit_10f": 0.0,
            "hit_15f": 0.0,
            "motion_peaks": float(len(motion_peaks)),
            "music_peaks": float(len(music_peaks)),
        }
    distances = np.asarray(
        [np.min(np.abs(music_peaks - p)) for p in motion_peaks],
        dtype=np.float32,
    )
    sigma = max(1.0, 0.12 * fps)
    return {
        "bas": float(np.mean(np.exp(-(distances**2) / (2.0 * sigma**2)))),
        "hit_6f": float(np.mean(distances <= 6)),
        "hit_10f": float(np.mean(distances <= 10)),
        "hit_15f": float(np.mean(distances <= 15)),
        "nearest_median_frames": float(np.median(distances)),
        "nearest_p90_frames": float(np.percentile(distances, 90)),
        "motion_peaks": float(len(motion_peaks)),
        "music_peaks": float(len(music_peaks)),
    }


def motion_window_features(
    motion: np.ndarray,
    window: int = 60,
    stride: int = 30,
    fps: float = 30.0,
) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if len(x) < 4:
        return np.zeros((0, 18), dtype=np.float32)
    joint_positions = motion_to_joint_positions_np(x)
    angular = angular_velocity_np(x) * fps
    features: List[np.ndarray] = []
    for start in range(0, max(1, len(x) - window + 1), max(1, stride)):
        end = min(start + window, len(x))
        segment = x[start:end]
        joints = joint_positions[start:end]
        if len(segment) < 4:
            continue
        joint_velocity = np.diff(joints, axis=0) * fps
        joint_acceleration = np.diff(joint_velocity, axis=0) * fps
        joint_jerk = np.diff(joint_acceleration, axis=0) * fps
        local_joint_velocity = joint_velocity - joint_velocity[:, :1]
        joint_speed = np.linalg.norm(local_joint_velocity, axis=-1)
        acceleration = np.linalg.norm(joint_acceleration, axis=-1)
        jerk = np.linalg.norm(joint_jerk, axis=-1) if len(joint_jerk) else np.zeros((1, 24))
        local_angular = angular[start : max(start, end - 1)]
        angular_speed = np.linalg.norm(local_angular, axis=-1)
        contact_switch = (
            np.abs(np.diff(segment[:, CONTACT], axis=0)).mean()
            if len(segment) > 1 else 0.0
        )
        root_y = segment[:, ROOT_Y]
        feature = np.asarray(
            [
                joint_speed.mean(),
                joint_speed.std(),
                np.percentile(joint_speed, 95),
                acceleration.mean(),
                acceleration.std(),
                np.percentile(acceleration, 95),
                jerk.mean(),
                np.percentile(jerk, 95),
                angular_speed.mean() if angular_speed.size else 0.0,
                angular_speed.std() if angular_speed.size else 0.0,
                np.percentile(angular_speed, 95) if angular_speed.size else 0.0,
                np.std(joints[:, :, 0]),
                np.std(joints[:, :, 1]),
                np.std(joints[:, :, 2]),
                float(root_y.mean()),
                float(root_y.std()),
                float(contact_switch),
                float(len(segment) / fps),
            ],
            dtype=np.float32,
        )
        features.append(feature)
    return (
        np.stack(features).astype(np.float32)
        if features else np.zeros((0, 18), dtype=np.float32)
    )


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)[None, :]) @ vectors.T


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False) + np.eye(a.shape[1]) * 1e-6
    cov_b = np.cov(b, rowvar=False) + np.eye(b.shape[1]) * 1e-6
    cov_mean = _sqrtm_psd(
        _sqrtm_psd(cov_a) @ cov_b @ _sqrtm_psd(cov_a)
    )
    value = float(
        np.sum((mean_a - mean_b) ** 2)
        + np.trace(cov_a + cov_b - 2.0 * cov_mean)
    )
    return max(value, 0.0)


def load_real_feature_bank(
    index_json: str | Path,
    index_npz: str | Path,
    fps: float,
    max_events: int,
) -> np.ndarray:
    _, _, items = load_shared_index(Path(index_json), Path(index_npz))
    rows = []
    for item in items[: max_events if max_events > 0 else len(items)]:
        try:
            motion = load_motion(
                Path(str(item.get("pkl", item.get("path", ""))))
            )
        except Exception:
            continue
        feature = motion_window_features(
            motion,
            window=min(60, max(8, len(motion))),
            stride=30,
            fps=fps,
        )
        if len(feature):
            rows.append(feature)
    return (
        np.concatenate(rows, axis=0).astype(np.float32)
        if rows else np.zeros((0, 18), dtype=np.float32)
    )


def diversity_score(features: np.ndarray, max_pairs: int = 5000) -> float:
    if len(features) < 2:
        return 0.0
    rng = np.random.default_rng(20260610)
    n = len(features)
    pairs = rng.integers(0, n, size=(min(max_pairs, n * (n - 1)), 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return (
        float(np.mean(np.linalg.norm(
            features[pairs[:, 0]] - features[pairs[:, 1]], axis=1
        )))
        if len(pairs) else 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--feature_dir", default="data/v29_public_metric_features"
    )
    parser.add_argument("--real_max_events", type=int, default=1200)
    args = parser.parse_args()

    motion = _load_generated(Path(args.motion))
    generated = motion_window_features(motion, fps=args.fps)
    real = load_real_feature_bank(
        args.index_json,
        args.duration_index_npz,
        fps=args.fps,
        max_events=args.real_max_events,
    )
    bas = beat_alignment_score(
        motion, args.audio, args.fps, args.feature_dir
    )
    result: Dict[str, Any] = {
        "version": "v29_so3_fk_public_style_metrics",
        "motion": str(args.motion),
        "audio": str(args.audio),
        "num_generated_windows": int(len(generated)),
        "num_real_windows": int(len(real)),
        "bas": bas,
        "fgd_kinematic": frechet_distance(generated, real),
        "generated_diversity": diversity_score(generated),
        "real_diversity": diversity_score(real),
        "note": (
            "Transparent SO(3)+FK kinematic FGD-style metric. "
            "It is suitable for internal ablation under one implementation, "
            "but should not be claimed as a standard learned-encoder FGD."
        ),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[V29 public metrics]",
        Path(args.motion).stem,
        "BAS=", round(float(bas["bas"]), 4),
        "FGDkin=", round(float(result["fgd_kinematic"]), 4),
    )


if __name__ == "__main__":
    main()
