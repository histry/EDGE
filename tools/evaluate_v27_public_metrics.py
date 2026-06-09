#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public-style metrics for V27/V28 comparison.

This does not replace domain metrics.  It adds:
- BAS: beat/accent alignment score.
- FGD-style distance: Frechet distance between generated and real motion
  feature distributions using transparent kinematic descriptors.
- Diversity: average pairwise distance among generated motion windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import ROT, load_motion
from tools.v26_music_phrase_segmentation import whole_song_features


def _load_generated(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    if x.ndim == 3:
        x = x[0]
    return np.asarray(x, dtype=np.float32)


def _local_peaks(x: np.ndarray, percentile: float = 75.0, gap: int = 8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < 3:
        return np.zeros((0,), dtype=np.int32)
    th = float(np.percentile(x, percentile))
    peaks: List[int] = []
    for i in range(1, len(x) - 1):
        if x[i] >= th and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            if not peaks or i - peaks[-1] >= gap:
                peaks.append(i)
    return np.asarray(peaks, dtype=np.int32)


def beat_alignment_score(motion: np.ndarray, audio: str | Path, fps: float, feature_dir: str) -> Dict[str, float]:
    features, _ = whole_song_features(audio, fps=fps, cache_dir=feature_dir or None)
    music_accent = 0.55 * features[:, 1] + 0.35 * features[:, 2] + 0.10 * features[:, 0]
    music_peaks = _local_peaks(music_accent, percentile=72.0, gap=max(4, int(round(0.18 * fps))))
    vel = np.diff(motion[:, ROT], axis=0, prepend=motion[:1, ROT]) * float(fps)
    kinetic = np.linalg.norm(vel, axis=1)
    motion_peaks = _local_peaks(kinetic, percentile=78.0, gap=max(6, int(round(0.22 * fps))))
    if len(motion_peaks) == 0 or len(music_peaks) == 0:
        return {"bas": 0.0, "hit_6f": 0.0, "hit_10f": 0.0, "hit_15f": 0.0, "motion_peaks": float(len(motion_peaks)), "music_peaks": float(len(music_peaks))}
    dists = np.asarray([np.min(np.abs(music_peaks - p)) for p in motion_peaks], dtype=np.float32)
    # Gaussian BAS variant; high when motion accents are close to music accents.
    sigma = max(1.0, 0.12 * float(fps))
    bas = float(np.mean(np.exp(-(dists ** 2) / (2.0 * sigma * sigma))))
    return {
        "bas": bas,
        "hit_6f": float(np.mean(dists <= 6)),
        "hit_10f": float(np.mean(dists <= 10)),
        "hit_15f": float(np.mean(dists <= 15)),
        "nearest_median_frames": float(np.median(dists)),
        "nearest_p90_frames": float(np.percentile(dists, 90)),
        "motion_peaks": float(len(motion_peaks)),
        "music_peaks": float(len(music_peaks)),
    }


def motion_window_features(motion: np.ndarray, window: int = 60, stride: int = 30, fps: float = 30.0) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if len(x) < 3:
        return np.zeros((0, 12), dtype=np.float32)
    feats: List[np.ndarray] = []
    for start in range(0, max(1, len(x) - window + 1), max(1, stride)):
        seg = x[start : start + window]
        if len(seg) < 3:
            continue
        rot = seg[:, ROT]
        vel = np.diff(rot, axis=0) * float(fps)
        acc = np.diff(vel, axis=0) * float(fps)
        feat = np.asarray(
            [
                np.mean(np.linalg.norm(vel, axis=1)),
                np.std(np.linalg.norm(vel, axis=1)),
                np.percentile(np.linalg.norm(vel, axis=1), 95),
                np.mean(np.linalg.norm(acc, axis=1)) if len(acc) else 0.0,
                np.std(np.linalg.norm(acc, axis=1)) if len(acc) else 0.0,
                np.mean(np.std(rot, axis=0)),
                np.percentile(np.std(rot, axis=0), 90),
                np.mean(np.abs(rot)),
                np.percentile(np.abs(rot), 95),
                len(seg) / float(fps),
                np.mean(np.linalg.norm(np.diff(seg[:, :7], axis=0), axis=1)) if seg.shape[1] >= 7 else 0.0,
                np.std(np.linalg.norm(np.diff(seg[:, :7], axis=0), axis=1)) if seg.shape[1] >= 7 else 0.0,
            ],
            dtype=np.float32,
        )
        feats.append(feat)
    return np.stack(feats).astype(np.float32) if feats else np.zeros((0, 12), dtype=np.float32)


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) * 0.5)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)[None, :]) @ vecs.T


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mu1, mu2 = a.mean(axis=0), b.mean(axis=0)
    c1 = np.cov(a, rowvar=False) + np.eye(a.shape[1]) * 1e-6
    c2 = np.cov(b, rowvar=False) + np.eye(b.shape[1]) * 1e-6
    covmean = _sqrtm_psd(_sqrtm_psd(c1) @ c2 @ _sqrtm_psd(c1))
    value = float(np.sum((mu1 - mu2) ** 2) + np.trace(c1 + c2 - 2.0 * covmean))
    return max(value, 0.0)


def load_real_feature_bank(index_json: str | Path, index_npz: str | Path, fps: float, max_events: int) -> np.ndarray:
    _, _, items = load_shared_index(Path(index_json), Path(index_npz))
    feats: List[np.ndarray] = []
    for item in items[: max_events if max_events > 0 else len(items)]:
        try:
            motion = load_motion(Path(str(item.get("pkl", item.get("path", "")))))
        except Exception:
            continue
        f = motion_window_features(motion, window=min(60, max(8, len(motion))), stride=30, fps=fps)
        if len(f):
            feats.append(f)
    return np.concatenate(feats, axis=0).astype(np.float32) if feats else np.zeros((0, 12), dtype=np.float32)


def diversity_score(features: np.ndarray, max_pairs: int = 5000) -> float:
    if len(features) < 2:
        return 0.0
    rng = np.random.default_rng(20260610)
    n = len(features)
    pairs = rng.integers(0, n, size=(min(max_pairs, n * (n - 1)), 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs) == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(features[pairs[:, 0]] - features[pairs[:, 1]], axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--feature_dir", default="data/v27_public_metric_features")
    parser.add_argument("--real_max_events", type=int, default=1200)
    args = parser.parse_args()

    motion = _load_generated(Path(args.motion))
    gen_feat = motion_window_features(motion, fps=args.fps)
    real_feat = load_real_feature_bank(args.index_json, args.duration_index_npz, fps=args.fps, max_events=args.real_max_events)
    bas = beat_alignment_score(motion, args.audio, args.fps, args.feature_dir)
    result: Dict[str, Any] = {
        "motion": str(args.motion),
        "audio": str(args.audio),
        "num_generated_windows": int(len(gen_feat)),
        "num_real_windows": int(len(real_feat)),
        "bas": bas,
        "fgd_kinematic": frechet_distance(gen_feat, real_feat),
        "generated_diversity": diversity_score(gen_feat),
        "real_diversity": diversity_score(real_feat),
        "note": "FGD-style metric uses transparent kinematic descriptors; replace feature extractor with a learned motion encoder when a common benchmark encoder is available.",
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[V27 public metrics]",
        Path(args.motion).stem,
        "BAS=",
        round(float(bas["bas"]), 4),
        "FGDkin=",
        round(float(result["fgd_kinematic"]), 4),
    )


if __name__ == "__main__":
    main()
