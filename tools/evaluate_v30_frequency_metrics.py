#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frequency, distribution and foot-dynamics evaluation for EDGE V30.

Names are intentionally transparent:
  * multi_scale_mmd is an RBF-kernel MMD over explicit kinematic descriptors;
  * foot_motion_frechet is a Fréchet distance over explicit foot descriptors;
  * neither is presented as a standard learned-encoder benchmark metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.fft import dct

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import load_motion
from tools.v29_motion_geometry import (
    CONTACT,
    motion_to_joint_positions_np,
)


FOOT_JOINTS = (7, 8, 10, 11)


def _load(path: str | Path) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    return x


def _bands(power: np.ndarray, fps: float) -> Dict[str, float]:
    frequency = np.arange(len(power), dtype=np.float32) * fps / max(2 * (len(power) - 1), 1)
    total = float(power.sum()) + 1e-10
    low = float(power[frequency < 2.0].sum() / total)
    mid = float(power[(frequency >= 2.0) & (frequency < 6.0)].sum() / total)
    high = float(power[frequency >= 6.0].sum() / total)
    probability = power / total
    entropy = float(
        -(probability * np.log(probability + 1e-10)).sum()
        / np.log(max(len(probability), 2))
    )
    return {
        "low_below_2hz_ratio": low,
        "mid_2_to_6hz_ratio": mid,
        "high_above_6hz_ratio": high,
        "spectral_entropy": entropy,
    }


def _spectrum_power(motion: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    _frequency, power = _spectrum_power(motion, fps)
    frequency = np.arange(len(power), dtype=np.float32) * fps / max(2 * (len(power) - 1), 1)
    return frequency, power


def spectrum_metrics(motion: np.ndarray, fps: float) -> Dict[str, float]:
    positions = motion_to_joint_positions_np(motion)
    local = positions - positions[:, :1]
    velocity = np.diff(local, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    signal = jerk.reshape(len(jerk), -1) if len(jerk) else acceleration.reshape(len(acceleration), -1)
    coefficients = dct(signal, axis=0, norm="ortho")
    power = np.mean(coefficients**2, axis=1)
    result = _bands(power, fps)
    result["dct_total_energy"] = float(power.sum())
    result["dct_high_low_ratio"] = float(
        result["high_above_6hz_ratio"] / max(result["low_below_2hz_ratio"], 1e-8)
    )
    return result



def transition_segments(motion: np.ndarray, report_path: str) -> List[np.ndarray]:
    if not report_path or not Path(report_path).is_file():
        return []
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    allocation = report.get("allocation", {})
    boundaries = [int(x) for x in allocation.get("output_boundaries", [])]
    lengths = [int(x) for x in allocation.get("transition_lengths", [])]
    result = []
    for slot in range(1, min(len(boundaries), len(lengths))):
        start = boundaries[slot]
        end = min(len(motion), start + lengths[slot])
        if end - start >= 6:
            result.append(motion[start:end])
    return result


def aggregate_transition_spectrum(segments: List[np.ndarray], fps: float) -> Dict[str, float]:
    rows = [spectrum_metrics(segment, fps) for segment in segments if len(segment) >= 6]
    if not rows:
        return {
            "num_transitions": 0,
            "low_below_2hz_ratio": 0.0,
            "mid_2_to_6hz_ratio": 0.0,
            "high_above_6hz_ratio": 0.0,
            "spectral_entropy": 0.0,
            "dct_total_energy": 0.0,
            "dct_high_low_ratio": 0.0,
        }
    keys = rows[0].keys()
    return {
        "num_transitions": len(rows),
        **{key: float(np.mean([row[key] for row in rows])) for key in keys},
    }


def foot_descriptor(motion: np.ndarray, fps: float) -> Tuple[np.ndarray, Dict[str, float]]:
    positions = motion_to_joint_positions_np(motion)
    feet = positions[:, FOOT_JOINTS]
    velocity = np.diff(feet, axis=0, prepend=feet[:1]) * fps
    speed = np.linalg.norm(velocity, axis=-1)
    contacts = np.asarray(motion[:, CONTACT] > 0.5)
    if contacts.shape[1] != 4:
        contacts = speed < np.percentile(speed, 25)
    slide_values = speed[contacts]
    descriptor = np.asarray(
        [
            speed.mean(),
            speed.std(),
            np.percentile(speed, 95),
            float(slide_values.mean()) if slide_values.size else 0.0,
            float(np.percentile(slide_values, 95)) if slide_values.size else 0.0,
            np.abs(np.diff(contacts.astype(np.float32), axis=0)).mean(),
            feet[..., 1].mean(),
            feet[..., 1].std(),
        ],
        np.float32,
    )
    return descriptor, {
        "contact_slide_mean": float(slide_values.mean()) if slide_values.size else 0.0,
        "contact_slide_p95": float(np.percentile(slide_values, 95)) if slide_values.size else 0.0,
        "contact_switch_rate": float(
            np.abs(np.diff(contacts.astype(np.float32), axis=0)).mean()
        ) if len(contacts) > 1 else 0.0,
    }


def window_features(
    motion: np.ndarray, fps: float, window: int = 60, stride: int = 30
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for start in range(0, max(1, len(motion) - window + 1), stride):
        segment = motion[start : start + window]
        if len(segment) < 12:
            continue
        positions = motion_to_joint_positions_np(segment)
        local = positions - positions[:, :1]
        velocity = np.diff(local, axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        jerk = np.diff(acceleration, axis=0) * fps
        foot, _ = foot_descriptor(segment, fps)
        spectral = spectrum_metrics(segment, fps)
        rows.append(np.concatenate([
            np.asarray([
                np.linalg.norm(velocity, axis=-1).mean(),
                np.linalg.norm(velocity, axis=-1).std(),
                np.linalg.norm(acceleration, axis=-1).mean(),
                np.percentile(np.linalg.norm(acceleration, axis=-1), 95),
                np.linalg.norm(jerk, axis=-1).mean() if len(jerk) else 0.0,
                np.percentile(np.linalg.norm(jerk, axis=-1), 95) if len(jerk) else 0.0,
                spectral["low_below_2hz_ratio"],
                spectral["high_above_6hz_ratio"],
            ], np.float32),
            foot,
        ]))
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0, 16), np.float32)


def _rbf_mmd(a: np.ndarray, b: np.ndarray, scales=(0.5, 1.0, 2.0, 4.0)) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    combined = np.concatenate([a, b], axis=0)
    scale = np.std(combined, axis=0, keepdims=True) + 1e-6
    a = a / scale
    b = b / scale
    aa = np.sum((a[:, None] - a[None]) ** 2, axis=-1)
    bb = np.sum((b[:, None] - b[None]) ** 2, axis=-1)
    ab = np.sum((a[:, None] - b[None]) ** 2, axis=-1)
    value = 0.0
    for sigma in scales:
        denom = 2.0 * float(sigma) ** 2
        value += (
            np.exp(-aa / denom).mean()
            + np.exp(-bb / denom).mean()
            - 2.0 * np.exp(-ab / denom).mean()
        )
    return float(max(value / len(scales), 0.0))


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)[None]) @ vectors.T


def _frechet(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False) + np.eye(a.shape[1]) * 1e-6
    cb = np.cov(b, rowvar=False) + np.eye(b.shape[1]) * 1e-6
    middle = _sqrtm_psd(_sqrtm_psd(ca) @ cb @ _sqrtm_psd(ca))
    return float(max(np.sum((ma - mb) ** 2) + np.trace(ca + cb - 2 * middle), 0.0))


def real_bank(
    index_json: str, index_npz: str, fps: float, max_events: int
) -> Tuple[np.ndarray, np.ndarray]:
    _, _, items = load_shared_index(Path(index_json), Path(index_npz))
    motion_rows, foot_rows = [], []
    for item in items[:max_events]:
        try:
            motion = load_motion(Path(str(item.get("pkl", item.get("path", "")))))
        except Exception:
            continue
        features = window_features(motion, fps, min(60, len(motion)), 30)
        if len(features):
            motion_rows.append(features)
        foot_rows.append(foot_descriptor(motion, fps)[0])
    return (
        np.concatenate(motion_rows, axis=0) if motion_rows else np.zeros((0, 16), np.float32),
        np.stack(foot_rows) if foot_rows else np.zeros((0, 8), np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--schedule_report", default="")
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_png", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--real_max_events", type=int, default=1000)
    args = parser.parse_args()

    motion = _load(args.motion)
    generated = window_features(motion, args.fps)
    generated_foot = []
    for start in range(0, max(1, len(motion) - 60 + 1), 30):
        segment = motion[start : start + 60]
        if len(segment) >= 12:
            generated_foot.append(foot_descriptor(segment, args.fps)[0])
    generated_foot = np.stack(generated_foot) if generated_foot else np.zeros((0, 8), np.float32)
    real, real_foot = real_bank(
        args.index_json, args.duration_index_npz, args.fps, args.real_max_events
    )
    foot_summary = foot_descriptor(motion, args.fps)[1]
    segments = transition_segments(motion, args.schedule_report)
    result = {
        "version": "v30_frequency_distribution_foot_metrics",
        "motion": args.motion,
        "whole_song_spectrum": spectrum_metrics(motion, args.fps),
        "transition_spectrum": aggregate_transition_spectrum(segments, args.fps),
        "foot": foot_summary,
        "multi_scale_mmd": _rbf_mmd(generated, real),
        "foot_motion_frechet": _frechet(generated_foot, real_foot),
        "generated_windows": int(len(generated)),
        "real_windows": int(len(real)),
        "note": (
            "multi_scale_mmd and foot_motion_frechet use explicit transparent "
            "kinematic descriptors; they are not claimed as standard learned-encoder metrics."
        ),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_png:
        import matplotlib.pyplot as plt
        plot_motion = max(segments, key=len) if segments else motion
        frequency, power = _spectrum_power(plot_motion, args.fps)
        plt.figure(figsize=(8, 4.5))
        plt.semilogy(frequency, power + 1e-12)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mean DCT power")
        plt.title("V30 transition jerk DCT spectrum")
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=180)
        plt.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
