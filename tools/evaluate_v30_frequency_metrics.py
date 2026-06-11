#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correct V31 frequency, distribution and foot-dynamics evaluator.

This replaces the V30 evaluator whose helper `_spectrum_power` recursively
called itself. Metrics are transparent internal descriptors; they are not
presented as universal learned-encoder benchmarks.
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


def load_generated(path: str | Path) -> np.ndarray:
    motion = np.asarray(np.load(path, allow_pickle=True), np.float32)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim != 2 or motion.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {motion.shape}")
    return motion


def spectrum_power(
    motion: np.ndarray, fps: float
) -> Tuple[np.ndarray, np.ndarray]:
    positions = motion_to_joint_positions_np(motion)
    local = positions - positions[:, :1]
    velocity = np.diff(local, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    signal = (
        jerk.reshape(len(jerk), -1)
        if len(jerk)
        else acceleration.reshape(len(acceleration), -1)
    )
    if len(signal) < 2:
        return np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    coefficient = dct(signal, axis=0, norm="ortho")
    power = np.mean(coefficient**2, axis=1).astype(np.float32)
    frequency = (
        np.arange(len(power), dtype=np.float32)
        * fps
        / max(2 * (len(power) - 1), 1)
    )
    return frequency, power


def band_metrics(motion: np.ndarray, fps: float) -> Dict[str, float]:
    frequency, power = spectrum_power(motion, fps)
    if not len(power):
        return {
            "low_below_2hz_ratio": 0.0,
            "mid_2_to_6hz_ratio": 0.0,
            "high_above_6hz_ratio": 0.0,
            "spectral_entropy": 0.0,
            "dct_total_energy": 0.0,
            "dct_high_low_ratio": 0.0,
        }
    total = float(power.sum()) + 1e-12
    low = float(power[frequency < 2.0].sum() / total)
    middle = float(
        power[(frequency >= 2.0) & (frequency < 6.0)].sum() / total
    )
    high = float(power[frequency >= 6.0].sum() / total)
    probability = power / total
    entropy = float(
        -(probability * np.log(probability + 1e-12)).sum()
        / np.log(max(len(probability), 2))
    )
    return {
        "low_below_2hz_ratio": low,
        "mid_2_to_6hz_ratio": middle,
        "high_above_6hz_ratio": high,
        "spectral_entropy": entropy,
        "dct_total_energy": total,
        "dct_high_low_ratio": high / max(low, 1e-8),
    }


def transition_segments(
    motion: np.ndarray, report_path: str
) -> List[np.ndarray]:
    if not report_path or not Path(report_path).is_file():
        return []
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    allocation = report.get("allocation", {})
    boundaries = [int(x) for x in allocation.get("output_boundaries", [])]
    lengths = [int(x) for x in allocation.get("transition_lengths", [])]
    segments = []
    for slot in range(1, min(len(boundaries), len(lengths))):
        start = boundaries[slot]
        end = min(len(motion), start + lengths[slot])
        if end - start >= 8:
            segments.append(motion[start:end])
    return segments


def aggregate_spectrum(
    segments: List[np.ndarray], fps: float
) -> Dict[str, float]:
    rows = [band_metrics(segment, fps) for segment in segments]
    if not rows:
        return {"num_transitions": 0, **band_metrics(
            np.zeros((2, 151), np.float32), fps
        )}
    return {
        "num_transitions": len(rows),
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        },
    }


def foot_descriptor(
    motion: np.ndarray, fps: float
) -> Tuple[np.ndarray, Dict[str, float]]:
    positions = motion_to_joint_positions_np(motion)
    feet = positions[:, FOOT_JOINTS]
    velocity = np.diff(feet, axis=0, prepend=feet[:1]) * fps
    speed = np.linalg.norm(velocity, axis=-1)
    contacts = np.asarray(motion[:, CONTACT] > 0.5)
    slide = speed[contacts]
    descriptor = np.asarray([
        speed.mean(),
        speed.std(),
        np.percentile(speed, 95),
        slide.mean() if slide.size else 0.0,
        np.percentile(slide, 95) if slide.size else 0.0,
        np.abs(np.diff(contacts.astype(np.float32), axis=0)).mean()
        if len(contacts) > 1 else 0.0,
        feet[..., 1].mean(),
        feet[..., 1].std(),
    ], np.float32)
    return descriptor, {
        "contact_slide_mean": float(slide.mean()) if slide.size else 0.0,
        "contact_slide_p95": (
            float(np.percentile(slide, 95)) if slide.size else 0.0
        ),
        "contact_switch_rate": (
            float(np.abs(np.diff(
                contacts.astype(np.float32), axis=0
            )).mean())
            if len(contacts) > 1 else 0.0
        ),
    }


def window_features(
    motion: np.ndarray,
    fps: float,
    window: int = 60,
    stride: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    motion_rows, foot_rows = [], []
    for start in range(0, max(1, len(motion) - window + 1), stride):
        segment = motion[start : start + window]
        if len(segment) < 12:
            continue
        positions = motion_to_joint_positions_np(segment)
        local = positions - positions[:, :1]
        velocity = np.diff(local, axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        jerk = np.diff(acceleration, axis=0) * fps
        spectrum = band_metrics(segment, fps)
        foot, _ = foot_descriptor(segment, fps)
        motion_rows.append(np.concatenate([
            np.asarray([
                np.linalg.norm(velocity, axis=-1).mean(),
                np.linalg.norm(velocity, axis=-1).std(),
                np.linalg.norm(acceleration, axis=-1).mean(),
                np.percentile(
                    np.linalg.norm(acceleration, axis=-1), 95
                ),
                np.linalg.norm(jerk, axis=-1).mean()
                if len(jerk) else 0.0,
                np.percentile(
                    np.linalg.norm(jerk, axis=-1), 95
                ) if len(jerk) else 0.0,
                spectrum["low_below_2hz_ratio"],
                spectrum["high_above_6hz_ratio"],
            ], np.float32),
            foot,
        ]))
        foot_rows.append(foot)
    return (
        np.stack(motion_rows).astype(np.float32)
        if motion_rows else np.zeros((0, 16), np.float32),
        np.stack(foot_rows).astype(np.float32)
        if foot_rows else np.zeros((0, 8), np.float32),
    )


def rbf_mmd(
    first: np.ndarray,
    second: np.ndarray,
    scales=(0.5, 1.0, 2.0, 4.0),
) -> float:
    if len(first) < 2 or len(second) < 2:
        return 0.0
    combined = np.concatenate([first, second])
    standard = np.std(combined, axis=0, keepdims=True) + 1e-6
    first = first / standard
    second = second / standard
    aa = np.sum((first[:, None] - first[None]) ** 2, axis=-1)
    bb = np.sum((second[:, None] - second[None]) ** 2, axis=-1)
    ab = np.sum((first[:, None] - second[None]) ** 2, axis=-1)
    values = []
    for sigma in scales:
        denominator = 2.0 * float(sigma) ** 2
        values.append(
            np.exp(-aa / denominator).mean()
            + np.exp(-bb / denominator).mean()
            - 2.0 * np.exp(-ab / denominator).mean()
        )
    return float(max(np.mean(values), 0.0))


def sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    value, vector = np.linalg.eigh((matrix + matrix.T) * 0.5)
    value = np.clip(value, 0.0, None)
    return (vector * np.sqrt(value)[None]) @ vector.T


def frechet(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(second) < 2:
        return 0.0
    mean_a, mean_b = first.mean(0), second.mean(0)
    cov_a = np.cov(first, rowvar=False) + np.eye(first.shape[1]) * 1e-6
    cov_b = np.cov(second, rowvar=False) + np.eye(second.shape[1]) * 1e-6
    middle = sqrt_psd(sqrt_psd(cov_a) @ cov_b @ sqrt_psd(cov_a))
    return float(max(
        np.sum((mean_a - mean_b) ** 2)
        + np.trace(cov_a + cov_b - 2.0 * middle),
        0.0,
    ))


def real_bank(
    index_json: str,
    index_npz: str,
    fps: float,
    maximum: int,
) -> Tuple[np.ndarray, np.ndarray]:
    _, _, items = load_shared_index(Path(index_json), Path(index_npz))
    motion_rows, foot_rows = [], []
    for item in items[:maximum]:
        try:
            motion = load_motion(
                Path(str(item.get("pkl", item.get("path", ""))))
            )
        except Exception:
            continue
        motion_feature, foot_feature = window_features(
            motion, fps, min(60, len(motion)), 30
        )
        if len(motion_feature):
            motion_rows.append(motion_feature)
        if len(foot_feature):
            foot_rows.append(foot_feature)
    return (
        np.concatenate(motion_rows)
        if motion_rows else np.zeros((0, 16), np.float32),
        np.concatenate(foot_rows)
        if foot_rows else np.zeros((0, 8), np.float32),
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

    motion = load_generated(args.motion)
    generated, generated_foot = window_features(motion, args.fps)
    real, real_foot = real_bank(
        args.index_json,
        args.duration_index_npz,
        args.fps,
        args.real_max_events,
    )
    segments = transition_segments(motion, args.schedule_report)
    _, foot_summary = foot_descriptor(motion, args.fps)
    result = {
        "version": "v31_corrected_frequency_distribution_foot_metrics",
        "motion": args.motion,
        "whole_song_spectrum": band_metrics(motion, args.fps),
        "transition_spectrum": aggregate_spectrum(segments, args.fps),
        "foot": foot_summary,
        "multi_scale_rbf_mmd": rbf_mmd(generated, real),
        "foot_motion_frechet": frechet(generated_foot, real_foot),
        "generated_windows": int(len(generated)),
        "real_windows": int(len(real)),
        "note": (
            "Transparent descriptor metrics; not a substitute for a named "
            "community learned-encoder benchmark."
        ),
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.out_png:
        import matplotlib.pyplot as plt
        frequency, power = spectrum_power(motion, args.fps)
        plt.figure(figsize=(8, 4.5))
        plt.semilogy(frequency, power + 1e-12)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mean DCT jerk power")
        plt.title("V31 whole-song motion spectrum")
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=180)
        plt.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
