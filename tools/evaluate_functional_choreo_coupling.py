#!/usr/bin/env python3
"""Evaluate functional choreography coupling for one or more motions.

This replacement is backward-compatible with the previous script, and adds
optional target-trajectory event metrics when --trajectory is provided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from functional_choreo_metrics import functional_choreo_stats
from turn_aware_event_utils import TurnEventConfig, event_feature_matrix, parse_int_list


def load_motion(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def rms_delta(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros((len(x),), dtype=np.float32)
    out = np.zeros((len(x),), dtype=np.float32)
    out[1:] = np.sqrt(np.mean((x[1:] - x[:-1]) ** 2, axis=-1))
    return out.astype(np.float32)


def window_response(signal: np.ndarray, gate: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)[: len(signal)]
    if len(signal) == 0 or float(gate.max()) <= 1e-8:
        return 0.0
    base = float(signal.mean()) + 1e-8
    val = float((signal * gate).sum() / (gate.sum() + 1e-8))
    return float(np.clip(val / base - 1.0, 0.0, 1.0))


def add_target_event_metrics(stats: Dict[str, float], motion: np.ndarray, trajectory: str) -> Dict[str, float]:
    ev, names, report = event_feature_matrix(trajectory, TurnEventConfig.from_env(seq_len=len(motion), count=5))
    # Conservative fallback body groups; functional_choreo_metrics already computes global metrics.
    try:
        from footstep_phase_utils import LOWER_ROT_INDEX, TORSO_ROT_INDEX, UPPER_ROT_INDEX, CONTACT_SLICE
    except Exception:
        rot = np.arange(7, 151).reshape(24, 6)
        LOWER_ROT_INDEX = rot[[1, 2, 4, 5, 7, 8, 10, 11]].reshape(-1)
        TORSO_ROT_INDEX = rot[[0, 3, 6, 9, 12]].reshape(-1)
        UPPER_ROT_INDEX = rot[[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].reshape(-1)
        CONTACT_SLICE = slice(0, 4)

    lower_e = rms_delta(motion[:, LOWER_ROT_INDEX])
    torso_e = rms_delta(motion[:, TORSO_ROT_INDEX])
    upper_e = rms_delta(motion[:, UPPER_ROT_INDEX])
    expr_e = 0.5 * torso_e + 0.5 * upper_e
    contacts = (motion[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_delta = np.zeros((len(motion),), dtype=np.float32)
    if len(motion) > 1:
        contact_delta[1:] = np.abs(contacts[1:] - contacts[:-1]).sum(axis=1)

    # event_feature_matrix columns: 6 turn, 7 support, 8 expressive, 10 speed.
    turn_gate = ev[:, 6]
    support_gate = ev[:, 7]
    expressive_gate = ev[:, 8]
    speed_gate = ev[:, 10]

    stats["target_turn_expression_response"] = window_response(expr_e, turn_gate)
    stats["target_support_lower_response"] = window_response(lower_e + contact_delta, support_gate)
    stats["target_expressive_response"] = window_response(expr_e, expressive_gate)
    stats["target_speed_lower_response"] = window_response(lower_e, speed_gate)
    stats["target_event_centers"] = report["event_centers"]  # type: ignore
    stats["target_support_frames"] = report["support_frames"]  # type: ignore
    stats["target_expressive_frames"] = report["expressive_frames"]  # type: ignore
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", required=True, help="Comma-separated .npy motions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectory", default="")
    args = ap.parse_args()

    paths = [x.strip() for x in args.motions.split(",") if x.strip()]
    result = {}
    for path in paths:
        motion = load_motion(path)
        stats = functional_choreo_stats(motion)
        if args.trajectory:
            stats = add_target_event_metrics(stats, motion, args.trajectory)
        result[path] = stats
        print("\n" + "=" * 100)
        print(path)
        for k in [
            "root_path",
            "root_max_step",
            "lower_activity",
            "torso_activity",
            "upper_activity",
            "contact_switch",
            "support_expression_coupling",
            "turn_expression_response",
            "target_turn_expression_response",
            "target_support_lower_response",
            "target_expressive_response",
            "speed_lower_sync",
            "speed_torso_sync",
            "speed_expression_sync",
            "lower_torso_sync",
            "lower_upper_sync",
        ]:
            if k in stats:
                print(f"{k}: {stats[k]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ saved: {out}")


if __name__ == "__main__":
    main()
