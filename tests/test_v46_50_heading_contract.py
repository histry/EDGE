#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v46_50_heading_contract import (
    EDGE_DIM,
    ROT6D_START,
    adaptive_event_segments,
    candidate_heading_penalty,
    canonicalize_event_entry_heading_np,
    enforce_event_heading_contract,
    heading_metrics_np,
    matrix_to_rot6d_np,
    restore_planned_root_heading_np,
    root_yaw_np,
    yaw_matrix_np,
)


def make_motion(frames: int, start_deg: float, end_deg: float) -> np.ndarray:
    x = np.zeros((frames, EDGE_DIM), dtype=np.float32)
    x[:, 0:4] = 1.0
    x[:, 5] = 1.0
    y = np.radians(np.linspace(start_deg, end_deg, frames, dtype=np.float32))
    root = yaw_matrix_np(y)
    ident = np.eye(3, dtype=np.float32)
    all_r = np.repeat(ident[None, None], frames, axis=0)
    all_r = np.repeat(all_r, 24, axis=1)
    all_r[:, 0] = root
    x[:, ROT6D_START:] = matrix_to_rot6d_np(all_r).reshape(frames, -1)
    return x


def test_entry_heading_canonicalization():
    x = make_motion(120, 75.0, 95.0)
    y, rep = canonicalize_event_entry_heading_np(x, fps=30.0)
    assert abs(float(np.degrees(root_yaw_np(y[:1])[0]))) < 1.0
    assert abs(rep["entry_heading_before_deg"] - 75.0) < 1.0


def test_nonturn_reset_is_dropped():
    os.environ["V46_50_DROP_RESET_OR_DRIFT"] = "1"
    x = make_motion(240, 0.0, 330.0)
    _, rep = enforce_event_heading_contract(
        x,
        {
            "dance_key": "thirty_six_postures",
            "event_family": "pose_motif",
            "music_alignment_label": "pose_hold",
        },
        fps=30.0,
    )
    assert rep["intent"] == "reset_or_drift", rep
    assert not rep["valid"], rep


def test_sogdian_spin_is_retained():
    x = make_motion(180, 20.0, 380.0)
    y, rep = enforce_event_heading_contract(
        x,
        {
            "dance_key": "sogdian_whirl",
            "event_family": "turning_flow",
            "music_alignment_label": "turning_climax",
        },
        fps=30.0,
    )
    assert rep["intent"] == "explicit_spin", rep
    assert rep["valid"], rep
    assert abs(heading_metrics_np(y)["net_yaw_deg"] - 360.0) < 2.0


def test_planned_heading_restore():
    ref = make_motion(90, 0.0, 60.0)
    gen = make_motion(90, 25.0, 130.0)
    fixed, rep = restore_planned_root_heading_np(gen, ref)
    err = np.degrees(
        np.arctan2(
            np.sin(root_yaw_np(fixed) - root_yaw_np(ref)),
            np.cos(root_yaw_np(fixed) - root_yaw_np(ref)),
        )
    )
    assert np.percentile(np.abs(err), 95) < 1e-3
    assert rep["yaw_error_after_deg_p95"] < 1e-3


def test_slot_policy_rejects_spin_for_pose():
    penalty, detail = candidate_heading_penalty(
        {
            "event_turn_intent": "explicit_spin",
            "event_stage_delta_yaw_rad": 2 * math.pi,
            "event_heading_quality": 1.0,
            "event_heading_valid": True,
        },
        {"role": "resolution", "music_alignment_label": "pose_hold"},
        0.0,
        recent_turn_count=0,
    )
    assert detail["hard_reject"]
    assert penalty > 10.0


def test_segmentation_covers_sequence_without_gaps():
    x = make_motion(900, 0.0, 40.0)
    segs, rep = adaptive_event_segments(
        x,
        {
            "dance_key": "lotus_steps",
            "event_family": "footwork_flow",
            "natural_duration_range_sec": [1.5, 4.0],
        },
        fps=30.0,
    )
    assert segs[0][0] == 0
    assert segs[-1][1] == len(x)
    for (a, b), (c, d) in zip(segs[:-1], segs[1:]):
        assert b == c
        assert b > a
        assert d > c


if __name__ == "__main__":
    tests = [
        test_entry_heading_canonicalization,
        test_nonturn_reset_is_dropped,
        test_sogdian_spin_is_retained,
        test_planned_heading_restore,
        test_slot_policy_rejects_spin_for_pose,
        test_segmentation_covers_sequence_without_gaps,
    ]
    for fn in tests:
        fn()
        print("[PASS]", fn.__name__)
    print(f"[PASS] {len(tests)} V46.50 tests")
