#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V32 overlay for V26 whole-song scheduling.

V26 remains the single planner. This overlay only replaces transition geometry:
  * captures real previous/next event context in make_linear_transition;
  * uses the V32 continuous contact-aware INR sampler;
  * disables post-sampling root-rotation rewriting;
  * disables event-edge damping by default;
  * uses SO(3)-interpretable boundary metrics.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

import tools.schedule_v26_whole_song as scheduler
from tools.v29_motion_geometry import (
    apply_start_anchor_so3,
    endpoint_metrics_np,
    project_motion_rotations_np,
)
from tools.v27_transition_diffusion import (
    sample_transition_diffusion as v32_sample_transition,
)
from tools.v32_contact_inr import make_c2_transition_np


_PREVIOUS_CONTEXT = np.zeros((0, 151), np.float32)
_NEXT_CONTEXT = np.zeros((0, 151), np.float32)
_ORIGINAL_ALLOCATE_DURATIONS = scheduler.allocate_whole_song_durations


def _capture_c2_transition(
    previous: np.ndarray,
    following: np.ndarray,
    length: int,
) -> np.ndarray:
    global _PREVIOUS_CONTEXT, _NEXT_CONTEXT
    _PREVIOUS_CONTEXT = np.asarray(previous, np.float32)[-4:].copy()
    _NEXT_CONTEXT = np.asarray(following, np.float32)[:4].copy()
    return make_c2_transition_np(previous, following, length)


def _strict_duration_allocation(*args, **kwargs):
    allocation = _ORIGINAL_ALLOCATE_DURATIONS(*args, **kwargs)
    strict = os.getenv(
        "V32_STRICT_LOCKED_WARP", "1"
    ).lower() in {"1", "true", "yes", "on"}
    if not strict:
        return allocation

    minimum = float(kwargs.get("min_warp", 0.0))
    maximum = float(kwargs.get("max_warp", 1e9))
    tolerance = float(os.getenv("V32_WARP_TOLERANCE", "0.02"))
    maximum_violations = int(
        os.getenv("V32_MAX_WARP_VIOLATIONS", "0")
    )
    ratios = [
        float(value) for value in allocation.get("warp_ratios", [])
    ]
    violations = [
        {
            "slot": index,
            "warp_ratio": value,
            "minimum": minimum,
            "maximum": maximum,
        }
        for index, value in enumerate(ratios)
        if value < minimum - tolerance or value > maximum + tolerance
    ]
    allocation["v32_strict_warp"] = {
        "enabled": True,
        "tolerance": tolerance,
        "violations": violations,
    }
    if len(violations) > maximum_violations:
        raise RuntimeError(
            "V32 strict locked-boundary warp rejected the schedule: "
            f"violations={len(violations)}, allowed={maximum_violations}, "
            f"examples={violations[:8]}"
        )
    return allocation


def _sample_with_context(
    bundle,
    start_frame,
    end_frame,
    length,
    music_query,
    rough=None,
    device="cpu",
    blend=0.35,
    steps=36,
):
    global _PREVIOUS_CONTEXT, _NEXT_CONTEXT
    result = v32_sample_transition(
        bundle,
        start_frame,
        end_frame,
        length,
        music_query,
        rough=rough,
        device=device,
        blend=blend,
        steps=steps,
        previous_context=_PREVIOUS_CONTEXT,
        next_context=_NEXT_CONTEXT,
    )
    _PREVIOUS_CONTEXT = np.zeros((0, 151), np.float32)
    _NEXT_CONTEXT = np.zeros((0, 151), np.float32)
    return result


def _identity_postprocess(
    transition: np.ndarray,
    previous: np.ndarray,
    following: np.ndarray,
) -> np.ndarray:
    # Root SO(3) is already generated and evaluated by V32. A post-hoc root
    # replacement would invalidate the safety gate and recreate exit spikes.
    result = project_motion_rotations_np(
        np.asarray(transition, np.float32)
    )
    result[:, 4] = 0.0
    result[:, 6] = 0.0
    return result


def _edge_damping(
    motion: np.ndarray, edge_frames: int, strength: float
) -> np.ndarray:
    if os.getenv(
        "V32_ENABLE_EDGE_DAMPING", "0"
    ).lower() not in {"1", "true", "yes", "on"}:
        return np.asarray(motion, np.float32).copy()
    return scheduler.dampen_event_edges_original(
        motion, edge_frames, strength
    )


def _boundary_metrics(
    previous: np.ndarray, following: np.ndarray
) -> Dict[str, float]:
    return endpoint_metrics_np(previous, following, fps=30.0)


# Preserve the original only for an explicit edge-damping ablation.
scheduler.dampen_event_edges_original = scheduler.dampen_event_edges
scheduler.make_linear_transition = _capture_c2_transition
scheduler.sample_transition_diffusion = _sample_with_context
scheduler.enforce_yaw_safe_transition = _identity_postprocess
scheduler.dampen_event_edges = _edge_damping
scheduler.boundary_metrics = _boundary_metrics
scheduler.apply_start_anchor = apply_start_anchor_so3
scheduler.allocate_whole_song_durations = _strict_duration_allocation


if __name__ == "__main__":
    scheduler.main()
