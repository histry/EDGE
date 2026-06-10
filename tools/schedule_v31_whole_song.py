#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk-controlled V31 overlay for the existing V26 whole-song scheduler.

V31 intentionally reduces pipeline coupling:
- V26 remains the single source of truth for phrase boundaries, duration
  allocation, retrieval beam search and exact whole-song length.
- SO(3) geometry is replaced by deterministic C2 paths.
- Event edge damping is disabled by default.
- Learned cross-modal retrieval is an optional, auditable score term and is
  zero-weight by default.
- Transition music conditions are associated by exact slot order, never by
  nearest-neighbour matching of repeated rule queries.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch

import tools.schedule_v26_whole_song as scheduler
import tools.v26_hierarchical_graph_scheduler as hierarchy_module
from tools.v29_motion_geometry import (
    apply_start_anchor_so3,
    endpoint_metrics_np,
    project_motion_rotations_np,
)
from tools.v31_bandlimited_transition import make_c2_transition_np
from tools.v30_geometric_alignment import poincare_distance_pairwise
from tools.v27_transition_diffusion import set_transition_risk_context

_original_phrase_semantic_matrix = scheduler.phrase_semantic_matrix
_original_build_slot_query = hierarchy_module.build_slot_query
_original_node_scores = hierarchy_module.hierarchical_node_scores
_original_sample_transition = scheduler.sample_transition_diffusion
_original_allocate_durations = scheduler.allocate_whole_song_durations

_SLOT_GEOMETRIC = np.zeros((0, 1), np.float32)
_TRANSITION_SLOT = 1
_CURRENT_PREVIOUS: np.ndarray | None = None
_CURRENT_FOLLOWING: np.ndarray | None = None


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _boundary_metrics(previous, following):
    return endpoint_metrics_np(previous, following, fps=30.0)


def _no_edge_damping(
    motion: np.ndarray, edge_frames: int, strength: float
) -> np.ndarray:
    # Event-internal timing should not be modified unless an ablation explicitly
    # enables it. Previous versions damped every selected event before every
    # transition, which can create visible acceleration shoulders.
    if not _bool_env("V31_ENABLE_EDGE_DAMPING", False):
        return np.asarray(motion, np.float32).copy()
    from tools.v29_motion_geometry import dampen_event_edges_so3
    return dampen_event_edges_so3(motion, edge_frames, strength)


def _identity_yaw_safety(
    transition: np.ndarray,
    previous: np.ndarray,
    following: np.ndarray,
) -> np.ndarray:
    # V31 paths are already full-SO(3). A post-hoc root interpolation after the
    # safety gate would invalidate the measured candidate.
    result = project_motion_rotations_np(
        np.asarray(transition, np.float32)
    )
    result[:, 4] = 0.0
    result[:, 6] = 0.0
    return result


def _phrase_semantics(
    audio_path,
    phrases,
    enabled=False,
    model_name="clap",
    cache_dir=None,
    require_deep=False,
    min_deep_success=0.80,
):
    global _SLOT_GEOMETRIC, _TRANSITION_SLOT
    # Stable rule retrieval deliberately disables the old fixed-random CLAP
    # projection. Real CLAP enters only through the trained geometric aligner.
    rule, rule_meta = _original_phrase_semantic_matrix(
        audio_path,
        phrases,
        enabled=False,
        model_name=model_name,
        cache_dir=cache_dir,
        require_deep=False,
        min_deep_success=min_deep_success,
    )
    _TRANSITION_SLOT = 1
    checkpoint = os.getenv("V30_ALIGNMENT_CKPT", "").strip()
    if checkpoint:
        from tools.v30_deep_music_features import phrase_geometric_matrix
        geometric, geometric_meta = phrase_geometric_matrix(
            audio_path,
            phrases,
            enabled=enabled,
            model_name=model_name,
            cache_dir=cache_dir,
            require_deep=require_deep,
            min_deep_success=min_deep_success,
        )
        _SLOT_GEOMETRIC = np.asarray(geometric, np.float32)
        rule_meta = {
            **rule_meta,
            "v31_geometric_transition_condition": geometric_meta,
        }
    else:
        if require_deep:
            raise RuntimeError(
                "V31 strict deep-music mode requires V30_ALIGNMENT_CKPT; "
                "the old fixed random CLAP projection is intentionally disabled."
            )
        _SLOT_GEOMETRIC = np.asarray(rule, np.float32)
        rule_meta = {
            **rule_meta,
            "v31_geometric_transition_condition": {
                "enabled": False,
                "reason": "V30_ALIGNMENT_CKPT_not_set",
            },
        }
    # Keep the stable rule-level retrieval path unless the explicit geometry
    # ablation is enabled.
    if _bool_env("V31_ENABLE_GEOMETRIC_RETRIEVAL", False):
        return _SLOT_GEOMETRIC, rule_meta
    return np.asarray(rule, np.float32), rule_meta


def _build_slot_query(
    phrase: Any,
    predicted_event: str,
    target_natural: float,
    desired_activity: float,
    music_semantic: np.ndarray | None = None,
    deep_music_weight: float = 0.0,
) -> Dict[str, Any]:
    query = _original_build_slot_query(
        phrase,
        predicted_event,
        target_natural,
        desired_activity,
        music_semantic=None,
        deep_music_weight=0.0,
    )
    if (
        _bool_env("V31_ENABLE_GEOMETRIC_RETRIEVAL", False)
        and music_semantic is not None
    ):
        query["v31_crossmodal_embed"] = np.asarray(
            music_semantic, np.float32
        ).reshape(-1)
    return query


def _node_scores(
    hierarchy: Dict[str, np.ndarray],
    query: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    base, parts = _original_node_scores(hierarchy, query)
    if not _bool_env("V31_ENABLE_GEOMETRIC_RETRIEVAL", False):
        return base, parts
    event = hierarchy.get("v30_crossmodal_embed")
    music = query.get("v31_crossmodal_embed")
    if event is None or music is None:
        raise RuntimeError(
            "Geometric retrieval enabled but v30_crossmodal_embed is absent"
        )
    curvature = float(np.asarray(
        hierarchy.get("v30_crossmodal_curvature", [1.0]),
        np.float32,
    ).reshape(-1)[0])
    with torch.no_grad():
        distance = poincare_distance_pairwise(
            torch.from_numpy(np.asarray(music, np.float32)).reshape(1, -1),
            torch.from_numpy(np.asarray(event, np.float32)),
            curvature,
        )[0].cpu().numpy()
    geometric = np.exp(-0.65 * distance).astype(np.float32)
    weight = float(np.clip(
        float(os.getenv("V31_GEOMETRIC_RETRIEVAL_WEIGHT", "0.0")),
        0.0, 0.50,
    ))
    score = ((1.0 - weight) * base + weight * geometric).astype(np.float32)
    parts["v31_geometric_score"] = geometric
    parts["v31_geometric_distance"] = distance.astype(np.float32)
    return score, parts


def _strict_duration_allocation(*args, **kwargs):
    result = _original_allocate_durations(*args, **kwargs)
    if not _bool_env("V31_STRICT_LOCKED_WARP", True):
        return result
    ratios = np.asarray(result.get("warp_ratios", []), np.float32)
    if ratios.size == 0:
        return result
    minimum = float(kwargs.get("min_warp", 0.0))
    maximum = float(kwargs.get("max_warp", 1e9))
    tolerance = float(os.getenv("V31_WARP_TOLERANCE", "0.02"))
    bad = np.flatnonzero(
        (ratios < minimum - tolerance)
        | (ratios > maximum + tolerance)
    )
    allowed = int(os.getenv("V31_MAX_WARP_VIOLATIONS", "0"))
    if len(bad) > allowed:
        raise RuntimeError(
            "V31 rejected locked-boundary duration allocation because "
            f"{len(bad)} slots exceed natural warp limits; indices={bad.tolist()}, "
            f"ratios={ratios[bad].tolist()}, allowed={allowed}. "
            "Adjust phrase splitting/retrieval rather than silently time-warping "
            "events into jitter."
        )
    result["v31_strict_warp_checked"] = True
    result["v31_warp_violation_indices"] = bad.tolist()
    return result


def _make_c2_with_context(
    previous: np.ndarray,
    following: np.ndarray,
    length: int,
) -> np.ndarray:
    global _CURRENT_PREVIOUS, _CURRENT_FOLLOWING
    _CURRENT_PREVIOUS = np.asarray(previous, np.float32).copy()
    _CURRENT_FOLLOWING = np.asarray(following, np.float32).copy()
    return make_c2_transition_np(previous, following, length)


def _transition_by_exact_slot(
    bundle,
    start_frame,
    end_frame,
    length,
    music_query,
    rough=None,
    device="cpu",
    blend=0.20,
    steps=32,
):
    global _TRANSITION_SLOT, _CURRENT_PREVIOUS, _CURRENT_FOLLOWING
    if _CURRENT_PREVIOUS is not None and _CURRENT_FOLLOWING is not None:
        set_transition_risk_context(
            _CURRENT_PREVIOUS, _CURRENT_FOLLOWING
        )
    _CURRENT_PREVIOUS = None
    _CURRENT_FOLLOWING = None
    condition = np.asarray(music_query, np.float32)
    if len(_SLOT_GEOMETRIC):
        index = min(max(_TRANSITION_SLOT, 0), len(_SLOT_GEOMETRIC) - 1)
        condition = _SLOT_GEOMETRIC[index]
    _TRANSITION_SLOT += 1
    return _original_sample_transition(
        bundle,
        start_frame,
        end_frame,
        length,
        condition,
        rough=rough,
        device=device,
        blend=blend,
        steps=steps,
    )


scheduler.make_linear_transition = _make_c2_with_context
scheduler.dampen_event_edges = _no_edge_damping
scheduler.apply_start_anchor = apply_start_anchor_so3
scheduler.boundary_metrics = _boundary_metrics
scheduler.enforce_yaw_safe_transition = _identity_yaw_safety
scheduler.phrase_semantic_matrix = _phrase_semantics
scheduler.build_slot_query = _build_slot_query
scheduler.hierarchical_node_scores = _node_scores
scheduler.sample_transition_diffusion = _transition_by_exact_slot
scheduler.allocate_whole_song_durations = _strict_duration_allocation

hierarchy_module.build_slot_query = _build_slot_query
hierarchy_module.hierarchical_node_scores = _node_scores


if __name__ == "__main__":
    scheduler.main()
