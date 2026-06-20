#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V34 warp-aware Event-RAG beam search.

Unlike V26, duration feasibility is evaluated after the candidate-specific
transition length is known.  In strict mode, an event is never admitted to the
beam when its exact locked-slot warp lies outside the allowed interval.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

import tools.schedule_v26_whole_song as scheduler

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


def _soft_ratio(value: float, limit: float) -> float:
    return float(max(0.0, value / max(limit, 1e-8) - 1.0))


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
    recent = [int(x) for x in selected[-memory_window:]]
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
        "body": _soft_ratio(body_jump, body_allow),
        "activity": _soft_ratio(activity_jump, activity_allow),
        "turn": _soft_ratio(turn_jump, turn_allow),
        "event": _soft_ratio(event_jump, event_allow),
        "duration": _soft_ratio(duration_jump, duration_allow),
        "memory_activity": _soft_ratio(memory_activity, memory_activity_allow),
        "memory_body": _soft_ratio(memory_body, memory_body_allow),
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
    queries = [np.asarray(phrase.query, np.float32) for phrase in phrases]
    similarities = scheduler.precompute_music_similarity(
        router, queries, motion_desc, device
    )
    transition_choices = scheduler.planner_bundle_lengths(args.planner_ckpt)
    gpu_cache = None
    if build_v34_gpu_candidate_cache is not None:
        gpu_cache = build_v34_gpu_candidate_cache(arrays, motions, device)

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

    beam = [scheduler.CandidateState(0.0, [], [], [])]
    for slot, phrase in enumerate(phrases):
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
        shortlist = ranked_feasible[: min(node_top_k, len(ranked_feasible))]
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
                    if int(items[previous].get("source_id", -1))
                    == int(items[idx].get("source_id", -2))
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
                    "candidate_top_k": int(args.candidate_top_k),
                    "graph_node_top_k": int(node_top_k),
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
            expanded = relaxed_expanded
            print(
                "[V34-RELAX] "
                f"slot={slot} strict feasible set empty; "
                f"using {len(expanded)} relaxed candidates "
                f"(compat_deferred={compat_deferred}, "
                f"semantic_deferred={semantic_deferred}, "
                f"contact_deferred={contact_deferred})."
            )
            for relaxed_state in expanded:
                if not relaxed_state.parts:
                    continue
                relaxed_part = relaxed_state.parts[-1]
                relaxed_part["constraint_relaxation_used"] = True
                relaxed_transition_meta = dict(
                    relaxed_part.get("transition_meta", {})
                )
                relax_meta = dict(
                    relaxed_transition_meta.get("constraint_relaxation", {})
                )
                relax_meta["used_due_to_empty_strict"] = True
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
