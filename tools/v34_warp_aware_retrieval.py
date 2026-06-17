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
        if len(feasible_indices) == 0:
            natural_min = float(np.min(natural)) if len(natural) else float("nan")
            natural_max = float(np.max(natural)) if len(natural) else float("nan")
            raise RuntimeError(
                f"V34 slot has no globally warp-feasible event: slot={slot}, "
                f"music_length={phrase.length}, bounds=[{minimum},{maximum}], "
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
        expanded: List[Any] = []
        rejected_warp = 0
        compat_rejected = 0
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
                            continue
                        transition_meta = dict(transition_meta)
                        transition_meta["boundary_compatibility"] = (
                            boundary_compat_meta
                        )

                desired_transition_len = int(transition_len)
                negotiated = _negotiate_transition_budget(
                    natural_duration=float(natural[idx]),
                    slot_length=int(phrase.length),
                    desired_transition=desired_transition_len,
                    first_slot=(slot == 0),
                    min_content_frames=int(args.min_content_frames),
                    transition_min_frames=int(args.transition_min_frames),
                    transition_max_frames=int(args.transition_max_frames),
                    minimum_warp=minimum,
                    maximum_warp=maximum,
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
                    - args.mmr_weight * mmr
                    - args.family_repeat_weight * same_family
                    - args.source_repeat_weight * same_source
                    - warp_weight * warp_penalty
                    - transition_budget_penalty
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
                    "warp_penalty": float(warp_penalty),
                    "transition_len": int(transition_len),
                    "transition_meta": transition_meta,
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
                    "gpu_boundary_cache": bool(gpu_boundary_cache_hit),
                    "mmr_penalty": float(mmr),
                    "score": float(score),
                }
                expanded.append(scheduler.CandidateState(
                    score=score,
                    selected=state.selected + [idx],
                    transition_lengths=state.transition_lengths
                    + [transition_len],
                    parts=state.parts + [part],
                ))

        if not expanded:
            raise RuntimeError(
                f"V34 found no warp-feasible candidate for slot={slot}, "
                f"music_length={phrase.length}, bounds=[{minimum},{maximum}], "
                f"warp_rejected={rejected_warp}, "
                f"compat_rejected={compat_rejected}, "
                f"globally_feasible={len(feasible_indices)}, "
                f"negotiated={negotiated_count}. The failure is now caused by "
                "boundary compatibility, graph/family/duplicate constraints, "
                "or an undersized shortlist rather than warp ranking. Increase "
                "the feasible shortlist first; relax compatibility only for "
                "ablation, and keep the strict warp gate enabled."
            )
        expanded.sort(key=lambda state: state.score, reverse=True)
        beam = expanded[: int(args.beam_size)]
    return beam[0]
