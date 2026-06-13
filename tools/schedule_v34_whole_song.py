#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V34 whole-song overlay: contact-consistent C3 stitching and warp pruning."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Importing V32 first installs the context capture, no-post-hoc-root rewrite and
# strict locked-warp audit onto the single V26 scheduler module.
import tools.schedule_v32_whole_song as v32_overlay
from tools.v32_transition_quality import transition_risk
from tools.v34_boundary_dynamics import apply_exit_handshake_np
from tools.v34_warp_aware_retrieval import choose_events_v34

scheduler = v32_overlay.scheduler


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _absolute_boundary_checks(risk: Dict[str, float]) -> Dict[str, bool]:
    return {
        "boundary_jerk": risk["boundary_joint_jerk_max"]
        <= float(os.getenv("V34_MAX_BOUNDARY_JERK", "5000")),
        "boundary_angular_jerk": risk["boundary_angular_jerk_max"]
        <= float(os.getenv("V34_MAX_BOUNDARY_ANGULAR_JERK", "5000")),
        "exit_rotation": risk["exit_rotation_step_rad"]
        <= float(os.getenv("V34_MAX_EXIT_ROTATION_STEP_RAD", "0.12")),
        "exit_fk": risk["exit_fk_jump"]
        <= float(os.getenv("V34_MAX_EXIT_FK_JUMP", "0.040")),
        "exit_acceleration": risk["exit_acceleration"]
        <= float(os.getenv("V34_MAX_EXIT_ACCELERATION", "12.0")),
    }


def _risk_score(risk: Dict[str, float]) -> float:
    thresholds = {
        "boundary_joint_jerk_max": float(
            os.getenv("V34_MAX_BOUNDARY_JERK", "5000")
        ),
        "boundary_angular_jerk_max": float(
            os.getenv("V34_MAX_BOUNDARY_ANGULAR_JERK", "5000")
        ),
        "exit_rotation_step_rad": float(
            os.getenv("V34_MAX_EXIT_ROTATION_STEP_RAD", "0.12")
        ),
        "exit_fk_jump": float(os.getenv("V34_MAX_EXIT_FK_JUMP", "0.040")),
        "exit_acceleration": float(
            os.getenv("V34_MAX_EXIT_ACCELERATION", "12.0")
        ),
    }
    return float(max(
        float(risk[key]) / max(value, 1e-8)
        for key, value in thresholds.items()
    ))


def _adaptive_exit_handshake(
    transition: np.ndarray,
    content: np.ndarray,
    previous_content: np.ndarray,
    fps: float,
    default_frames: int,
    strength: float,
    max_rotation_deg: float,
    max_root: float,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, float], Dict[str, float]]:
    raw = os.getenv(
        "V34_EXIT_HANDSHAKE_CANDIDATES",
        f"{default_frames},{default_frames + 4},{default_frames + 8}",
    )
    lengths = sorted({
        max(3, int(value.strip()))
        for value in raw.split(",") if value.strip()
    })
    attempts: List[Dict[str, Any]] = []
    best = None
    for frames in lengths:
        candidate, metadata = apply_exit_handshake_np(
            transition,
            content,
            frames=frames,
            strength=strength,
            max_rotation_deg=max_rotation_deg,
            max_root_y=max_root,
        )
        main_risk = transition_risk(
            previous_content, transition, candidate, fps=fps
        )
        h = int(metadata.get("frames", 0))
        if h >= 3 and h < len(content) - 1:
            tail_previous = transition[-4:] if len(transition) >= 4 else transition
            tail_transition = candidate[:h]
            tail_following = content[h:min(len(content), h + 4)]
            tail_risk = transition_risk(
                tail_previous, tail_transition, tail_following, fps=fps
            )
        else:
            tail_risk = main_risk
        main_checks = _absolute_boundary_checks(main_risk)
        tail_checks = _absolute_boundary_checks(tail_risk)
        safe = bool(all(main_checks.values()) and all(tail_checks.values()))
        score = max(_risk_score(main_risk), _risk_score(tail_risk))
        # Prefer smaller corrections when two candidates are equally safe.
        score += 0.001 * float(
            metadata.get("raw_rotation_correction_deg", 0.0)
        )
        row = {
            "frames_requested": int(frames),
            "metadata": metadata,
            "main_risk": main_risk,
            "tail_risk": tail_risk,
            "main_checks": main_checks,
            "tail_checks": tail_checks,
            "safe": safe,
            "score": float(score),
        }
        attempts.append(row)
        candidate_tuple = (not safe, score, frames, candidate, row)
        if best is None or candidate_tuple[:3] < best[:3]:
            best = candidate_tuple
    if best is None:
        return content.copy(), {
            "enabled": False,
            "reason": "no_handshake_candidates",
            "attempts": [],
        }, transition_risk(previous_content, transition, content, fps=fps), transition_risk(previous_content, transition, content, fps=fps)
    _, _, _, selected, row = best
    metadata = dict(row["metadata"])
    metadata.update({
        "adaptive": True,
        "safe": bool(row["safe"]),
        "selected_score": float(row["score"]),
        "attempts": attempts,
    })
    return selected, metadata, row["main_risk"], row["tail_risk"]


def generate_one_v34(
    audio_path: Path,
    arrays,
    hierarchy,
    items,
    motions,
    router,
    transition_bundle,
    v23_bundle,
    planner_bundle,
    device,
    args,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    features, audio_meta = scheduler.whole_song_features(
        audio_path,
        fps=args.fps,
        cache_dir=args.feature_dir,
        max_seconds=args.max_seconds,
    )
    source_phrases, segmentation = scheduler.segment_music_phrases(
        features,
        fps=args.fps,
        min_phrase_seconds=args.min_phrase_seconds,
        max_phrase_seconds=args.max_phrase_seconds,
        boundary_quantile=args.boundary_quantile,
        beat_snap_seconds=args.beat_snap_seconds,
    )
    phrases, slot_expansion = scheduler.split_music_phrases_for_events(
        features,
        source_phrases,
        fps=args.fps,
        enabled=args.multi_event_phrases,
        max_slot_seconds=args.max_single_event_seconds,
        min_slot_seconds=args.min_subphrase_seconds,
        max_events_per_phrase=args.max_events_per_phrase,
        beat_snap_seconds=args.slot_beat_snap_seconds,
        calm_max_slot_seconds=args.calm_max_single_event_seconds,
    )
    if len(phrases) > args.max_phrases:
        raise RuntimeError(
            f"{audio_path}: detected {len(phrases)} slots above "
            f"--max_phrases={args.max_phrases}"
        )
    phrase_semantics, semantic_meta = scheduler.phrase_semantic_matrix(
        audio_path,
        phrases,
        enabled=bool(args.deep_music_features),
        model_name=str(args.deep_music_model),
        cache_dir=args.deep_music_cache or args.feature_dir,
        require_deep=bool(args.require_deep_music),
        min_deep_success=float(args.deep_music_min_success),
    )
    predictions = scheduler.planner_predictions(
        phrases, planner_bundle, device
    )
    selected_state = scheduler.choose_events(
        phrases,
        phrase_semantics,
        predictions,
        arrays,
        hierarchy,
        items,
        router,
        motions,
        transition_bundle,
        device,
        args,
    )

    phrase_lengths = [phrase.length for phrase in phrases]
    natural_durations = [
        part["natural_duration"] for part in selected_state.parts
    ]
    planner_durations = [float(x) for x in predictions["durations"]]
    event_types = [part["motion_event"] for part in selected_state.parts]
    music_events = [phrase.music_event for phrase in phrases]
    transition_lengths = list(selected_state.transition_lengths)
    transition_lengths[0] = 0
    if args.lock_music_boundaries:
        for index, phrase in enumerate(phrases):
            cap = (
                0 if index == 0 else
                max(0, int(phrase.length) - int(args.min_content_frames))
            )
            if int(transition_lengths[index]) > cap:
                previous = int(transition_lengths[index])
                transition_lengths[index] = int(cap)
                selected_state.parts[index]["transition_len"] = int(cap)
                meta = dict(
                    selected_state.parts[index].get("transition_meta", {})
                )
                meta["pre_allocation_slot_budget_cap"] = int(cap)
                meta["pre_allocation_capped_from"] = previous
                meta["dominant_reason"] = "slot_budget"
                selected_state.parts[index]["transition_meta"] = meta

    music_speed_factors = [phrase.speed_factor for phrase in phrases]
    music_content_targets = [
        max(
            args.min_content_frames,
            phrase.length - transition_lengths[index],
        )
        for index, phrase in enumerate(phrases)
    ]
    allocation = scheduler.allocate_whole_song_durations(
        phrase_lengths=phrase_lengths,
        natural_durations=natural_durations,
        planner_durations=planner_durations,
        event_types=event_types,
        music_events=music_events,
        transition_lengths=transition_lengths,
        total_frames=len(features),
        music_weight=args.global_music_weight,
        natural_weight=args.global_natural_weight,
        planner_weight=args.global_planner_weight,
        min_content_frames=args.min_content_frames,
        min_warp=args.min_time_warp,
        max_warp=args.max_time_warp,
        music_speed_factors=music_speed_factors,
        music_content_targets=music_content_targets,
        allow_music_bound_override=args.allow_music_bound_override,
        lock_music_boundaries=args.lock_music_boundaries,
    )

    contents: List[np.ndarray] = []
    resampling_reports: List[Dict[str, Any]] = []
    for event_index, target_len in zip(
        selected_state.selected, allocation["content_lengths"]
    ):
        content, report = scheduler.resample_event_with_v23(
            motions[event_index],
            int(target_len),
            v23_bundle,
            device,
            min_turn_angle=args.v23_min_turn_angle,
            min_peak_dps=args.v23_min_peak_dps,
        )
        content[:, scheduler.ROOT_X] = 0.0
        content[:, scheduler.ROOT_Z] = 0.0
        content = scheduler.dampen_event_edges(
            content, args.edge_damping_frames, args.edge_damping_strength
        )
        contents.append(content)
        resampling_reports.append(report)

    pieces: List[np.ndarray] = []
    boundary_reports: List[Dict[str, Any]] = []
    handshake_frames = int(os.getenv("V34_EXIT_HANDSHAKE_FRAMES", "10"))
    handshake_strength = float(
        os.getenv("V34_EXIT_HANDSHAKE_STRENGTH", "1.0")
    )
    handshake_max_rotation = float(
        os.getenv("V34_HANDSHAKE_MAX_ROTATION_DEG", "18.0")
    )
    handshake_max_root = float(
        os.getenv("V34_HANDSHAKE_MAX_ROOT", "0.08")
    )

    for slot in range(len(contents)):
        content = contents[slot]
        if slot > 0:
            previous_content = contents[slot - 1]
            k = int(transition_lengths[slot])
            rough = scheduler.make_linear_transition(
                previous_content, content, k
            )
            transition = scheduler.refine_transition(
                transition_bundle,
                rough,
                previous_content[-1],
                content[0],
                np.asarray(phrases[slot].query, np.float32),
                device,
            )
            transition = scheduler.enforce_yaw_safe_transition(
                transition, previous_content, content
            )
            if args.transition_diffusion and args.transition_diffusion_ckpt:
                transition, diffusion_meta = scheduler.sample_transition_diffusion(
                    args.transition_diffusion_bundle,
                    previous_content[-1],
                    content[0],
                    k,
                    np.asarray(phrases[slot].query, np.float32),
                    rough=transition,
                    device=device,
                    blend=args.transition_diffusion_blend,
                    steps=args.transition_diffusion_steps,
                )
                transition = scheduler.enforce_yaw_safe_transition(
                    transition, previous_content, content
                )
            else:
                diffusion_meta = {"enabled": False}

            risk_before = transition_risk(
                previous_content, transition, content, fps=float(args.fps)
            )
            if _enabled("V34_EXIT_HANDSHAKE", "1"):
                content, handshake_meta, risk_after, handshake_tail_risk = (
                    _adaptive_exit_handshake(
                        transition,
                        content,
                        previous_content,
                        fps=float(args.fps),
                        default_frames=handshake_frames,
                        strength=handshake_strength,
                        max_rotation_deg=handshake_max_rotation,
                        max_root=handshake_max_root,
                    )
                )
                contents[slot] = content
            else:
                handshake_meta = {
                    "enabled": False,
                    "reason": "disabled_by_environment",
                }
                risk_after = transition_risk(
                    previous_content, transition, content, fps=float(args.fps)
                )
                handshake_tail_risk = risk_after

            if _enabled("V34_POST_HANDSHAKE_ABSOLUTE_VETO", "1"):
                safe = bool(
                    all(_absolute_boundary_checks(risk_after).values())
                    and all(
                        _absolute_boundary_checks(handshake_tail_risk).values()
                    )
                )
                handshake_meta["post_handshake_safe"] = safe
                if not safe:
                    if _enabled("V34_FAIL_ON_UNSAFE_BOUNDARY", "1"):
                        raise RuntimeError(
                            f"Unsafe V34 post-handshake boundary slot={slot}: "
                            f"main={risk_after}, tail={handshake_tail_risk}"
                        )
                    handshake_meta["post_handshake_unsafe"] = True

            metrics = scheduler.boundary_metrics(previous_content, content)
            metrics["transition_len"] = k
            metrics["transition_meta"] = selected_state.parts[slot].get(
                "transition_meta", {}
            )
            metrics["transition_diffusion"] = diffusion_meta
            metrics["v34_risk_before_handshake"] = risk_before
            metrics["v34_risk_after_handshake"] = risk_after
            metrics["v34_handshake_tail_risk"] = handshake_tail_risk
            metrics["v34_exit_handshake"] = handshake_meta
            boundary_reports.append(metrics)
            pieces.append(transition)
        pieces.append(content)

    motion = np.concatenate(pieces, axis=0).astype(np.float32)
    if len(motion) != len(features):
        raise AssertionError(
            f"V34 output length mismatch: generated={len(motion)} "
            f"music_frames={len(features)}"
        )
    motion[:, scheduler.ROOT_X] = 0.0
    motion[:, scheduler.ROOT_Z] = 0.0
    if args.start_pose:
        start_path = Path(args.start_pose)
        if start_path.is_file():
            motion = scheduler.apply_start_anchor(
                motion,
                np.load(start_path).astype(np.float32).reshape(-1),
                args.start_anchor_blend,
            )

    report = {
        "version": "v34_contact_backinjected_c3_whole_song",
        "audio": str(audio_path),
        "audio_meta": audio_meta,
        "planner_mode": str(predictions["mode"][0]),
        "music_semantic": semantic_meta,
        "segmentation": {
            **segmentation,
            "source_num_phrases": len(source_phrases),
            "source_boundaries": (
                [int(source_phrases[0].start)]
                + [int(phrase.end) for phrase in source_phrases]
                if source_phrases else []
            ),
            "event_slot_expansion": slot_expansion,
            "effective_num_slots": len(phrases),
            "effective_slot_boundaries": (
                [int(phrases[0].start)]
                + [int(phrase.end) for phrase in phrases]
                if phrases else []
            ),
        },
        "allocation": allocation,
        "score": selected_state.score,
        "schedule": [],
        "boundary_metrics": boundary_reports,
        "v34_policy": {
            "contact_back_injected": True,
            "warp_aware_retrieval": True,
            "regularised_septic_so3": True,
            "c3_zero_inr_envelope": True,
            "cross_boundary_absolute_gate": True,
            "exit_handshake": _enabled("V34_EXIT_HANDSHAKE", "1"),
            "exit_handshake_frames": handshake_frames,
        },
        "timing_policy": {
            "hierarchical_retrieval": bool(args.hierarchical_retrieval),
            "graph_scheduler": bool(args.graph_scheduler),
            "hierarchy_index_npz": str(args.hierarchy_index_npz),
            "hierarchy_weight": float(args.hierarchy_weight),
            "graph_node_top_k": int(args.graph_node_top_k),
            "graph_edge_weight": float(args.graph_edge_weight),
            "music_dominant_timing": bool(args.music_dominant_timing),
            "transition_min_frames": int(args.transition_min_frames),
            "transition_max_frames": int(args.transition_max_frames),
            "transition_diffusion": bool(args.transition_diffusion),
            "transition_diffusion_ckpt": str(args.transition_diffusion_ckpt),
            "transition_diffusion_blend": float(
                args.transition_diffusion_blend
            ),
            "transition_diffusion_steps": int(
                args.transition_diffusion_steps
            ),
        },
    }
    for slot, part in enumerate(selected_state.parts):
        merged = dict(part)
        if slot < len(slot_expansion.get("slot_meta", [])):
            merged["slot_meta"] = slot_expansion["slot_meta"][slot]
        merged["allocated_content_len"] = int(
            allocation["content_lengths"][slot]
        )
        merged["allocated_phrase_total"] = int(
            allocation["phrase_total_lengths"][slot]
        )
        merged["time_warp_ratio"] = float(
            allocation["warp_ratios"][slot]
        )
        merged["resampling"] = resampling_reports[slot]
        report["schedule"].append(merged)
    return motion, report


scheduler.choose_events = choose_events_v34
scheduler.generate_one = generate_one_v34


if __name__ == "__main__":
    scheduler.main()
