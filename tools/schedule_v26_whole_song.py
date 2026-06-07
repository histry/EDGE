#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V26 whole-song duration-aware ChoreoRAG scheduler.

Pipeline:
1. extract one music feature vector per output motion frame;
2. predict variable phrase structure for the complete song;
3. optionally predict phrase-level motion event, duration and transition sequence;
4. retrieve style/quality/safety-gated Dunhuang motion events;
5. solve one exact global duration budget for the whole song;
6. resample each event independently (V23 Tau never crosses phrase boundaries);
7. synthesize boundary-aware transitions;
8. assert exact output length; no hidden pad/trim.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model.v21_music_router import load_router_checkpoint
from model.v23_monotonic_duration import load_v23_checkpoint
from model.v26_whole_song_planner import load_v26_planner_checkpoint
from tools.schedule_v21_multi_music import (
    load_optional_transition,
    load_shared_index,
    precompute_music_similarity,
    predict_transition_len,
    refine_transition,
    rule_transition_len,
)
from tools.v21_common import (
    CONTACT,
    EVENT_TYPES,
    ROOT_X,
    ROOT_Z,
    ROT,
    apply_start_anchor,
    event_compatibility,
    json_safe,
    load_motion,
    make_linear_transition,
    transition_cost_from_arrays,
)
from tools.v26_event_resampling import resample_event_with_v23
from tools.v26_global_duration_alignment import allocate_whole_song_durations
from tools.v26_music_phrase_segmentation import (
    MusicPhrase,
    segment_music_phrases,
    whole_song_features,
)


@dataclass
class CandidateState:
    score: float
    selected: List[int]
    transition_lengths: List[int]
    parts: List[Dict[str, Any]]


def planner_predictions(
    phrases: Sequence[MusicPhrase],
    planner_bundle,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    k = len(phrases)
    if planner_bundle is None:
        event_map = {
            "calm_flow": "calm_flow",
            "release": "release",
            "build_up": "build_up",
            "climax": "high_tension",
            "accent": "arm_flourish",
            "section_change": "support_shift",
            "neutral_flow": "neutral_flow",
        }
        event_ids = np.asarray(
            [EVENT_TYPES.index(event_map.get(p.music_event, "neutral_flow")) for p in phrases],
            dtype=np.int64,
        )
        durations = np.asarray([max(12, p.length - (0 if i == 0 else 10)) for i, p in enumerate(phrases)], dtype=np.float32)
        transitions = np.asarray([0] + [3] * max(0, k - 1), dtype=np.int64)  # index 3 -> 12 frames
        activity = np.asarray([float(np.asarray(p.query)[0]) for p in phrases], dtype=np.float32)
        return {
            "event_ids": event_ids,
            "durations": durations,
            "transition_class": transitions,
            "activity": activity,
            "mode": np.asarray(["rule"], dtype=object),
        }

    model = planner_bundle["model"]
    features = np.stack([np.asarray(p.planner_feature, dtype=np.float32) for p in phrases])[None]
    with torch.no_grad():
        output = model(torch.from_numpy(features).to(device))
    return {
        "event_ids": output["event_logits"][0].argmax(-1).cpu().numpy().astype(np.int64),
        "durations": output["duration_frames"][0].cpu().numpy().astype(np.float32),
        "transition_class": output["transition_logits"][0].argmax(-1).cpu().numpy().astype(np.int64),
        "activity": output["activity"][0].cpu().numpy().astype(np.float32),
        "mode": np.asarray(["learned"], dtype=object),
    }


def choose_events(
    phrases: Sequence[MusicPhrase],
    predictions: Dict[str, np.ndarray],
    arrays,
    items: List[Dict[str, Any]],
    router,
    motions: Sequence[np.ndarray],
    transition_bundle,
    device: torch.device,
    args: argparse.Namespace,
) -> CandidateState:
    motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    mmr_embed = np.asarray(arrays["mmr_embed"], dtype=np.float32)
    style = np.asarray(arrays["style_score"], dtype=np.float32)
    quality = np.asarray(arrays["quality_score"], dtype=np.float32)
    safety = np.asarray(arrays["safety_score"], dtype=np.float32)
    natural = np.asarray(arrays["natural_duration"], dtype=np.float32)
    entry_pose = np.asarray(arrays["entry_pose"], dtype=np.float32)
    exit_pose = np.asarray(arrays["exit_pose"], dtype=np.float32)
    entry_vel = np.asarray(arrays["entry_vel"], dtype=np.float32)
    exit_vel = np.asarray(arrays["exit_vel"], dtype=np.float32)
    event_types = [str(item.get("event_type", "neutral_flow")) for item in items]
    families = [str(item.get("family_id", "")) for item in items]
    queries = [np.asarray(p.query, dtype=np.float32) for p in phrases]
    similarities = precompute_music_similarity(router, queries, motion_desc, device)

    beam = [CandidateState(0.0, [], [], [])]
    transition_choices = tuple(
        int(x)
        for x in (
            planner_bundle_lengths(args.planner_ckpt)
            if args.planner_ckpt
            else (6, 8, 10, 12, 14, 16)
        )
    )

    for slot, phrase in enumerate(phrases):
        predicted_event = EVENT_TYPES[int(predictions["event_ids"][slot])]
        predicted_duration = float(predictions["durations"][slot])
        desired_activity = float(predictions["activity"][slot])
        compat = np.asarray(
            [
                0.60 * event_compatibility(phrase.music_event, event)
                + 0.40 * (1.0 if event == predicted_event else event_compatibility(predicted_event, event))
                for event in event_types
            ],
            dtype=np.float32,
        )
        duration_match = 1.0 - np.minimum(
            np.abs(natural - predicted_duration) / max(predicted_duration, 1.0),
            1.0,
        )
        activity_match = 1.0 - np.minimum(np.abs(motion_desc[:, 0] - desired_activity), 1.0)
        base = (
            args.style_weight * style
            + args.quality_weight * quality
            + args.safety_weight * safety
            + args.music_weight * similarities[slot]
            + args.event_weight * compat
            + args.duration_weight * duration_match
            + args.activity_weight * activity_match
        )
        shortlist = np.argsort(base)[::-1][: min(args.candidate_top_k, len(items))]
        expanded: List[CandidateState] = []
        for state in beam:
            for raw_idx in shortlist:
                idx = int(raw_idx)
                if idx in state.selected:
                    continue
                family = families[idx]
                same_family = sum(1 for previous in state.selected if families[previous] == family)
                same_source = sum(
                    1
                    for previous in state.selected
                    if int(items[previous].get("source_id", -1)) == int(items[idx].get("source_id", -2))
                )
                if args.hard_family_unique and same_family > 0:
                    continue

                transition_len = 0
                transition_cost = 0.0
                if state.selected:
                    previous = state.selected[-1]
                    transition_cost = transition_cost_from_arrays(
                        exit_pose[previous],
                        exit_vel[previous],
                        entry_pose[idx],
                        entry_vel[idx],
                    )
                    if transition_bundle is not None:
                        transition_len = predict_transition_len(
                            transition_bundle,
                            motions[previous],
                            motions[idx],
                            queries[slot],
                            phrases[slot - 1].music_event,
                            event_types[idx],
                            device,
                        )
                    else:
                        class_index = int(predictions["transition_class"][slot])
                        transition_len = int(transition_choices[min(class_index, len(transition_choices) - 1)])
                        if transition_len <= 0:
                            transition_len = rule_transition_len(
                                phrases[slot - 1].music_event,
                                event_types[idx],
                                queries[slot],
                            )
                mmr = 0.0
                if state.selected:
                    mmr = max(float(mmr_embed[idx] @ mmr_embed[previous]) for previous in state.selected)
                score = (
                    state.score
                    + float(base[idx])
                    - args.transition_weight * transition_cost
                    - args.mmr_weight * mmr
                    - args.family_repeat_weight * same_family
                    - args.source_repeat_weight * same_source
                )
                part = {
                    "slot": slot,
                    "music_start": phrase.start,
                    "music_end": phrase.end,
                    "music_length": phrase.length,
                    "music_event": phrase.music_event,
                    "predicted_motion_event": predicted_event,
                    "predicted_duration": predicted_duration,
                    "event_index": idx,
                    "event_id": str(items[idx].get("event_id", idx)),
                    "family_id": family,
                    "motion_event": event_types[idx],
                    "natural_duration": float(natural[idx]),
                    "transition_len": int(transition_len),
                    "style": float(style[idx]),
                    "quality": float(quality[idx]),
                    "safety": float(safety[idx]),
                    "music_similarity": float(similarities[slot, idx]),
                    "event_compatibility": float(compat[idx]),
                    "duration_match": float(duration_match[idx]),
                    "activity_match": float(activity_match[idx]),
                    "transition_cost": float(transition_cost),
                    "mmr_penalty": float(mmr),
                    "score": float(score),
                }
                expanded.append(
                    CandidateState(
                        score=score,
                        selected=state.selected + [idx],
                        transition_lengths=state.transition_lengths + [transition_len],
                        parts=state.parts + [part],
                    )
                )
        if not expanded:
            raise RuntimeError(
                f"No V26 candidate for phrase {slot}. Increase candidate_top_k or relax hard family uniqueness."
            )
        expanded.sort(key=lambda state: state.score, reverse=True)
        beam = expanded[: args.beam_size]
    return beam[0]


def planner_bundle_lengths(path: str) -> Tuple[int, ...]:
    if not path:
        return (6, 8, 10, 12, 14, 16)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(int(x) for x in checkpoint.get("config", {}).get("transition_lengths", (6, 8, 10, 12, 14, 16)))


def boundary_metrics(prev: np.ndarray, nxt: np.ndarray) -> Dict[str, float]:
    pv = prev[-1, ROT] - prev[-2, ROT] if len(prev) > 1 else np.zeros((144,), dtype=np.float32)
    nv = nxt[1, ROT] - nxt[0, ROT] if len(nxt) > 1 else np.zeros((144,), dtype=np.float32)
    pa = prev[-1, ROT] - 2.0 * prev[-2, ROT] + prev[-3, ROT] if len(prev) > 2 else np.zeros((144,), dtype=np.float32)
    na = nxt[2, ROT] - 2.0 * nxt[1, ROT] + nxt[0, ROT] if len(nxt) > 2 else np.zeros((144,), dtype=np.float32)
    return {
        "pose_jump": float(np.linalg.norm(prev[-1, ROT] - nxt[0, ROT]) / np.sqrt(144.0)),
        "velocity_jump": float(np.linalg.norm(pv - nv) / np.sqrt(144.0)),
        "acceleration_jump": float(np.linalg.norm(pa - na) / np.sqrt(144.0)),
        "contact_jump": float(np.abs(prev[-1, CONTACT] - nxt[0, CONTACT]).mean()),
    }


def generate_one(
    audio_path: Path,
    arrays,
    items,
    motions,
    router,
    transition_bundle,
    v23_bundle,
    planner_bundle,
    device,
    args,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    features, audio_meta = whole_song_features(
        audio_path,
        fps=args.fps,
        cache_dir=args.feature_dir,
        max_seconds=args.max_seconds,
    )
    phrases, segmentation = segment_music_phrases(
        features,
        fps=args.fps,
        min_phrase_seconds=args.min_phrase_seconds,
        max_phrase_seconds=args.max_phrase_seconds,
        boundary_quantile=args.boundary_quantile,
        beat_snap_seconds=args.beat_snap_seconds,
    )
    if len(phrases) > args.max_phrases:
        raise RuntimeError(
            f"{audio_path}: detected {len(phrases)} phrases, above --max_phrases={args.max_phrases}. "
            "Increase max phrase duration or max_phrases."
        )
    predictions = planner_predictions(phrases, planner_bundle, device)
    selected_state = choose_events(
        phrases,
        predictions,
        arrays,
        items,
        router,
        motions,
        transition_bundle,
        device,
        args,
    )

    phrase_lengths = [phrase.length for phrase in phrases]
    natural_durations = [part["natural_duration"] for part in selected_state.parts]
    planner_durations = [float(x) for x in predictions["durations"]]
    event_types = [part["motion_event"] for part in selected_state.parts]
    music_events = [phrase.music_event for phrase in phrases]
    transition_lengths = selected_state.transition_lengths
    transition_lengths[0] = 0
    allocation = allocate_whole_song_durations(
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
    )

    contents: List[np.ndarray] = []
    resampling_reports: List[Dict[str, Any]] = []
    for idx, target_len in zip(selected_state.selected, allocation["content_lengths"]):
        content, report = resample_event_with_v23(
            motions[idx],
            int(target_len),
            v23_bundle,
            device,
            min_turn_angle=args.v23_min_turn_angle,
            min_peak_dps=args.v23_min_peak_dps,
        )
        content[:, ROOT_X] = 0.0
        content[:, ROOT_Z] = 0.0
        contents.append(content)
        resampling_reports.append(report)

    pieces: List[np.ndarray] = []
    boundary_reports: List[Dict[str, Any]] = []
    for slot, content in enumerate(contents):
        if slot > 0:
            k = int(transition_lengths[slot])
            rough = make_linear_transition(contents[slot - 1], content, k)
            transition = refine_transition(
                transition_bundle,
                rough,
                contents[slot - 1][-1],
                content[0],
                np.asarray(phrases[slot].query, dtype=np.float32),
                device,
            )
            metrics = boundary_metrics(contents[slot - 1], content)
            metrics["transition_len"] = k
            boundary_reports.append(metrics)
            pieces.append(transition)
        pieces.append(content)

    motion = np.concatenate(pieces, axis=0).astype(np.float32)
    if len(motion) != len(features):
        raise AssertionError(
            f"V26 output length mismatch: generated={len(motion)} music_frames={len(features)}. "
            "No pad/trim fallback is permitted."
        )
    motion[:, ROOT_X] = 0.0
    motion[:, ROOT_Z] = 0.0
    if args.start_pose:
        start_path = Path(args.start_pose)
        if start_path.is_file():
            motion = apply_start_anchor(
                motion,
                np.load(start_path).astype(np.float32).reshape(-1),
                args.start_anchor_blend,
            )

    report = {
        "version": "v26_whole_song_duration_aware_choreorag",
        "audio": str(audio_path),
        "audio_meta": audio_meta,
        "planner_mode": str(predictions["mode"][0]),
        "segmentation": segmentation,
        "allocation": allocation,
        "score": selected_state.score,
        "schedule": [],
        "boundary_metrics": boundary_reports,
    }
    for slot, part in enumerate(selected_state.parts):
        merged = dict(part)
        merged["allocated_content_len"] = int(allocation["content_lengths"][slot])
        merged["allocated_phrase_total"] = int(allocation["phrase_total_lengths"][slot])
        merged["time_warp_ratio"] = float(allocation["warp_ratios"][slot])
        merged["resampling"] = resampling_reports[slot]
        report["schedule"].append(merged)
    return motion, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--music", action="append", default=[])
    parser.add_argument("--music_glob", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--v23_ckpt", required=True)
    parser.add_argument("--planner_ckpt", default="")
    parser.add_argument("--transition_ckpt", default="")
    parser.add_argument("--feature_dir", default="")
    parser.add_argument("--start_pose", default="")
    parser.add_argument("--start_anchor_blend", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_seconds", type=float, default=0.0)
    parser.add_argument("--min_phrase_seconds", type=float, default=2.5)
    parser.add_argument("--max_phrase_seconds", type=float, default=7.5)
    parser.add_argument("--boundary_quantile", type=float, default=0.68)
    parser.add_argument("--beat_snap_seconds", type=float, default=0.35)
    parser.add_argument("--max_phrases", type=int, default=96)
    parser.add_argument("--beam_size", type=int, default=24)
    parser.add_argument("--candidate_top_k", type=int, default=256)
    parser.add_argument("--style_weight", type=float, default=1.35)
    parser.add_argument("--quality_weight", type=float, default=0.65)
    parser.add_argument("--safety_weight", type=float, default=0.35)
    parser.add_argument("--music_weight", type=float, default=0.85)
    parser.add_argument("--event_weight", type=float, default=0.70)
    parser.add_argument("--duration_weight", type=float, default=0.45)
    parser.add_argument("--activity_weight", type=float, default=0.25)
    parser.add_argument("--transition_weight", type=float, default=0.60)
    parser.add_argument("--mmr_weight", type=float, default=0.40)
    parser.add_argument("--family_repeat_weight", type=float, default=0.58)
    parser.add_argument("--source_repeat_weight", type=float, default=0.18)
    parser.add_argument("--hard_family_unique", action="store_true")
    parser.add_argument("--global_music_weight", type=float, default=1.0)
    parser.add_argument("--global_natural_weight", type=float, default=1.25)
    parser.add_argument("--global_planner_weight", type=float, default=0.75)
    parser.add_argument("--min_content_frames", type=int, default=12)
    parser.add_argument("--min_time_warp", type=float, default=0.65)
    parser.add_argument("--max_time_warp", type=float, default=1.55)
    parser.add_argument("--v23_min_turn_angle", type=float, default=10.0)
    parser.add_argument("--v23_min_peak_dps", type=float, default=14.0)
    args = parser.parse_args()

    paths = [Path(x) for x in args.music]
    if args.music_glob:
        paths.extend(Path(x) for x in sorted(glob.glob(args.music_glob)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise RuntimeError("Provide --music or --music_glob")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.feature_dir:
        args.feature_dir = str(out_dir / "music_features")

    _, arrays, items = load_shared_index(Path(args.index_json), Path(args.duration_index_npz))
    if "natural_duration" not in arrays.files:
        raise RuntimeError(
            "duration_index_npz lacks natural_duration. Run tools/build_v26_duration_index.py first."
        )
    motions = [
        load_motion(Path(str(item.get("pkl", item.get("path", "")))))
        for item in items
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = load_router_checkpoint(args.router_ckpt, device=device)
    transition_bundle = load_optional_transition(args.transition_ckpt, device)
    v23_bundle = load_v23_checkpoint(args.v23_ckpt, device=device)
    planner_bundle = load_v26_planner_checkpoint(args.planner_ckpt, device=device) if args.planner_ckpt else None

    summary = {
        "version": "v26_whole_song_duration_aware_choreorag",
        "planner_ckpt": args.planner_ckpt,
        "router_ckpt": args.router_ckpt,
        "v23_ckpt": args.v23_ckpt,
        "transition_ckpt": args.transition_ckpt,
        "results": {},
    }
    for path in paths:
        motion, report = generate_one(
            path,
            arrays,
            items,
            motions,
            router,
            transition_bundle,
            v23_bundle,
            planner_bundle,
            device,
            args,
        )
        key = path.stem
        npy_path = out_dir / f"{key}_v26.npy"
        report_path = out_dir / f"{key}_v26.schedule_report.json"
        np.save(npy_path, motion[None].astype(np.float32))
        report["out_npy"] = str(npy_path)
        report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
        summary["results"][key] = {
            "npy": str(npy_path),
            "report": str(report_path),
            "frames": int(len(motion)),
            "phrases": len(report["schedule"]),
            "event_ids": [row["event_id"] for row in report["schedule"]],
            "families": [row["family_id"] for row in report["schedule"]],
        }
        print(f"[SAVED] {key}: frames={len(motion)} phrases={len(report['schedule'])}")

    summary_path = out_dir / "V26_WHOLE_SONG_SUMMARY.json"
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
