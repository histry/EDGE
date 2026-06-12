#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build event-level contact labels before any sliding-window sampling.

This is the only supported contact reconstruction entry point for V33.  It
loads each complete indexed event, computes foot kinematics with the full event
context, calibrates thresholds globally across the event bank, smooths each
contact sequence once, and writes an immutable cache consumed by the transition
dataset builder.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import CONTACT, load_motion
from tools.v29_motion_geometry import motion_to_joint_positions_torch
from tools.v33_event_contacts import (
    CACHE_VERSION,
    CONTACT_CHANNELS,
    FOOT_JOINTS,
    LABEL_SOURCE,
    ContactCalibration,
    EventFeatures,
    calibrate_global_thresholds,
    event_identifier,
    event_path,
    existing_contacts_are_plausible,
    foot_features,
    infer_from_features,
    motion_fingerprint,
    parse_target_rates,
    write_cache,
)


def _batched_foot_positions(
    motions: Sequence[np.ndarray | None],
    device: torch.device,
    batch_size: int,
) -> List[np.ndarray | None]:
    output: List[np.ndarray | None] = [None] * len(motions)
    by_length: Dict[int, List[int]] = defaultdict(list)
    for index, motion in enumerate(motions):
        if motion is not None and len(motion) > 0:
            by_length[len(motion)].append(index)

    processed = 0
    total = sum(len(indices) for indices in by_length.values())
    for length in sorted(by_length):
        indices = by_length[length]
        for offset in range(0, len(indices), max(1, int(batch_size))):
            chunk = indices[offset : offset + max(1, int(batch_size))]
            array = np.stack([motions[index] for index in chunk]).astype(np.float32)
            with torch.no_grad():
                tensor = torch.from_numpy(array).to(device)
                positions = motion_to_joint_positions_torch(tensor)
                feet = positions[:, :, FOOT_JOINTS].detach().cpu().numpy()
            for local, index in enumerate(chunk):
                output[index] = feet[local].astype(np.float32)
            processed += len(chunk)
            print(f"[EVENT CONTACT FK] {processed}/{total}", flush=True)
    return output


def _summarise_labels(
    contacts: Sequence[np.ndarray],
    valid: Sequence[bool],
) -> Dict[str, Any]:
    rows = [np.asarray(value, np.float32) for value, ok in zip(contacts, valid) if ok]
    if not rows:
        return {
            "overall_rate": 0.0,
            "channel_rate": [0.0] * CONTACT_CHANNELS,
            "all_four_rate": 0.0,
            "no_contact_rate": 1.0,
            "switch_rate": [0.0] * CONTACT_CHANNELS,
        }
    concatenated = np.concatenate(rows, axis=0)
    switch_rows = [np.abs(np.diff(row, axis=0)) for row in rows if len(row) > 1]
    switches = (
        np.concatenate(switch_rows, axis=0)
        if switch_rows else np.zeros((0, CONTACT_CHANNELS), np.float32)
    )
    return {
        "overall_rate": float(concatenated.mean()),
        "channel_rate": concatenated.mean(axis=0).astype(float).tolist(),
        "all_four_rate": float(np.mean(np.all(concatenated >= 0.5, axis=1))),
        "no_contact_rate": float(np.mean(np.all(concatenated < 0.5, axis=1))),
        "switch_rate": (
            switches.mean(axis=0).astype(float).tolist()
            if len(switches) else [0.0] * CONTACT_CHANNELS
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--target_rates", default="0.42,0.42,0.38,0.38",
        help="Global target contact occupancies for four channels",
    )
    parser.add_argument("--height_weight", type=float, default=1.0)
    parser.add_argument("--horizontal_speed_weight", type=float, default=0.65)
    parser.add_argument("--vertical_speed_weight", type=float, default=0.20)
    parser.add_argument("--transition_penalty", type=float, default=1.40)
    parser.add_argument("--min_run", type=int, default=2)
    parser.add_argument("--max_gap", type=int, default=2)
    parser.add_argument("--probability_temperature", type=float, default=0.20)
    parser.add_argument(
        "--existing_contact_policy",
        choices=["auto", "preserve", "rebuild"],
        default="auto",
    )
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--min_overall_rate", type=float, default=0.05)
    parser.add_argument("--max_overall_rate", type=float, default=0.75)
    parser.add_argument("--max_all_four_rate", type=float, default=0.55)
    args = parser.parse_args()

    _, _, all_items = load_shared_index(
        Path(args.index_json), Path(args.duration_index_npz)
    )
    items = all_items[: args.max_events] if args.max_events > 0 else all_items
    config = ContactCalibration(
        fps=float(args.fps),
        target_rates=parse_target_rates(args.target_rates),
        height_weight=float(args.height_weight),
        horizontal_speed_weight=float(args.horizontal_speed_weight),
        vertical_speed_weight=float(args.vertical_speed_weight),
        transition_penalty=float(args.transition_penalty),
        min_run=int(args.min_run),
        max_gap=int(args.max_gap),
        probability_temperature=float(args.probability_temperature),
    )

    motions: List[np.ndarray | None] = []
    event_ids: List[str] = []
    event_paths: List[str] = []
    lengths: List[int] = []
    valid: List[bool] = []
    fingerprints: List[str] = []
    unresolved: List[Dict[str, Any]] = []

    for index, item in enumerate(items):
        identifier = event_identifier(item, index)
        path = event_path(item)
        event_ids.append(identifier)
        event_paths.append(path)
        try:
            motion = load_motion(path).astype(np.float32)
            if len(motion) < 3:
                raise ValueError(f"event too short: {len(motion)}")
            fingerprint = motion_fingerprint(motion)
        except Exception as error:
            motion = None
            fingerprint = ""
            lengths.append(0)
            valid.append(False)
            unresolved.append({
                "event_index": index,
                "event_id": identifier,
                "path": path,
                "error": repr(error),
            })
        else:
            lengths.append(len(motion))
            valid.append(True)
        motions.append(motion)
        fingerprints.append(fingerprint)
        if motion is None:
            continue

    device = torch.device(args.device)
    feet = _batched_foot_positions(
        motions, device=device, batch_size=int(args.batch_size)
    )
    features: List[EventFeatures | None] = []
    infer_indices: List[int] = []
    preserved_indices: List[int] = []
    for index, (motion, foot) in enumerate(zip(motions, feet)):
        if motion is None or foot is None:
            features.append(None)
            continue
        preserve = False
        if args.existing_contact_policy == "preserve":
            preserve = True
        elif args.existing_contact_policy == "auto":
            preserve = existing_contacts_are_plausible(motion, config)
        if preserve:
            preserved_indices.append(index)
        else:
            infer_indices.append(index)
        features.append(foot_features(foot, float(args.fps)))

    calibration_rows = [
        features[index] for index in infer_indices if features[index] is not None
    ]
    if calibration_rows:
        thresholds = calibrate_global_thresholds(
            calibration_rows, config
        )
    else:
        # This branch is reached only when every valid event already has
        # plausible contacts.  Thresholds are retained for future full-source
        # inference, using all events as a stable calibration bank.
        thresholds = calibrate_global_thresholds(
            [row for row in features if row is not None], config
        )

    contacts: List[np.ndarray] = []
    confidences: List[np.ndarray] = []
    probability_means: List[List[float]] = []
    for index, motion in enumerate(motions):
        if motion is None or features[index] is None:
            contacts.append(np.zeros((0, CONTACT_CHANNELS), np.float32))
            confidences.append(np.zeros((0, CONTACT_CHANNELS), np.float32))
            probability_means.append([0.0] * CONTACT_CHANNELS)
            continue
        if index in preserved_indices:
            hard = (motion[:, CONTACT] >= 0.5).astype(np.float32)
            confidence = np.ones_like(hard, dtype=np.float32)
            probability = np.clip(motion[:, CONTACT], 0.0, 1.0)
        else:
            hard, confidence, probability = infer_from_features(
                features[index], thresholds, config
            )
        contacts.append(hard)
        confidences.append(confidence)
        probability_means.append(
            probability.mean(axis=0).astype(float).tolist()
        )

    summary = _summarise_labels(contacts, valid)
    if not (
        float(args.min_overall_rate)
        <= summary["overall_rate"]
        <= float(args.max_overall_rate)
    ):
        raise RuntimeError(
            "Event-level contact occupancy is outside the safety range: "
            f"rate={summary['overall_rate']:.6f}, "
            f"range=[{args.min_overall_rate},{args.max_overall_rate}]"
        )
    if summary["all_four_rate"] > float(args.max_all_four_rate):
        raise RuntimeError(
            "All-four-contact collapse detected: "
            f"rate={summary['all_four_rate']:.6f}, "
            f"maximum={args.max_all_four_rate}"
        )

    metadata = {
        "version": CACHE_VERSION,
        "label_source": LABEL_SOURCE,
        "label_status": "kinematic_pseudo_contact_not_human_ground_truth",
        "level": "complete_indexed_event_before_window_sampling",
        "index_json": str(args.index_json),
        "duration_index_npz": str(args.duration_index_npz),
        "num_events": len(items),
        "valid_events": int(sum(valid)),
        "unresolved_events": int(len(unresolved)),
        "preserved_existing_events": len(preserved_indices),
        "reconstructed_events": len(infer_indices),
        "calibration": asdict(config),
        "thresholds": asdict(thresholds),
        "summary": summary,
        "probability_mean_per_event": probability_means,
        "unresolved": unresolved,
        "invariants": {
            "contact_computed_before_sliding_windows": True,
            "one_label_array_per_event": True,
            "window_level_relabeling_forbidden": True,
        },
    }
    output = Path(args.out_npz)
    write_cache(
        output,
        event_ids=event_ids,
        event_paths=event_paths,
        lengths=lengths,
        valid=valid,
        fingerprints=fingerprints,
        contacts=contacts,
        confidences=confidences,
        metadata=metadata,
    )
    report = Path(args.out_json)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        key: value for key, value in metadata.items()
        if key not in {"probability_mean_per_event", "unresolved"}
    }, ensure_ascii=False, indent=2))
    print(f"[SAVED] {output}")
    print(f"[SAVED] {report}")


if __name__ == "__main__":
    main()
