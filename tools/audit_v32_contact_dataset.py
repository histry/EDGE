#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit V33 event-level contact supervision and overlap consistency.

Historical filename is retained for command compatibility.  The audit fails if
contacts were reconstructed after sliding-window sampling, if per-channel
occupancy collapses, or if overlapping windows disagree on any original event
frame.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _masked_contact_statistics(
    target: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
) -> Dict[str, object]:
    contact = np.clip(target[..., :4], 0.0, 1.0)
    valid = mask[..., None]
    denominator = valid.sum(axis=(0, 1)).clip(min=1.0)
    channel_rate = (contact * valid).sum(axis=(0, 1)) / denominator
    overall = float((contact * valid).sum() / max(float(valid.sum()) * 4.0, 1.0))
    hard = contact >= 0.5
    frame_valid = mask > 0.5
    valid_contacts = hard[frame_valid]
    all_four = float(np.mean(np.all(valid_contacts, axis=1))) if len(valid_contacts) else 0.0
    none = float(np.mean(np.all(~valid_contacts, axis=1))) if len(valid_contacts) else 1.0

    pair_mask = mask[:, 1:] * mask[:, :-1]
    switches = np.abs(np.diff(contact, axis=1))
    switch_rate = (
        (switches * pair_mask[..., None]).sum(axis=(0, 1))
        / max(float(pair_mask.sum()), 1.0)
    )
    confidence_mean = (
        (confidence * valid).sum(axis=(0, 1)) / denominator
    )
    return {
        "contact_rate_overall": overall,
        "contact_rate_per_channel": channel_rate.astype(float).tolist(),
        "contact_switch_rate_per_channel": switch_rate.astype(float).tolist(),
        "all_four_contact_rate": all_four,
        "no_contact_rate": none,
        "confidence_mean_per_channel": confidence_mean.astype(float).tolist(),
    }


def _overlap_consistency(data: np.lib.npyio.NpzFile) -> Dict[str, int]:
    required = {
        "contact_origin_id", "contact_target_start",
        "contact_target_end_exclusive", "contact_confidence",
    }
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(
            "Dataset lacks V33 synchronized-contact provenance arrays: "
            f"{sorted(missing)}"
        )
    target = np.asarray(data["target"], np.float32)
    length = np.asarray(data["length"], np.int32)
    confidence = np.asarray(data["contact_confidence"], np.float32)
    origin = np.asarray(data["contact_origin_id"], object)
    start = np.asarray(data["contact_target_start"], np.int32)
    end = np.asarray(data["contact_target_end_exclusive"], np.int32)

    seen: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    comparisons = overlaps = 0
    for sample_index in range(len(target)):
        source = str(origin[sample_index])
        left = int(start[sample_index])
        right = int(end[sample_index])
        k = int(length[sample_index])
        if not source or left < 0 or right <= left:
            continue
        if right - left != k:
            raise AssertionError(
                f"Sample {sample_index}: origin interval [{left},{right}) "
                f"does not match length={k}"
            )
        contacts = target[sample_index, :k, :4]
        conf = confidence[sample_index, :k]
        for local in range(k):
            key = (source, left + local)
            comparisons += 1
            if key in seen:
                overlaps += 1
                old_contact, old_conf = seen[key]
                if not np.array_equal(old_contact, contacts[local]):
                    raise AssertionError(
                        f"Contact conflict for original frame {key}: "
                        f"{old_contact.tolist()} vs {contacts[local].tolist()}"
                    )
                if not np.allclose(old_conf, conf[local], atol=1e-7, rtol=0.0):
                    raise AssertionError(
                        f"Confidence conflict for original frame {key}"
                    )
            else:
                seen[key] = (contacts[local].copy(), conf[local].copy())
    return {
        "unique_origin_frames": len(seen),
        "frame_comparisons": comparisons,
        "overlap_comparisons": overlaps,
        "conflicts": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--require_real_samples", type=int, default=1000)
    parser.add_argument("--min_contact_rate", type=float, default=0.05)
    parser.add_argument("--max_contact_rate", type=float, default=0.75)
    parser.add_argument("--min_channel_rate", type=float, default=0.02)
    parser.add_argument("--max_channel_rate", type=float, default=0.80)
    parser.add_argument("--max_all_four_rate", type=float, default=0.55)
    parser.add_argument("--min_mean_switch_rate", type=float, default=0.001)
    parser.add_argument("--require_overlap_comparisons", type=int, default=1000)
    parser.add_argument("--require_event_level_pipeline", type=int, default=1)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    required = {"target", "mask", "length", "sample_kind", "real_target", "meta"}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(f"Dataset missing arrays: {sorted(missing)}")
    target = np.asarray(data["target"], np.float32)
    mask = np.asarray(data["mask"], np.float32)
    kinds = np.asarray(data["sample_kind"], object)
    real = np.asarray(data["real_target"], np.bool_)
    confidence = np.asarray(
        data["contact_confidence"]
        if "contact_confidence" in data.files
        else np.zeros((*mask.shape, 4), np.float32),
        np.float32,
    )
    meta = json.loads(str(data["meta"].item()))
    contact_pipeline = dict(meta.get("contact_pipeline", {}))

    if bool(args.require_event_level_pipeline):
        if contact_pipeline.get("level") != "complete_event_before_window_sampling":
            raise RuntimeError(
                "Dataset is not event-level contact supervised. "
                f"contact_pipeline={contact_pipeline}"
            )
        if not bool(contact_pipeline.get("synchronised_slicing", False)):
            raise RuntimeError("Synchronized contact slicing is not enabled")
        if bool(contact_pipeline.get("window_level_relabeling", True)):
            raise RuntimeError("Window-level contact relabeling is forbidden")

    statistics = _masked_contact_statistics(target, mask, confidence)
    consistency = _overlap_consistency(data)
    source_counts = Counter(
        str(value) for value in np.asarray(
            data["contact_label_source"]
            if "contact_label_source" in data.files
            else ["missing"] * len(target),
            object,
        )
    )
    kind_counts = Counter(str(value) for value in kinds)
    report = {
        "version": "v33_event_level_contact_dataset_audit",
        "num_samples": int(len(target)),
        "real_samples": int(real.sum()),
        "synthetic_samples": int((~real).sum()),
        "sample_kind_counts": dict(kind_counts),
        "contact_label_source_counts": dict(source_counts),
        **statistics,
        "overlap_consistency": consistency,
        "length_min": int(np.asarray(data["length"]).min()),
        "length_max": int(np.asarray(data["length"]).max()),
        "length_mean": float(np.asarray(data["length"]).mean()),
        "contact_pipeline": contact_pipeline,
        "source_meta": meta,
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        key: value for key, value in report.items() if key != "source_meta"
    }, ensure_ascii=False, indent=2))

    errors = []
    if int(real.sum()) < int(args.require_real_samples):
        errors.append(
            f"real samples={int(real.sum())} < required={args.require_real_samples}"
        )
    overall = float(statistics["contact_rate_overall"])
    if not float(args.min_contact_rate) <= overall <= float(args.max_contact_rate):
        errors.append(
            f"overall contact rate={overall:.6f} outside "
            f"[{args.min_contact_rate},{args.max_contact_rate}]"
        )
    for channel, rate in enumerate(statistics["contact_rate_per_channel"]):
        if not float(args.min_channel_rate) <= float(rate) <= float(args.max_channel_rate):
            errors.append(
                f"channel {channel} contact rate={float(rate):.6f} outside "
                f"[{args.min_channel_rate},{args.max_channel_rate}]"
            )
    if float(statistics["all_four_contact_rate"]) > float(args.max_all_four_rate):
        errors.append(
            f"all-four contact rate={statistics['all_four_contact_rate']:.6f} "
            f"> {args.max_all_four_rate}"
        )
    mean_switch = float(np.mean(statistics["contact_switch_rate_per_channel"]))
    if mean_switch < float(args.min_mean_switch_rate):
        errors.append(
            f"mean switch rate={mean_switch:.6f} < {args.min_mean_switch_rate}"
        )
    if consistency["overlap_comparisons"] < int(args.require_overlap_comparisons):
        errors.append(
            f"overlap comparisons={consistency['overlap_comparisons']} < "
            f"required={args.require_overlap_comparisons}"
        )
    if errors:
        raise RuntimeError("V33 contact dataset audit failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
