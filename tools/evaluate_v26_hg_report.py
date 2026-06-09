#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate V26/V27-HG whole-song choreography reports.

This tool complements the existing long-dance evaluator with metrics that are
specific to boundary-locked hierarchical graph scheduling:

- beat/accent alignment between motion activity peaks and music accents;
- warp-ratio distribution after multi-event phrase splitting;
- hierarchy retrieval and graph-edge cost summaries;
- event/family diversity and immediate repeat rates.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from tools.v26_music_phrase_segmentation import whole_song_features


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _load_motion(path: Path) -> np.ndarray:
    motion = np.load(path, allow_pickle=True)
    if motion.ndim == 3:
        motion = motion[0]
    return np.asarray(motion, dtype=np.float32)


def _local_peaks(x: np.ndarray, percentile: float = 75.0, min_gap: int = 8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < 3:
        return np.zeros((0,), dtype=np.int32)
    threshold = float(np.percentile(x, percentile))
    peaks: List[int] = []
    for i in range(1, len(x) - 1):
        if x[i] >= threshold and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            if not peaks or i - peaks[-1] >= int(min_gap):
                peaks.append(i)
    return np.asarray(peaks, dtype=np.int32)


def _nearest_distances(peaks: np.ndarray, references: np.ndarray) -> np.ndarray:
    if len(peaks) == 0 or len(references) == 0:
        return np.zeros((0,), dtype=np.float32)
    references = np.asarray(references, dtype=np.int32)
    dists = []
    for p in peaks:
        dists.append(int(np.min(np.abs(references - int(p)))))
    return np.asarray(dists, dtype=np.float32)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else 0.0


def evaluate(motion_path: Path, report_path: Path, out_json: Path, fps: float, feature_dir: str) -> Dict[str, Any]:
    motion = _load_motion(motion_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    schedule = list(report.get("schedule", []))
    audio = Path(str(report.get("audio", "")))
    if not audio.is_file():
        raise RuntimeError(f"Audio path in schedule report does not exist: {audio}")

    features, audio_meta = whole_song_features(audio, fps=fps, cache_dir=feature_dir or None)
    music_accent = 0.55 * features[:, 1] + 0.35 * features[:, 2] + 0.10 * features[:, 0]
    music_peaks = _local_peaks(music_accent, percentile=72.0, min_gap=max(4, int(round(0.18 * fps))))

    velocity = np.diff(motion, axis=0, prepend=motion[:1]) * float(fps)
    motion_activity = np.linalg.norm(velocity, axis=1)
    motion_peaks = _local_peaks(motion_activity, percentile=78.0, min_gap=max(6, int(round(0.22 * fps))))
    dists = _nearest_distances(motion_peaks, music_peaks)

    warp = [float(x) for x in report.get("allocation", {}).get("warp_ratios", [])]
    event_ids = [str(x.get("event_id", "")) for x in schedule]
    families = [str(x.get("family_id", "")) for x in schedule]
    graph_costs = [float(x.get("graph_edge_cost", 0.0)) for x in schedule[1:]]
    hierarchy_scores = [float(x.get("hierarchy_score", 0.0)) for x in schedule]
    hierarchy_hyper_scores = [float(x.get("hierarchy_hyper_score", 0.0)) for x in schedule]
    graph_reasons = Counter(
        str(x.get("transition_meta", {}).get("dominant_reason", "none"))
        for x in schedule
        if isinstance(x.get("transition_meta", {}), dict)
    )

    immediate_event_repeats = sum(1 for a, b in zip(event_ids, event_ids[1:]) if a == b)
    immediate_family_repeats = sum(1 for a, b in zip(families, families[1:]) if a == b)
    num_slots = max(len(schedule), 1)

    result: Dict[str, Any] = {
        "motion": str(motion_path),
        "schedule_report": str(report_path),
        "audio": str(audio),
        "audio_seconds": float(audio_meta.get("duration", 0.0)),
        "frames": int(motion.shape[0]),
        "fps": float(fps),
        "num_slots": int(len(schedule)),
        "source_num_phrases": int(report.get("segmentation", {}).get("source_num_phrases", 0)),
        "effective_num_slots": int(report.get("segmentation", {}).get("effective_num_slots", len(schedule))),
        "exact_length_match": bool(abs(int(motion.shape[0]) - int(round(float(audio_meta.get("duration", 0.0)) * fps))) <= 2),
        "beat_alignment": {
            "music_peak_count": int(len(music_peaks)),
            "motion_peak_count": int(len(motion_peaks)),
            "nearest_median_frames": float(np.median(dists)) if len(dists) else 0.0,
            "nearest_p90_frames": float(np.percentile(dists, 90)) if len(dists) else 0.0,
            "hit_rate_6f": float(np.mean(dists <= 6)) if len(dists) else 0.0,
            "hit_rate_10f": float(np.mean(dists <= 10)) if len(dists) else 0.0,
            "hit_rate_15f": float(np.mean(dists <= 15)) if len(dists) else 0.0,
        },
        "warp": {
            "max": max(warp) if warp else 0.0,
            "mean": _mean(warp),
            "p95": _percentile(warp, 95),
            "count_gt_1_3": int(sum(x > 1.3 for x in warp)),
            "count_gt_1_5": int(sum(x > 1.5 for x in warp)),
            "count_gt_2_0": int(sum(x > 2.0 for x in warp)),
        },
        "hierarchy": {
            "mean_score": _mean(hierarchy_scores),
            "p10_score": _percentile(hierarchy_scores, 10),
            "mean_hyper_score": _mean(hierarchy_hyper_scores),
        },
        "graph": {
            "mean_edge_cost": _mean(graph_costs),
            "p95_edge_cost": _percentile(graph_costs, 95),
            "max_edge_cost": max(graph_costs) if graph_costs else 0.0,
            "transition_reason_counts": dict(graph_reasons),
        },
        "diversity": {
            "unique_event_ratio": float(len(set(event_ids)) / num_slots),
            "unique_family_ratio": float(len(set(families)) / num_slots),
            "immediate_event_repeat_rate": float(immediate_event_repeats / max(len(schedule) - 1, 1)),
            "immediate_family_repeat_rate": float(immediate_family_repeats / max(len(schedule) - 1, 1)),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--schedule_report", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--feature_dir", default="data/v26_music_features_eval")
    args = parser.parse_args()

    result = evaluate(
        motion_path=Path(args.motion),
        report_path=Path(args.schedule_report),
        out_json=Path(args.out_json),
        fps=args.fps,
        feature_dir=args.feature_dir,
    )
    print(
        "[V27-HG eval]",
        Path(args.motion).stem,
        "hit10=",
        round(float(result["beat_alignment"]["hit_rate_10f"]), 4),
        "max_warp=",
        round(float(result["warp"]["max"]), 4),
        "unique_event=",
        round(float(result["diversity"]["unique_event_ratio"]), 4),
    )


if __name__ == "__main__":
    main()
