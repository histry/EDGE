#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a unified V28 experiment table across methods/baselines.

Each run is passed as ``name=directory``.  For every music key, the script
searches the directory for ``{key}_v26.npy`` and the matching schedule report,
then computes public-style metrics and extracts auditable module evidence:

- BAS / hit rates;
- kinematic FGD-style distance and diversity;
- warp statistics from schedule reports;
- deep music success rate and backend modes;
- transition diffusion usage count.

This is not a replacement for a canonical benchmark encoder.  It creates the
single comparison table needed before writing the paper, and makes baseline
comparison scripts reproducible instead of manually collecting JSON snippets.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from tools.evaluate_v27_public_metrics import (
    _load_generated,
    beat_alignment_score,
    diversity_score,
    frechet_distance,
    load_real_feature_bank,
    motion_window_features,
)


def _split_keys(value: str) -> List[str]:
    out: List[str] = []
    for chunk in str(value).replace(",", ";").split(";"):
        key = chunk.strip()
        if key:
            out.append(key)
    return out


def _parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def _find_motion(run_dir: Path, key: str) -> Path | None:
    candidates = [
        run_dir / f"{key}_v26.npy",
        run_dir / f"{key}.npy",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(run_dir.glob(f"{key}*.npy"))
    return matches[0] if matches else None


def _find_report(run_dir: Path, key: str) -> Path | None:
    candidates = [
        run_dir / f"{key}_v26.schedule_report.json",
        run_dir / f"{key}.schedule_report.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(run_dir.glob(f"{key}*schedule_report.json"))
    return matches[0] if matches else None


def _find_audio(audio_dir: Path, key: str, report: Dict[str, Any] | None) -> Path | None:
    if report:
        audio = Path(str(report.get("audio", "")))
        if audio.is_file():
            return audio
    for suffix in (".wav", ".mp3", ".flac", ".m4a"):
        path = audio_dir / f"{key}{suffix}"
        if path.is_file():
            return path
    matches = sorted(audio_dir.glob(f"{key}.*"))
    return matches[0] if matches else None


def _read_json(path: Path | None) -> Dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _warp_summary(report: Dict[str, Any] | None) -> Dict[str, Any]:
    if not report:
        return {
            "slots": 0,
            "source_phrases": 0,
            "max_warp": 0.0,
            "warp_gt_1_5": 0,
            "warp_gt_2_0": 0,
        }
    warp = [float(x) for x in report.get("allocation", {}).get("warp_ratios", [])]
    seg = report.get("segmentation", {})
    return {
        "slots": int(len(report.get("schedule", []))),
        "source_phrases": int(seg.get("source_num_phrases", seg.get("num_source_phrases", 0))),
        "max_warp": float(max(warp)) if warp else 0.0,
        "warp_gt_1_5": int(sum(x > 1.5 for x in warp)),
        "warp_gt_2_0": int(sum(x > 2.0 for x in warp)),
    }


def _module_evidence(report: Dict[str, Any] | None) -> Dict[str, Any]:
    if not report:
        return {
            "deep_music_success_rate": 0.0,
            "deep_music_modes": "",
            "transition_diffusion_count": 0,
            "transition_diffusion_enabled": False,
        }
    semantic = report.get("music_semantic", {})
    boundary = report.get("boundary_metrics", [])
    diff_count = 0
    for item in boundary:
        meta = item.get("transition_diffusion", {}) if isinstance(item, dict) else {}
        if isinstance(meta, dict) and bool(meta.get("enabled", False)):
            diff_count += 1
    return {
        "deep_music_success_rate": _safe_float(semantic.get("deep_success_rate", 0.0)),
        "deep_music_modes": "|".join(str(x) for x in semantic.get("unique_modes", [])),
        "transition_diffusion_count": int(diff_count),
        "transition_diffusion_enabled": bool(diff_count > 0),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _write_markdown(rows: List[Dict[str, Any]], out_path: Path) -> None:
    columns = [
        "method",
        "key",
        "bas",
        "hit_10f",
        "fgd_kinematic",
        "generated_diversity",
        "max_warp",
        "deep_music_success_rate",
        "transition_diffusion_count",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="name=directory. Repeat for baselines and proposed method.")
    parser.add_argument("--keys", required=True, help="semicolon/comma separated music keys")
    parser.add_argument("--audio_dir", default="test_music_bank")
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--out_md", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--feature_dir", default="data/v28_public_metric_features")
    parser.add_argument("--real_max_events", type=int, default=1200)
    args = parser.parse_args()

    if not args.run:
        raise RuntimeError("Provide at least one --run name=directory")

    keys = _split_keys(args.keys)
    audio_dir = Path(args.audio_dir)
    real_feat = load_real_feature_bank(args.index_json, args.duration_index_npz, fps=args.fps, max_events=args.real_max_events)
    rows: List[Dict[str, Any]] = []

    for run_arg in args.run:
        method, run_dir = _parse_run(run_arg)
        for key in keys:
            motion_path = _find_motion(run_dir, key)
            report_path = _find_report(run_dir, key)
            report = _read_json(report_path)
            audio_path = _find_audio(audio_dir, key, report)
            if motion_path is None or audio_path is None:
                rows.append(
                    {
                        "method": method,
                        "key": key,
                        "status": "missing_motion_or_audio",
                        "motion": str(motion_path or ""),
                        "audio": str(audio_path or ""),
                    }
                )
                continue
            motion = _load_generated(motion_path)
            gen_feat = motion_window_features(motion, fps=args.fps)
            bas = beat_alignment_score(motion, audio_path, args.fps, args.feature_dir)
            row: Dict[str, Any] = {
                "method": method,
                "key": key,
                "status": "ok",
                "motion": str(motion_path),
                "report": str(report_path or ""),
                "audio": str(audio_path),
                "frames": int(len(motion)),
                "bas": float(bas.get("bas", 0.0)),
                "hit_6f": float(bas.get("hit_6f", 0.0)),
                "hit_10f": float(bas.get("hit_10f", 0.0)),
                "hit_15f": float(bas.get("hit_15f", 0.0)),
                "nearest_median_frames": float(bas.get("nearest_median_frames", 0.0)),
                "fgd_kinematic": frechet_distance(gen_feat, real_feat),
                "generated_diversity": diversity_score(gen_feat),
                "num_generated_windows": int(len(gen_feat)),
                "num_real_windows": int(len(real_feat)),
            }
            row.update(_warp_summary(report))
            row.update(_module_evidence(report))
            rows.append(row)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary = []
    for method in sorted({str(row.get("method", "")) for row in ok_rows}):
        sub = [row for row in ok_rows if row.get("method") == method]
        summary.append(
            {
                "method": method,
                "num_cases": int(len(sub)),
                "bas_mean": _mean(float(row.get("bas", 0.0)) for row in sub),
                "hit_10f_mean": _mean(float(row.get("hit_10f", 0.0)) for row in sub),
                "fgd_kinematic_mean": _mean(float(row.get("fgd_kinematic", 0.0)) for row in sub),
                "generated_diversity_mean": _mean(float(row.get("generated_diversity", 0.0)) for row in sub),
                "max_warp_mean": _mean(float(row.get("max_warp", 0.0)) for row in sub),
                "deep_music_success_rate_mean": _mean(float(row.get("deep_music_success_rate", 0.0)) for row in sub),
                "transition_diffusion_count_sum": int(sum(int(row.get("transition_diffusion_count", 0)) for row in sub)),
            }
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(rows, out_md)

    print(f"[SAVED] table json: {out_json}")
    if args.out_csv:
        print(f"[SAVED] table csv: {args.out_csv}")
    if args.out_md:
        print(f"[SAVED] table md: {args.out_md}")
    for row in summary:
        print(
            "[SUMMARY]",
            row["method"],
            "BAS=",
            round(float(row["bas_mean"]), 4),
            "FGDkin=",
            round(float(row["fgd_kinematic_mean"]), 4),
            "deep=",
            round(float(row["deep_music_success_rate_mean"]), 3),
        )


if __name__ == "__main__":
    main()
