#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate V21 multi-music schedules and pairwise diversity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.v21_common import ROOT_X, ROOT_Z, ROT, json_safe, motion_mmr_embedding


def motion_metrics(x: np.ndarray) -> Dict[str, float]:
    if x.ndim == 3:
        x = x[0]
    rot = x[:, ROT]
    vel = np.linalg.norm(np.diff(rot, axis=0), axis=-1) if len(x) > 1 else np.zeros((0,), dtype=np.float32)
    acc = np.abs(np.diff(vel)) if len(vel) > 1 else np.zeros((0,), dtype=np.float32)
    tail = vel[-30:] if len(vel) else vel
    return {
        "mean_activity": float(vel.mean()) if len(vel) else 0.0,
        "tail_activity": float(tail.mean()) if len(tail) else 0.0,
        "activity_p90": float(np.percentile(vel, 90)) if len(vel) else 0.0,
        "jerk": float(acc.mean()) if len(acc) else 0.0,
        "root_radius": float(np.sqrt(x[:, ROOT_X] ** 2 + x[:, ROOT_Z] ** 2).max()) if len(x) else 0.0,
        "contact_switch": float(np.abs(np.diff(x[:, :4], axis=0)).sum()) if len(x) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    motions: Dict[str, np.ndarray] = {}
    reports: Dict[str, dict] = {}
    for p in sorted(run_dir.glob("*_v21.npy")):
        key = p.name.replace("_v21.npy", "")
        motions[key] = np.load(p, allow_pickle=True).astype(np.float32)
        report = run_dir / f"{key}_v21.schedule_report.json"
        if report.is_file():
            reports[key] = json.loads(report.read_text(encoding="utf-8"))

    result = {"run_dir": str(run_dir), "motions": {}, "pairwise": []}
    embeddings = {}
    event_ids = {}
    families = {}
    for key, x in motions.items():
        xx = x[0] if x.ndim == 3 else x
        result["motions"][key] = motion_metrics(xx)
        embeddings[key] = motion_mmr_embedding(xx)
        schedule = reports.get(key, {}).get("schedule", [])
        event_ids[key] = {str(s.get("event_id", "")) for s in schedule}
        families[key] = {str(s.get("family_id", s.get("motion_family_id", ""))) for s in schedule}
        style_scores = [float(s.get("v21_style_score", s.get("dunhuang_style_score_v20f3", 0.0))) for s in schedule]
        result["motions"][key]["mean_style_score"] = float(np.mean(style_scores)) if style_scores else 0.0

    keys = sorted(motions)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            result["pairwise"].append(
                {
                    "a": a,
                    "b": b,
                    "motion_cosine": float(embeddings[a] @ embeddings[b]),
                    "event_overlap": len(event_ids[a] & event_ids[b]),
                    "family_overlap": len(families[a] & families[b]),
                }
            )

    out = Path(args.out) if args.out else run_dir / "V21_EVALUATION.json"
    out.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
