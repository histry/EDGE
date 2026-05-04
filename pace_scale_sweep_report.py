#!/usr/bin/env python3
"""Summarize trajectory-scale sweep diagnostics into CSV and print a table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_diag(path: Path):
    d = json.load(open(path, encoding="utf-8"))
    seg = (d.get("segments") or [{}])[0]
    return {
        "name": path.name.replace("_diag.json", ""),
        "retrieved_expr": d.get("retrieved_expressiveness_mean", ""),
        "retrieved_energy": d.get("retrieved_energy_mean", ""),
        "upper": d.get("generated_upper_activity", ""),
        "lower": d.get("generated_lower_activity", ""),
        "root_speed": d.get("generated_root_speed", ""),
        "spatial_range": d.get("generated_spatial_range", ""),
        "ADE": d.get("trajectory_ade_m", ""),
        "jerk": seg.get("transition_jerk", ""),
        "contact_break": seg.get("contact_phase_break", ""),
        "freezing": d.get("freezing_score", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics_dir", default="output/reward_collapse/diagnostics")
    ap.add_argument("--out", default="output/reward_collapse/pace_scale_sweep_summary.csv")
    ap.add_argument("--glob", default="*traj*_diag.json")
    args = ap.parse_args()
    paths = sorted(Path(args.diagnostics_dir).glob(args.glob))
    rows = [read_diag(p) for p in paths]
    if not rows:
        print("no diagnostics found")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved", args.out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
