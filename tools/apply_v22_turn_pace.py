#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply a trained V22 turn-pace refiner to an existing V21/V22 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.v22_turn_runtime import load_pace_bundle, refine_motion_turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_version", choices=["v21", "v22"], default="v21")
    ap.add_argument("--output_suffix", default="v22_turnpace")
    ap.add_argument("--threshold_ratio", type=float, default=1.08)
    ap.add_argument("--window", type=int, default=72)
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--strength", type=float, default=0.90)
    ap.add_argument("--max_events", type=int, default=4)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_pace_bundle(args.checkpoint, device=device)
    reports = sorted(run_dir.glob(f"*_{args.input_version}.schedule_report.json"))
    if not reports:
        raise RuntimeError(f"No {args.input_version} schedule reports found in {run_dir}")

    summary = []
    for report_path in reports:
        name = report_path.name.replace(f"_{args.input_version}.schedule_report.json", "")
        motion_path = run_dir / f"{name}_{args.input_version}.npy"
        if not motion_path.is_file():
            print("[SKIP]", motion_path)
            continue
        raw = np.load(motion_path, allow_pickle=True).astype(np.float32)
        had_batch = raw.ndim == 3
        motion = raw[0] if had_batch else raw
        report = json.loads(report_path.read_text(encoding="utf-8"))
        schedule = report.get("schedule", [])
        queries = []
        for item in schedule:
            slot = item.get("v22_slot", item.get("v21_slot", {}))
            queries.append(slot.get("query", [0.0] * 12))
        refined, runtime = refine_motion_turns(
            motion,
            boundaries=report.get("boundaries", [0, len(motion)]),
            music_events=report.get("music_events", ["neutral_flow"]),
            queries=queries,
            bundle=bundle,
            device=device,
            fps=args.fps,
            threshold_ratio=args.threshold_ratio,
            window_len=args.window,
            context=args.context,
            strength=args.strength,
            max_events=args.max_events,
        )
        out_path = run_dir / f"{name}_{args.output_suffix}.npy"
        np.save(out_path, refined[None] if had_batch else refined)
        row = {"name": name, "input": str(motion_path), "output": str(out_path), "runtime": runtime}
        summary.append(row)
        print("[SAVED]", out_path, "events_refined=", runtime.get("events_refined", 0))

    output_report = run_dir / "V22_TURN_PACE_APPLY_REPORT.json"
    output_report.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "input_version": args.input_version,
                "output_suffix": args.output_suffix,
                "results": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("report:", output_report)


if __name__ == "__main__":
    main()
