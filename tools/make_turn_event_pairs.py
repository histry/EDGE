#!/usr/bin/env python3
"""Create JSONL training pairs for Turn-aware Event Refiner v2.

Each row links:
  base motion      = weak Text/Pose RAG base
  anchor motion    = no-train turn-aware functional compositor output
  target motion    = one pseudo target, e.g. f35 manual, f40 manual, turn_event, mild
  trajectory       = trajectory string used to build event features

This intentionally supports multiple pseudo targets so the refiner learns a
small event-conditioned correction distribution instead of overfitting one
single compositor result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from turn_aware_event_utils import TurnEventConfig, event_feature_matrix, save_event_report


def split_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_weights(text: str, n: int) -> List[float]:
    if not text:
        return [1.0] * n
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) == 1 and n > 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"--weights expects 1 or {n} values, got {len(vals)}")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--anchor", required=True, help="No-train turn-aware output; refiner predicts small delta on top of this.")
    ap.add_argument("--targets", required=True, help="Comma-separated pseudo target motions.")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--seq_len", type=int, default=150)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--weights", default="", help="Optional comma-separated target weights.")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_event_npy", default="")
    ap.add_argument("--out_event_report", default="")
    args = ap.parse_args()

    targets = split_csv(args.targets)
    weights = parse_weights(args.weights, len(targets))

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cfg = TurnEventConfig.from_env(seq_len=args.seq_len, count=args.count)
    event, names, ev_report = event_feature_matrix(args.trajectory, cfg)

    if args.out_event_npy:
        p = Path(args.out_event_npy)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, event.astype(np.float32))
        print(f"✅ saved event features: {p}")

    if args.out_event_report:
        save_event_report(args.out_event_report, ev_report)
        print(f"✅ saved event report: {args.out_event_report}")

    rows = []
    for target, weight in zip(targets, weights):
        if not Path(target).exists():
            raise FileNotFoundError(target)
        rows.append(
            {
                "base": args.base,
                "anchor": args.anchor,
                "target": target,
                "trajectory": args.trajectory,
                "seq_len": args.seq_len,
                "count": args.count,
                "weight": float(weight),
                "feature_names": names,
                "event_report": {
                    "event_centers": ev_report["event_centers"],
                    "support_frames": ev_report["support_frames"],
                    "expressive_frames": ev_report["expressive_frames"],
                },
            }
        )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ saved pairs jsonl: {out_jsonl}")
    print(f"   pairs={len(rows)}")
    print(f"   targets={targets}")


if __name__ == "__main__":
    main()
