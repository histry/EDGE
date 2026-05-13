#!/usr/bin/env python3
"""Apply Turn-aware Event Refiner v2 to a no-train anchor motion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from turn_aware_event_refiner import load_refiner
from turn_aware_event_utils import TurnEventConfig, event_feature_matrix


def load_motion(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {arr.shape}: {path}")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--anchor", required=True, help="No-train turn-aware result. Refiner adds a small delta on this.")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg, feature_names, extra = load_refiner(args.ckpt, map_location=args.device)
    model.to(args.device).eval()

    base = load_motion(args.base)
    anchor = load_motion(args.anchor)
    T = min(len(base), len(anchor))
    ev_cfg = TurnEventConfig.from_env(seq_len=T, count=5)
    event, names, ev_report = event_feature_matrix(args.trajectory, ev_cfg)
    T = min(T, len(event))

    with torch.no_grad():
        pred = model(
            torch.from_numpy(base[:T]).float().to(args.device),
            torch.from_numpy(anchor[:T]).float().to(args.device),
            torch.from_numpy(event[:T]).float().to(args.device),
        )[0].detach().cpu().numpy().astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred)
    report = {
        "ckpt": args.ckpt,
        "base": args.base,
        "anchor": args.anchor,
        "trajectory": args.trajectory,
        "out": str(out),
        "config": cfg.__dict__,
        "feature_names": names,
        "event_report": {
            "event_centers": ev_report["event_centers"],
            "support_frames": ev_report["support_frames"],
            "expressive_frames": ev_report["expressive_frames"],
        },
    }
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ saved refined motion: {out}")
    print(f"   report={report_path}")


if __name__ == "__main__":
    main()
