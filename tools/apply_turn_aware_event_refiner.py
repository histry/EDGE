#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from turn_aware_event_refiner import load_checkpoint
from turn_aware_event_utils import detect_turn_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base = np.load(args.base, allow_pickle=True).astype(np.float32)
    rep = detect_turn_events(args.trajectory, seq_len=len(base), count=5)
    event = rep["event_features"].astype(np.float32)
    model = load_checkpoint(args.checkpoint, map_location=args.device).to(args.device).eval()
    with torch.no_grad():
        x = torch.from_numpy(base)[None].to(args.device)
        e = torch.from_numpy(event)[None].to(args.device)
        pred, residual = model(x, e)
    pred_np = pred[0].detach().cpu().numpy().astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred_np)
    report = {
        "checkpoint": args.checkpoint,
        "base": args.base,
        "out": str(out),
        "trajectory": args.trajectory,
        "event_centers": rep["event_centers"],
        "support_frames": rep["support_frames"],
        "expressive_frames": rep["expressive_frames"],
    }
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ saved refined motion: {out}")
    print(f"   report={out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
