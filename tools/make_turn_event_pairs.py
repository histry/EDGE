#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turn_aware_event_utils import detect_turn_events, save_event_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="pair")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    event_npy = out.with_suffix(".event.npy")
    event_json = out.with_suffix(".event.json")
    rep = detect_turn_events(args.trajectory, seq_len=150, count=5)
    save_event_report(rep, event_json, event_npy)
    row = {"base": args.base, "target": args.target, "trajectory": args.trajectory, "event_features": str(event_npy), "count": 5}
    out.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ saved pairs jsonl: {out}")
    print(f"✅ saved event features: {event_npy}")


if __name__ == "__main__":
    main()
