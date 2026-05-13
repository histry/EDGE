#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from functional_choreo_metrics import functional_choreo_stats


def load_motion(path):
    x = np.load(path, allow_pickle=True)
    if x.ndim == 0 and isinstance(x.item(), dict):
        d = x.item()
        x = d.get("motion", d.get("pose", x))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != 151:
        raise ValueError(f"{path}: expected [T,151], got {x.shape}")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", required=True, help="comma-separated .npy files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [x.strip() for x in args.motions.replace(";", ",").split(",") if x.strip()]
    rows = {}
    for p in paths:
        if not Path(p).exists():
            rows[p] = {"error": "missing"}
            continue
        rows[p] = functional_choreo_stats(load_motion(p))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for p, r in rows.items():
        print("\n" + "=" * 100)
        print(p)
        if "error" in r:
            print("ERROR:", r["error"])
            continue
        keys = [
            "root_path",
            "root_max_step",
            "lower_activity",
            "torso_activity",
            "upper_activity",
            "contact_switch",
            "support_expression_coupling",
            "turn_expression_response",
            "speed_lower_sync",
            "speed_torso_sync",
            "speed_expression_sync",
            "lower_torso_sync",
            "lower_upper_sync",
        ]
        for k in keys:
            print(f"{k}: {r.get(k, 0.0):.6f}")

    print(f"\n✅ saved: {out}")


if __name__ == "__main__":
    main()
