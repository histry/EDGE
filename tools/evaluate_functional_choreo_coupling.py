#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from functional_choreo_metrics import functional_choreo_stats


def parse_paths(text: str):
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", required=True, help="Comma/semicolon separated .npy motions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectory", default="", help="Optional target trajectory 'x,z;...' for target-event metrics")
    args = ap.parse_args()

    paths = parse_paths(args.motions)
    if not paths:
        raise ValueError("--motions is empty")
    result = {}
    for p in paths:
        motion = np.load(p, allow_pickle=True)
        stats = functional_choreo_stats(motion, target_trajectory=args.trajectory or None)
        result[p] = stats
        print("\n" + "=" * 100)
        print(p)
        for k in [
            "root_path",
            "root_max_step",
            "lower_activity",
            "torso_activity",
            "upper_activity",
            "contact_switch",
            "support_expression_coupling",
            "turn_expression_response",
            "target_turn_expression_response",
            "target_support_lower_response",
            "target_expressive_response",
            "speed_lower_sync",
            "speed_torso_sync",
            "speed_expression_sync",
            "lower_torso_sync",
            "lower_upper_sync",
        ]:
            if k in stats:
                print(f"{k}: {stats[k]:.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ saved: {out}")


if __name__ == "__main__":
    main()
