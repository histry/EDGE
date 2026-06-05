#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from tools.v20_motion_utils import load_motion_any, write_json
from tools.evaluate_dunhuang_motion import evaluate_motion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--boundaries", default="")
    args = ap.parse_args()
    m, _ = load_motion_any(args.motion)
    boundaries = [int(x) for x in args.boundaries.replace(";", ",").split(",") if x.strip()]
    ev = evaluate_motion(m, boundaries)
    write_json({"motion": args.motion, "bad_regions": ev["bad_regions"], "metrics": ev}, args.out_json)
    print(f"bad_regions={len(ev['bad_regions'])}")
    print(f"saved: {args.out_json}")

if __name__ == "__main__":
    main()
