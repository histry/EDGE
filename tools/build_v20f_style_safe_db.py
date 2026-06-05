#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def get(item, key, default=0.0):
    if key in item:
        try:
            return float(item[key])
        except Exception:
            pass

    desc = item.get("descriptor", {})
    if isinstance(desc, dict) and key in desc:
        try:
            return float(desc[key])
        except Exception:
            pass

    return float(default)


def robust_norm(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(values, [10, 90])
    return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min_quality", type=float, default=0.92)
    ap.add_argument("--min_safety", type=float, default=0.88)
    ap.add_argument("--min_style_tension", type=float, default=0.40)
    ap.add_argument("--min_torso_activity", type=float, default=0.0045)
    ap.add_argument("--max_jerk", type=float, default=0.24)
    ap.add_argument("--max_upper_torso_ratio", type=float, default=18.0)
    args = ap.parse_args()

    src = Path(args.input)
    data = json.loads(src.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) and "items" in data else data

    upper = np.asarray([get(x, "upper_activity") for x in items])
    torso = np.asarray([get(x, "torso_activity") for x in items])
    lower = np.asarray([get(x, "lower_activity") for x in items])
    tension = np.asarray([get(x, "style_tension") for x in items])
    jerk = np.asarray([get(x, "jerk") for x in items])
    quality = np.asarray([get(x, "quality_score") for x in items])
    safety = np.asarray([get(x, "safety_score") for x in items])

    upper_n = robust_norm(upper)
    torso_n = robust_norm(torso)
    lower_n = robust_norm(lower)
    tension_n = robust_norm(tension)
    jerk_n = robust_norm(jerk)
    quality_n = robust_norm(quality)
    safety_n = robust_norm(safety)

    kept = []
    rejected = Counter()

    for i, item in enumerate(items):
        et = str(item.get("event_type", "neutral_flow"))

        u = upper[i]
        t = torso[i]
        st = tension[i]
        j = jerk[i]
        q = quality[i]
        sf = safety[i]

        ratio = u / max(t, 1e-5)

        if et == "pose_hold":
            rejected["pose_hold"] += 1
            continue
        if q < args.min_quality:
            rejected["quality"] += 1
            continue
        if sf < args.min_safety:
            rejected["safety"] += 1
            continue
        if st < args.min_style_tension:
            rejected["style_tension"] += 1
            continue
        if t < args.min_torso_activity:
            rejected["torso_activity"] += 1
            continue
        if j > args.max_jerk:
            rejected["jerk"] += 1
            continue
        if ratio > args.max_upper_torso_ratio:
            rejected["upper_torso_ratio"] += 1
            continue

        # 临时 style proxy：
        # 强调姿态张力、躯干参与、质量和支撑；
        # 惩罚高 jerk 和只有手臂没有躯干的动作。
        balance = min(t / max(u, 1e-5), 1.0)

        style_score = (
            0.30 * tension_n[i]
            + 0.22 * torso_n[i]
            + 0.12 * lower_n[i]
            + 0.16 * quality_n[i]
            + 0.10 * safety_n[i]
            + 0.10 * balance
            - 0.18 * jerk_n[i]
        )

        out = dict(item)
        out["original_quality_score"] = float(q)
        out["dunhuang_style_proxy"] = float(style_score)
        out["upper_torso_ratio"] = float(ratio)

        # 让现有 scheduler 优先读取风格安全后的综合分数
        combined = 0.60 * q + 0.40 * float(style_score)
        out["quality_score"] = float(combined)
        out["visual_score"] = float(combined)

        kept.append(out)

    kept.sort(
        key=lambda x: (
            float(x.get("dunhuang_style_proxy", 0.0)),
            float(x.get("quality_score", 0.0)),
        ),
        reverse=True,
    )

    result = {
        "version": "v20f_style_safe_proxy",
        "source": str(src),
        "items": kept,
        "event_type_counts": dict(
            Counter(str(x.get("event_type", "unknown")) for x in kept)
        ),
        "rejected": dict(rejected),
        "num_input": len(items),
        "num_kept": len(kept),
    }

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("input:", len(items))
    print("kept:", len(kept))
    print("event types:", result["event_type_counts"])
    print("rejected:", result["rejected"])
    print("saved:", dst)


if __name__ == "__main__":
    main()
