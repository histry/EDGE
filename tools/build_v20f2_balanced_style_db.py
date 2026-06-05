#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EVENT_QUOTAS = {
    "arm_flourish": 1200,
    "high_tension": 1800,
    "support_shift": 1800,
    "build_up": 1000,
    "release": 1000,
    "calm_flow": 1600,
    "neutral_flow": 2600,
    "pose_hold": 300,
}


def get(item, key, default=0.0):
    if key in item:
        try:
            return float(item[key])
        except (TypeError, ValueError):
            pass

    desc = item.get("descriptor", {})
    if isinstance(desc, dict) and key in desc:
        try:
            return float(desc[key])
        except (TypeError, ValueError):
            pass

    return float(default)


def percentile(values, q):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def robust_norm(values):
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(arr, [10, 90])
    if hi - lo < 1e-8:
        return np.full_like(arr, 0.5)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--min_quality", type=float, default=0.88)
    ap.add_argument("--min_safety", type=float, default=0.84)
    ap.add_argument("--max_upper_torso_ratio", type=float, default=28.0)

    ap.add_argument("--max_total", type=int, default=9000)
    args = ap.parse_args()

    src = Path(args.input)
    data = json.loads(src.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) and "items" in data else data

    keys = [
        "upper_activity",
        "torso_activity",
        "lower_activity",
        "full_activity",
        "entry_activity",
        "exit_activity",
        "activity_peak",
        "style_tension",
        "smoothness",
        "jerk",
        "quality_score",
        "safety_score",
        "contact_switch",
        "root_y_range",
        "root_radius",
    ]

    arrays = {
        key: np.asarray([get(x, key) for x in items], dtype=np.float32)
        for key in keys
    }

    norm = {key: robust_norm(value) for key, value in arrays.items()}

    upper = arrays["upper_activity"]
    torso = arrays["torso_activity"]
    lower = arrays["lower_activity"]
    full = arrays["full_activity"]
    entry = arrays["entry_activity"]
    exit_ = arrays["exit_activity"]
    tension = arrays["style_tension"]
    smooth = arrays["smoothness"]
    jerk = arrays["jerk"]
    contact = arrays["contact_switch"]
    root_radius = arrays["root_radius"]

    thresholds = {
        "upper_q55": percentile(upper, 55),
        "upper_q70": percentile(upper, 70),
        "upper_q85": percentile(upper, 85),

        "torso_q30": percentile(torso, 30),
        "torso_q55": percentile(torso, 55),
        "torso_q70": percentile(torso, 70),

        "lower_q55": percentile(lower, 55),
        "lower_q75": percentile(lower, 75),

        "full_q25": percentile(full, 25),
        "full_q45": percentile(full, 45),
        "full_q65": percentile(full, 65),

        "tension_q45": percentile(tension, 45),
        "tension_q60": percentile(tension, 60),
        "tension_q82": percentile(tension, 82),

        "smooth_q55": percentile(smooth, 55),
        "contact_q75": percentile(contact, 75),
        "contact_q85": percentile(contact, 85),
        "jerk_q92": percentile(jerk, 92),
    }

    def classify(i):
        u = upper[i]
        t = torso[i]
        l = lower[i]
        f = full[i]
        st = tension[i]
        sm = smooth[i]
        cs = contact[i]
        ent = entry[i]
        ext = exit_[i]

        # 真实 contact 与下肢活动共同定义支撑事件
        if (
            cs >= thresholds["contact_q85"]
            and l >= thresholds["lower_q55"]
        ):
            return "support_shift"

        # 上肢展开必须伴随一定躯干参与，避免普通甩臂
        if (
            u >= thresholds["upper_q85"]
            and st >= thresholds["tension_q45"]
            and t >= thresholds["torso_q30"]
        ):
            return "arm_flourish"

        if (
            st >= thresholds["tension_q82"]
            and (
                u >= thresholds["upper_q55"]
                or t >= thresholds["torso_q55"]
            )
        ):
            return "high_tension"

        # 动作能量由低到高
        if (
            ext > ent * 1.15
            and f >= thresholds["full_q45"]
            and t >= thresholds["torso_q30"]
        ):
            return "build_up"

        # 动作能量由高到低
        if (
            ent > ext * 1.15
            and f >= thresholds["full_q45"]
            and sm >= thresholds["smooth_q55"]
        ):
            return "release"

        if f <= thresholds["full_q25"]:
            return "pose_hold"

        if (
            sm >= thresholds["smooth_q55"]
            and st <= thresholds["tension_q60"]
            and f <= thresholds["full_q65"]
        ):
            return "calm_flow"

        return "neutral_flow"

    candidates = []
    rejected = Counter()

    for i, item in enumerate(items):
        event_type = classify(i)

        q = arrays["quality_score"][i]
        sf = arrays["safety_score"][i]
        j = jerk[i]
        u = upper[i]
        t = torso[i]

        upper_torso_ratio = u / max(t, 1e-5)

        if q < args.min_quality:
            rejected["quality"] += 1
            continue

        if sf < args.min_safety:
            rejected["safety"] += 1
            continue

        if j > thresholds["jerk_q92"]:
            rejected["jerk"] += 1
            continue

        # 只对高上肢活动事件检查上肢/躯干失衡
        if (
            event_type in {
                "arm_flourish",
                "high_tension",
                "build_up",
                "neutral_flow",
            }
            and upper_torso_ratio > args.max_upper_torso_ratio
        ):
            rejected["upper_torso_ratio"] += 1
            continue

        movement_balance = min(
            t / max(u, 1e-5),
            1.0,
        )

        # 注意：这只是敦煌风格代理分数，不是真实风格分类器
        style_proxy = (
            0.24 * norm["style_tension"][i]
            + 0.20 * norm["torso_activity"][i]
            + 0.10 * norm["lower_activity"][i]
            + 0.10 * norm["smoothness"][i]
            + 0.12 * norm["quality_score"][i]
            + 0.10 * norm["safety_score"][i]
            + 0.08 * movement_balance
            + 0.06 * norm["contact_switch"][i]
            - 0.14 * norm["jerk"][i]
            - 0.06 * norm["root_radius"][i]
        )

        # calm/release 不应因为张力较低而全部被删除
        if event_type in {"arm_flourish", "high_tension", "build_up"}:
            style_proxy += 0.08 * norm["style_tension"][i]

        out = dict(item)
        out["event_type"] = event_type
        out.setdefault("descriptor", {})
        out["descriptor"]["event_type"] = event_type

        out["original_quality_score"] = float(q)
        out["original_visual_score"] = float(
            get(item, "visual_score", q)
        )
        out["dunhuang_style_proxy"] = float(style_proxy)
        out["upper_torso_ratio"] = float(upper_torso_ratio)

        # Scheduler 当前读取 quality_score / visual_score
        combined = (
            0.50 * float(q)
            + 0.22 * float(sf)
            + 0.28 * float(style_proxy)
        )

        out["quality_score"] = float(combined)
        out["visual_score"] = float(combined)

        candidates.append(out)

    # 每个事件类别分别排序，避免数据库再次退化成单一类别
    groups = defaultdict(list)

    for item in candidates:
        groups[item["event_type"]].append(item)

    selected = []

    for event_type, group in groups.items():
        group.sort(
            key=lambda x: (
                float(x.get("dunhuang_style_proxy", 0.0)),
                float(x.get("quality_score", 0.0)),
            ),
            reverse=True,
        )

        quota = EVENT_QUOTAS.get(event_type, 1000)
        selected.extend(group[:quota])

    selected.sort(
        key=lambda x: (
            float(x.get("dunhuang_style_proxy", 0.0)),
            float(x.get("quality_score", 0.0)),
        ),
        reverse=True,
    )

    selected = selected[:args.max_total]

    result = {
        "version": "v20f2_balanced_style_proxy",
        "source": str(src),
        "num_input": len(items),
        "num_after_gate": len(candidates),
        "num_selected": len(selected),
        "thresholds": thresholds,
        "event_type_counts_before_quota": dict(
            Counter(x["event_type"] for x in candidates)
        ),
        "event_type_counts": dict(
            Counter(x["event_type"] for x in selected)
        ),
        "rejected": dict(rejected),
        "items": selected,
    }

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("input:", len(items))
    print("after gate:", len(candidates))
    print("selected:", len(selected))
    print("event types:", result["event_type_counts"])
    print("rejected:", result["rejected"])
    print("saved:", dst)


if __name__ == "__main__":
    main()
