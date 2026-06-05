#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrated phrase-level music-event encoder for V21.

修复：
1. 不再用 phrase 内 section.max() 直接判定 section_change；
2. 使用 section/accent 的持续密度，而不是单个峰值；
3. 使用 phrase 前后段 arousal/tension 趋势判断 build_up/release；
4. 离散事件只负责可解释标签，连续 12D query 仍负责细粒度路由。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


EVENT_DESIRED = {
    "accent": (0.88, 0.65, 0.30),
    "climax": (0.95, 0.78, 0.42),
    "section_change": (0.62, 0.72, 0.78),
    "build_up": (0.76, 0.70, 0.40),
    "release": (0.42, 0.52, 0.32),
    "calm_flow": (0.38, 0.55, 0.25),
    "neutral_flow": (0.55, 0.58, 0.40),
}


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else 0.0


def phrase_statistics(w: np.ndarray) -> Dict[str, float]:
    w = np.asarray(w, dtype=np.float32)

    if w.ndim != 2 or w.shape[1] < 12:
        raise ValueError(f"Expected phrase features [T,12+], got {w.shape}")

    n = len(w)
    split = max(1, n // 3)

    arousal = w[:, 4]
    tension = w[:, 6]

    arousal_start = _mean(arousal[:split])
    arousal_end = _mean(arousal[-split:])
    tension_start = _mean(tension[:split])
    tension_end = _mean(tension[-split:])

    return {
        "energy": _mean(w[:, 0]),
        "onset": _mean(w[:, 1]),
        "beat": _mean(w[:, 2]),
        "beat_density": float(np.mean(w[:, 2] > 0.50)),
        "arousal": _mean(arousal),
        "delta_mean": _mean(w[:, 5]),
        "arousal_trend": arousal_end - arousal_start,
        "tension": _mean(tension),
        "tension_trend": tension_end - tension_start,
        "calm": _mean(w[:, 7]),
        "novelty": _mean(w[:, 8]),
        "brightness": _mean(w[:, 9]),
        "section_mean": _mean(w[:, 10]),
        "section_density": float(np.mean(w[:, 10] > 0.75)),
        "accent_mean": _mean(w[:, 11]),
        "accent_density": float(np.mean(w[:, 11] > 0.70)),
    }


def classify_phrase_event(w: np.ndarray) -> Tuple[str, Dict[str, float]]:
    s = phrase_statistics(w)

    # 平静段优先识别，避免局部 novelty 峰值把整段误判为 section_change。
    if (
        s["calm"] >= 0.62
        and s["tension"] <= 0.56
        and s["arousal"] <= 0.58
    ):
        return "calm_flow", s

    # 高能量和高张力同时出现，才定义为高潮。
    if (
        s["arousal"] >= 0.67
        and s["tension"] >= 0.62
        and (
            s["accent_density"] >= 0.06
            or s["energy"] >= 0.55
        )
    ):
        return "climax", s

    # 使用 phrase 前后趋势，而不是整段 delta 均值。
    if (
        s["arousal_trend"] <= -0.055
        or s["tension_trend"] <= -0.055
        or s["delta_mean"] <= -0.012
    ):
        return "release", s

    if (
        s["arousal_trend"] >= 0.055
        or s["tension_trend"] >= 0.055
        or s["delta_mean"] >= 0.012
    ):
        return "build_up", s

    # section_change 必须在整个 phrase 内具有持续性，
    # 单个 novelty/section 峰值不能触发。
    if (
        s["section_density"] >= 0.16
        and s["section_mean"] >= 0.45
        and s["novelty"] >= 0.42
    ):
        return "section_change", s

    if (
        s["accent_density"] >= 0.16
        or (
            s["accent_mean"] >= 0.52
            and s["beat_density"] >= 0.08
        )
    ):
        return "accent", s

    return "neutral_flow", s


def build_phrase_query(
    w: np.ndarray,
    start: int,
    end: int,
) -> Tuple[np.ndarray, str]:
    event, s = classify_phrase_event(w)

    upper, torso, lower = EVENT_DESIRED[event]

    positive_trend = max(
        s["arousal_trend"],
        s["tension_trend"],
        0.0,
    )
    negative_trend = max(
        -s["arousal_trend"],
        -s["tension_trend"],
        0.0,
    )

    query = np.asarray(
        [
            np.clip(s["arousal"], 0.0, 1.0),
            upper,
            torso,
            lower,
            np.clip(s["tension"], 0.0, 1.0),
            np.clip(s["calm"], 0.0, 1.0),
            np.clip(
                0.60 * s["section_mean"]
                + 0.40 * s["beat"],
                0.0,
                1.0,
            ),
            np.clip(positive_trend * 6.0, 0.0, 1.0),
            np.clip(negative_trend * 6.0, 0.0, 1.0),
            np.clip(s["accent_mean"], 0.0, 1.0),
            np.clip(s["novelty"], 0.0, 1.0),
            np.clip((end - start) / 60.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    return query, event
