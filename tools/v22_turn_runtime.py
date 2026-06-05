#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime inference for the V22 learned turn-pace refiner."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from model.v22_turn_pace import load_turn_pace_checkpoint
from tools.v22_turn_utils import (
    TurnEvent,
    allowed_yaw_speed_dps,
    detect_turn_events,
    project_rot6d_np,
    yaw_speed_dps_np,
)


def load_pace_bundle(path: str, device: torch.device | str = "cpu"):
    if not path:
        return None
    return load_turn_pace_checkpoint(path, device=device)


def slot_for_frame(
    frame: int,
    boundaries: Sequence[int],
    music_events: Sequence[str],
    queries: Sequence[Sequence[float]],
) -> Tuple[str, np.ndarray, int]:
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if int(start) <= int(frame) < int(end):
            event = str(music_events[idx]) if idx < len(music_events) else "neutral_flow"
            query = np.asarray(queries[idx], dtype=np.float32) if idx < len(queries) else np.zeros((12,), dtype=np.float32)
            return event, query, idx
    idx = max(0, len(boundaries) - 2)
    event = str(music_events[idx]) if idx < len(music_events) else "neutral_flow"
    query = np.asarray(queries[idx], dtype=np.float32) if idx < len(queries) else np.zeros((12,), dtype=np.float32)
    return event, query, idx


def extract_fixed_window(motion: np.ndarray, center: int, window_len: int) -> Tuple[np.ndarray, int, int]:
    half = window_len // 2
    start = int(center - half)
    end = start + window_len
    if start < 0:
        start = 0
        end = window_len
    if end > len(motion):
        end = len(motion)
        start = max(0, end - window_len)
    window = motion[start:end]
    if len(window) < window_len:
        pad = np.repeat(window[-1:], window_len - len(window), axis=0)
        window = np.concatenate([window, pad], axis=0)
    return window.astype(np.float32), int(start), int(end)


def build_edit_mask(window_len: int, event_start: int, event_end: int, context: int = 10) -> np.ndarray:
    lo = max(0, int(event_start) - int(context))
    hi = min(window_len - 1, int(event_end) + int(context))
    mask = np.zeros((window_len,), dtype=np.float32)
    if hi <= lo:
        return mask
    phase = np.linspace(0.0, 1.0, hi - lo + 1, dtype=np.float32)
    mask[lo : hi + 1] = np.sin(np.pi * phase) ** 2
    return mask


def normalize_condition(raw: np.ndarray, bundle) -> np.ndarray:
    lo = bundle.get("condition_lo")
    hi = bundle.get("condition_hi")
    raw = np.asarray(raw, dtype=np.float32)
    if lo is None or hi is None:
        return np.clip(raw, 0.0, 1.0).astype(np.float32)
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    return np.clip((raw - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def refine_motion_turns(
    motion: np.ndarray,
    boundaries: Sequence[int],
    music_events: Sequence[str],
    queries: Sequence[Sequence[float]],
    bundle,
    device: torch.device,
    fps: float = 30.0,
    threshold_ratio: float = 1.08,
    window_len: int = 72,
    context: int = 10,
    strength: float = 0.90,
    max_events: int = 4,
    min_peak_dps: float = 50.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Refine every excessively fast turn, regardless of temporal position."""
    x = np.asarray(motion, dtype=np.float32).copy()
    if bundle is None:
        return x, {"enabled": False, "reason": "no_checkpoint", "events": []}
    model = bundle["model"]
    model.eval()

    candidates = detect_turn_events(
        x,
        fps=fps,
        min_peak_dps=min_peak_dps,
        threshold_ratio=0.28,
        min_gap=max(12, window_len // 3),
        min_duration=3,
        max_events=None,
    )
    selected: List[Tuple[TurnEvent, str, np.ndarray, int, float, float]] = []
    for event in candidates:
        music_event, query, slot = slot_for_frame(event.peak_index, boundaries, music_events, queries)
        allowed = allowed_yaw_speed_dps(music_event, query)
        ratio = event.peak_speed_dps / max(allowed, 1e-6)
        if ratio >= float(threshold_ratio):
            selected.append((event, music_event, query, slot, allowed, ratio))
    selected.sort(key=lambda row: row[5], reverse=True)

    # Keep strongest non-overlapping windows.
    kept: List[Tuple[TurnEvent, str, np.ndarray, int, float, float]] = []
    for row in selected:
        event = row[0]
        if all(abs(event.peak_index - old[0].peak_index) >= window_len // 2 for old in kept):
            kept.append(row)
            if len(kept) >= int(max_events):
                break
    kept.sort(key=lambda row: row[0].peak_index)

    reports: List[Dict[str, Any]] = []
    for event, music_event, query, slot, allowed, excess_ratio in kept:
        window, start, end = extract_fixed_window(x, event.peak_index, window_len)
        local_start = int(np.clip(event.start - start, 1, window_len - 4))
        local_end = int(np.clip(event.end - start, local_start + 2, window_len - 2))
        mask = build_edit_mask(window_len, local_start, local_end, context=context)
        turn_phase = float(0.5 * (local_start + local_end) / max(window_len - 1, 1))
        raw_condition = np.concatenate(
            [
                np.asarray(query, dtype=np.float32).reshape(12),
                np.asarray(
                    [
                        float(event.peak_speed_dps),
                        float(allowed),
                        float(event.path_angle_deg),
                        float(excess_ratio),
                        float(turn_phase),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        condition = normalize_condition(raw_condition, bundle)
        adaptive_strength = float(np.clip(strength * (0.75 + 0.35 * (excess_ratio - 1.0)), 0.55, 1.0))

        before_speed = yaw_speed_dps_np(window, fps=fps)
        with torch.no_grad():
            pred = model(
                torch.from_numpy(window[None]).to(device),
                torch.from_numpy(mask[None]).to(device),
                torch.from_numpy(condition[None]).to(device),
                strength=adaptive_strength,
            )[0].detach().cpu().numpy().astype(np.float32)
        pred = project_rot6d_np(pred)

        valid_len = end - start
        x[start:end] = pred[:valid_len]
        after_window = x[start:end]
        after_speed = yaw_speed_dps_np(after_window, fps=fps)
        reports.append(
            {
                "slot": int(slot),
                "music_event": str(music_event),
                "peak_frame": int(event.peak_index),
                "window_start": int(start),
                "window_end": int(end),
                "turn_start": int(event.start),
                "turn_end": int(event.end),
                "turn_path_angle_deg": float(event.path_angle_deg),
                "allowed_yaw_speed_dps": float(allowed),
                "peak_before_dps": float(before_speed.max()) if len(before_speed) else 0.0,
                "peak_after_dps": float(after_speed.max()) if len(after_speed) else 0.0,
                "excess_ratio_before": float(excess_ratio),
                "strength": float(adaptive_strength),
            }
        )

    return x.astype(np.float32), {
        "enabled": True,
        "checkpoint_epoch": int(bundle.get("epoch", -1)),
        "checkpoint_val_loss": float(bundle.get("val_loss", float("inf"))),
        "threshold_ratio": float(threshold_ratio),
        "window_len": int(window_len),
        "events_detected": int(len(candidates)),
        "events_refined": int(len(reports)),
        "events": reports,
    }
