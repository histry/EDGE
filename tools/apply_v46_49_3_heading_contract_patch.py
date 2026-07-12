#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# V46.49.3 absolute-heading contract patch.
from __future__ import annotations

import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "chang_e_edge_retarget.py"
MARKER = "# ===== V46.49.3 ABSOLUTE HEADING CONTRACT ====="

HELPER = r'''
# ===== V46.49.3 ABSOLUTE HEADING CONTRACT =====
def _v46_49_3_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _v46_49_3_moving_average(x: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if size <= 1:
        return x.copy()
    pad = size // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(size, dtype=np.float32) / float(size)
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def _v46_49_3_moving_median(x: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if median_filter is not None:
        return median_filter(x, size=size, mode="nearest").astype(np.float32)
    return _v46_49_3_moving_average(x, size)


def _v46_49_3_runs(mask: np.ndarray):
    m = np.asarray(mask, dtype=bool)
    if not m.size:
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _v46_49_3_body_yaw(
    positions: np.ndarray,
    mapping: Dict[int, int],
) -> np.ndarray:
    p = np.asarray(positions, dtype=np.float32)
    pelvis = p[:, mapping[0]]
    lhip = p[:, mapping[1]]
    rhip = p[:, mapping[2]]
    neck = p[:, mapping[12]]

    vectors = []
    hip_right = rhip - lhip
    vectors.append(
        hip_right / np.maximum(
            np.linalg.norm(hip_right, axis=-1, keepdims=True), 1e-8
        )
    )
    if 13 in mapping and 14 in mapping:
        shoulder_right = p[:, mapping[14]] - p[:, mapping[13]]
        vectors.append(
            shoulder_right / np.maximum(
                np.linalg.norm(shoulder_right, axis=-1, keepdims=True), 1e-8
            )
        )

    right = np.mean(np.stack(vectors, axis=0), axis=0)
    right /= np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8)

    up = neck - pelvis
    up -= np.sum(up * right, axis=-1, keepdims=True) * right
    up /= np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-8)

    forward = np.cross(right, up)
    forward /= np.maximum(
        np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8
    )
    return np.unwrap(
        np.arctan2(forward[:, 0], forward[:, 2])
    ).astype(np.float32)


def _v46_49_3_heading_metrics(
    yaw: np.ndarray,
    fps: float,
    min_rate_deg_s: float,
) -> dict:
    yaw = np.asarray(yaw, dtype=np.float32)
    rate = (
        np.gradient(yaw) * float(fps)
        if len(yaw) > 1
        else np.zeros_like(yaw)
    )
    rate_deg = np.degrees(rate)
    active = np.abs(rate_deg) >= float(min_rate_deg_s)
    longest = max(
        (b - a for a, b in _v46_49_3_runs(active)),
        default=0,
    )
    return {
        "net_turns": float((yaw[-1] - yaw[0]) / (2 * np.pi))
        if len(yaw) else 0.0,
        "absolute_turns": float(
            np.sum(np.abs(np.diff(yaw))) / (2 * np.pi)
        ) if len(yaw) > 1 else 0.0,
        "yaw_speed_deg_s_p50": float(
            np.percentile(np.abs(rate_deg), 50)
        ) if len(rate_deg) else 0.0,
        "yaw_speed_deg_s_p95": float(
            np.percentile(np.abs(rate_deg), 95)
        ) if len(rate_deg) else 0.0,
        "yaw_speed_deg_s_max": float(
            np.max(np.abs(rate_deg))
        ) if len(rate_deg) else 0.0,
        "active_turn_ratio": float(active.mean())
        if len(active) else 0.0,
        "longest_active_turn_seconds": float(
            longest / max(float(fps), 1e-8)
        ),
    }


def stabilize_source_heading_positions(
    positions: np.ndarray,
    mapping: Dict[int, int],
    fps: float,
) -> Tuple[np.ndarray, dict]:
    x = np.asarray(positions, dtype=np.float32).copy()
    mode = str(
        os.environ.get("V46_49_HEADING_MODE", "stabilize")
    ).strip().lower()
    if mode not in {"stabilize", "raw", "lock"}:
        raise ValueError(
            "V46_49_HEADING_MODE must be stabilize/raw/lock, "
            f"got {mode!r}"
        )

    raw_yaw = _v46_49_3_body_yaw(x, mapping)

    smooth_seconds = _v46_49_3_env_float(
        "V46_49_HEADING_SMOOTH_SECONDS", 0.45
    )
    baseline_seconds = _v46_49_3_env_float(
        "V46_49_HEADING_BASELINE_SECONDS", 4.0
    )
    min_rate_deg_s = _v46_49_3_env_float(
        "V46_49_HEADING_MIN_DRIFT_DEG_S", 7.0
    )
    min_persist_seconds = _v46_49_3_env_float(
        "V46_49_HEADING_MIN_PERSIST_SECONDS", 3.0
    )
    consistency_min = _v46_49_3_env_float(
        "V46_49_HEADING_SIGN_CONSISTENCY", 0.82
    )
    max_variation_deg_s = _v46_49_3_env_float(
        "V46_49_HEADING_MAX_BASELINE_VARIATION_DEG_S", 14.0
    )
    max_correction_deg_s = _v46_49_3_env_float(
        "V46_49_HEADING_MAX_CORRECTION_DEG_S", 60.0
    )

    smooth_n = max(3, int(round(smooth_seconds * fps)))
    baseline_n = max(5, int(round(baseline_seconds * fps)))
    min_persist_n = max(2, int(round(min_persist_seconds * fps)))

    yaw_smooth = _v46_49_3_moving_average(raw_yaw, smooth_n)
    yaw_rate = np.gradient(yaw_smooth) * float(fps)
    baseline = _v46_49_3_moving_median(yaw_rate, baseline_n)

    same_sign = (
        np.sign(yaw_rate) == np.sign(baseline)
    ).astype(np.float32)
    consistency = _v46_49_3_moving_average(
        same_sign, baseline_n
    )
    variation = _v46_49_3_moving_average(
        np.abs(yaw_rate - baseline), baseline_n
    )

    candidate = (
        (np.abs(np.degrees(baseline)) >= min_rate_deg_s)
        & (consistency >= consistency_min)
        & (np.degrees(variation) <= max_variation_deg_s)
    )

    persistent = np.zeros(len(x), dtype=bool)
    longest_candidate = 0
    for a, b in _v46_49_3_runs(candidate):
        longest_candidate = max(longest_candidate, b - a)
        if b - a >= min_persist_n:
            persistent[a:b] = True

    if mode == "raw":
        correction = np.zeros_like(raw_yaw)
        drift_rate = np.zeros_like(raw_yaw)
    elif mode == "lock":
        correction = raw_yaw - raw_yaw[:1]
        drift_rate = np.gradient(correction) * float(fps)
        persistent[:] = True
    else:
        max_rate = np.deg2rad(max_correction_deg_s)
        drift_rate = np.where(
            persistent,
            np.clip(baseline, -max_rate, max_rate),
            0.0,
        ).astype(np.float32)
        drift_rate = _v46_49_3_moving_average(
            drift_rate,
            max(3, int(round(0.75 * fps))),
        )
        correction = np.cumsum(
            drift_rate / float(fps)
        ).astype(np.float32)
        correction -= correction[:1]

    pelvis = x[:, mapping[0]].copy()
    rel = x - pelvis[:, None, :]
    theta = -correction
    c, s = np.cos(theta), np.sin(theta)
    old_x = rel[..., 0].copy()
    old_z = rel[..., 2].copy()
    rel[..., 0] = c[:, None] * old_x + s[:, None] * old_z
    rel[..., 2] = -s[:, None] * old_x + c[:, None] * old_z
    corrected = rel + pelvis[:, None, :]

    corrected_yaw = _v46_49_3_body_yaw(corrected, mapping)
    report = {
        "version": "v46_49_3_absolute_heading_contract",
        "mode": mode,
        "persistent_drift_ratio": float(persistent.mean()),
        "longest_candidate_drift_seconds": float(
            longest_candidate / max(float(fps), 1e-8)
        ),
        "removed_turns": float(
            (correction[-1] - correction[0]) / (2 * np.pi)
        ) if len(correction) else 0.0,
        "correction_speed_deg_s_p95": float(
            np.percentile(np.abs(np.degrees(drift_rate)), 95)
        ) if len(drift_rate) else 0.0,
        "raw": _v46_49_3_heading_metrics(
            raw_yaw, fps, min_rate_deg_s
        ),
        "corrected": _v46_49_3_heading_metrics(
            corrected_yaw, fps, min_rate_deg_s
        ),
    }
    return corrected.astype(np.float32), report
# ===== V46.49.3 ABSOLUTE HEADING CONTRACT END =====
'''

OLD_PIPELINE = '''    aligned = apply_similarity(native_pos, scale, basis_R, trans)
    aligned = resample_global_positions(aligned, bvh.fps, cfg.target_fps)
    motion, fit_report = fit_target_motion(aligned, mapping, cfg)
'''

NEW_PIPELINE = '''    aligned = apply_similarity(native_pos, scale, basis_R, trans)
    aligned = resample_global_positions(aligned, bvh.fps, cfg.target_fps)
    aligned, heading_report = stabilize_source_heading_positions(
        aligned,
        mapping,
        float(cfg.target_fps),
    )
    motion, fit_report = fit_target_motion(aligned, mapping, cfg)
    fit_report["heading_contract"] = heading_report
'''


def main() -> int:
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print("[SKIP] V46.49.3 heading patch already applied")
        return 0
    if OLD_PIPELINE not in text:
        raise RuntimeError(
            "Cannot locate V46.49.2 retarget pipeline"
        )

    anchor = "def fit_target_motion(\n"
    pos = text.find(anchor)
    if pos < 0:
        raise RuntimeError("Cannot locate def fit_target_motion")

    backup = TARGET.with_suffix(
        TARGET.suffix
        + f".v46_49_3_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(TARGET, backup)
    print("[BACKUP]", backup)

    text = text[:pos] + HELPER + "\n\n" + text[pos:]
    text = text.replace(OLD_PIPELINE, NEW_PIPELINE, 1)
    TARGET.write_text(text, encoding="utf-8")

    print("[DONE] V46.49.3 heading contract patched")
    print("[FORMAL] export V46_49_HEADING_MODE=stabilize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
