#!/usr/bin/env python3
"""Onset-driven, physics-first long prior composer for EDGE-Dunhuang.

This implements the most suitable scheme for the current project state:
  - Onset proposes phrase-change times.
  - Physics-aware boundary matching has hard priority over exact music hits.
  - Candidate prior retrieval uses physical compatibility first, music/quality second.
  - Dynamic transition tolerance scans nearby frames and may delay a boundary.
  - Distinct unit ids are preferred to avoid p4/unit9415 repeated phrase reset.
  - Output is a 150-frame root-stable stitched prior plus reports.

It does not replace the diffusion model. It gives the model / renderer / later
SDEdit step a much safer long temporal prior.

Example:
  python generate_onset_phrase_prior.py \
    --pool data/dunhuang_choreo_unit_rag/v15_physics_prior_pool.npz \
    --audio_wav test_music_bank/dunhuangwu2.wav \
    --out output/v15_onset_phrase/dw2_onset_phrase_prior.npy \
    --length 150 --fps 30 --inplace
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from choreorag_physics_state import (
    ROT_START,
    ROOT_X_IDX,
    ROOT_Z_IDX,
    TransitionWeights,
    blend_motions,
    boundary_state,
    local_jump_metrics,
    save_motion_pkl,
    transition_distance,
    write_json,
)


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    lo, hi = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if window <= 1 or len(x) < 3:
        return x
    if window % 2 == 0:
        window += 1
    window = min(window, len(x) if len(x) % 2 else len(x) - 1)
    if window <= 1:
        return x
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)


def load_audio_curve(audio_wav: Optional[str], audio_feature: Optional[str], length: int, fps: int) -> Tuple[np.ndarray, Dict]:
    """Return onset/energy curve at target motion FPS length."""
    meta: Dict = {"source": "none", "method": "fixed_fallback"}
    curve = None

    if audio_feature:
        arr = np.load(audio_feature, allow_pickle=True)
        if arr.ndim == 0 and isinstance(arr.item(), dict):
            d = arr.item()
            for key in ("feature", "audio", "audio_feat", "arr_0"):
                if key in d:
                    arr = d[key]
                    break
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"audio_feature must be [T,C], got {arr.shape}")
        if arr.shape[1] > 768:
            onset = arr[:, 768]
        elif arr.shape[1] >= 35:
            onset = arr[:, 0] + 0.5 * arr[:, -2] + 0.5 * arr[:, -1]
        else:
            onset = np.zeros((arr.shape[0],), dtype=np.float32)
            onset[1:] = np.linalg.norm(arr[1:] - arr[:-1], axis=-1)
        curve = _normalize01(_smooth(np.maximum(onset, 0.0), 5))
        meta = {"source": str(audio_feature), "method": "feature_onset"}

    elif audio_wav:
        try:
            import librosa
            y, sr = librosa.load(audio_wav, sr=None, mono=True)
            hop = max(1, int(sr / fps))
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
            rms = librosa.feature.rms(y=y, frame_length=max(512, 2 * hop), hop_length=hop)[0]
            n = min(len(onset_env), len(rms))
            curve = _normalize01(0.70 * _normalize01(onset_env[:n]) + 0.30 * _normalize01(rms[:n]))
            meta = {"source": str(audio_wav), "method": "librosa_onset_rms", "sr": int(sr), "hop": int(hop)}
        except Exception as exc:
            print(f"⚠️ Failed to extract audio onset from wav ({exc}); using fixed fallback.")

    if curve is None or len(curve) == 0:
        curve = np.zeros((length,), dtype=np.float32)
        # Conservative fallback: phrase every ~35-40 frames.
        for p in range(35, length, 35):
            curve[p] = 1.0
        curve = _smooth(curve, 3)

    x_old = np.linspace(0.0, 1.0, len(curve), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, int(length), dtype=np.float32)
    curve = np.interp(x_new, x_old, curve).astype(np.float32)
    return _normalize01(curve), meta


def pick_onsets(curve: np.ndarray, length: int, min_gap: int = 24, threshold: float = 0.55, max_phrases: int = 4) -> List[Tuple[int, float]]:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    candidates = []
    for i in range(1, len(curve) - 1):
        if curve[i] >= threshold and curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1]:
            if 8 <= i < length - 8:
                candidates.append((i, float(curve[i])))
    # Highest peaks first, then restore temporal order while respecting min_gap.
    candidates.sort(key=lambda x: x[1], reverse=True)
    picked: List[Tuple[int, float]] = []
    for frame, strength in candidates:
        if all(abs(frame - p[0]) >= min_gap for p in picked):
            picked.append((int(frame), float(strength)))
        if len(picked) >= max(1, max_phrases - 1):
            break
    picked.sort(key=lambda x: x[0])
    # Ensure at least 3 boundaries for 150-frame demo if audio is too flat.
    fallback = [35, 70, 105]
    for f in fallback:
        if len(picked) >= max(1, max_phrases - 1):
            break
        if 8 <= f < length - 8 and all(abs(f - p[0]) >= min_gap for p in picked):
            picked.append((f, float(curve[min(f, len(curve) - 1)])))
    picked.sort(key=lambda x: x[0])
    return picked[: max(1, max_phrases - 1)]


def load_pool(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ["motions", "entry_state", "unit_ids"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Prior pool missing keys {missing}; build it with tools/build_physics_aware_prior_pool.py")
    motions = np.asarray(data["motions"], dtype=np.float32)
    entry = np.asarray(data["entry_state"], dtype=np.float32)
    unit_ids = np.array([str(x) for x in data["unit_ids"].reshape(-1)], dtype=object)
    scores = np.asarray(data["base_scores"], dtype=np.float32).reshape(-1) if "base_scores" in data.files else np.ones(len(motions), dtype=np.float32) * 0.5
    if motions.ndim != 3 or motions.shape[-1] != 151:
        raise ValueError(f"pool motions must be [N,T,151], got {motions.shape}")
    return {"motions": motions, "entry": entry, "unit_ids": unit_ids, "scores": scores}


def music_affinity_for_candidate(idx: int, onset_strength: float, scores: np.ndarray) -> float:
    # We do not know exact per-unit audio score here. Use quality score as a weak
    # proxy and let physical compatibility dominate.
    q = float(scores[idx]) if idx < len(scores) else 0.5
    return float((0.3 + 0.7 * onset_strength) * q)


def choose_first_unit(pool: Dict[str, np.ndarray], used: set, prefer_high_activity: bool = True) -> int:
    motions, scores = pool["motions"], pool["scores"]
    activities = []
    for m in motions:
        if len(m) < 2:
            activities.append(0.0)
        else:
            activities.append(float(np.linalg.norm(m[1:, ROT_START:] - m[:-1, ROT_START:], axis=-1).mean()))
    activities = _normalize01(np.asarray(activities, dtype=np.float32))
    final = 0.65 * _normalize01(scores) + 0.35 * activities
    order = np.argsort(-final)
    for idx in order:
        if str(pool["unit_ids"][idx]) not in used:
            return int(idx)
    return int(order[0])


def choose_candidate(
    pool: Dict[str, np.ndarray],
    current_motion: np.ndarray,
    desired_frame: int,
    onset_strength: float,
    used: set,
    weights: TransitionWeights,
    tolerance: int,
    min_remaining: int,
) -> Tuple[int, int, Dict]:
    motions, entry, unit_ids, scores = pool["motions"], pool["entry"], pool["unit_ids"], pool["scores"]
    best = None
    # Scan a tolerance window AFTER the onset. This intentionally sacrifices exact
    # hit timing to find a safer physical transition frame.
    start = max(1, min(int(desired_frame), len(current_motion) - 2))
    end = min(len(current_motion) - 1, start + max(0, int(tolerance)))
    for cut in range(start, end + 1):
        cur_state = boundary_state(current_motion, cut)
        for idx in range(len(motions)):
            dist, parts = transition_distance(cur_state, entry[idx], weights, return_parts=True)
            if not np.isfinite(dist):
                continue
            uid = str(unit_ids[idx])
            novelty_bonus = 0.0 if uid in used else 1.0
            music = music_affinity_for_candidate(idx, onset_strength, scores)
            score = (
                weights.alpha_music * music
                + weights.quality * float(scores[idx])
                + weights.novelty * novelty_bonus
                - weights.beta_physics * float(dist)
            )
            # Prefer candidates that can contribute enough frames.
            if len(motions[idx]) < min_remaining:
                score -= 0.10
            item = (score, idx, cut, dist, parts, music, novelty_bonus)
            if best is None or item[0] > best[0]:
                best = item
    if best is not None:
        score, idx, cut, dist, parts, music, novelty_bonus = best
        return int(idx), int(cut), {
            "score": float(score),
            "dist": float(dist),
            "music_affinity": float(music),
            "novelty_bonus": float(novelty_bonus),
            "parts": parts,
            "fallback": False,
        }

    # Safe fallback: choose best quality + novelty unit and cut at the latest safe
    # point. This keeps the system outputting something rather than failing hard.
    order = np.argsort(-scores)
    for idx in order:
        if str(unit_ids[idx]) not in used:
            return int(idx), int(end), {
                "score": float(scores[idx]),
                "dist": float("inf"),
                "music_affinity": 0.0,
                "novelty_bonus": 1.0,
                "parts": {"hard_fail": 1.0},
                "fallback": True,
            }
    return int(order[0]), int(end), {
        "score": float(scores[order[0]]),
        "dist": float("inf"),
        "music_affinity": 0.0,
        "novelty_bonus": 0.0,
        "parts": {"hard_fail": 1.0},
        "fallback": True,
    }


def blend_window_from_strength(strength: float, min_blend: int, max_blend: int) -> int:
    s = float(np.clip(strength, 0.0, 1.0))
    # Strong onset = quicker but still safe; weak onset = longer blend.
    return int(round(max_blend - s * (max_blend - min_blend)))


def compose_phrase(args) -> Tuple[np.ndarray, Dict]:
    pool = load_pool(args.pool)
    curve, audio_meta = load_audio_curve(args.audio_wav, args.audio_feature, args.length, args.fps)
    onset_events = pick_onsets(curve, args.length, args.min_onset_gap, args.onset_threshold, args.max_phrases)
    weights = TransitionWeights.from_env()
    # Allow CLI to override common weights.
    weights.alpha_music = float(args.alpha_music)
    weights.beta_physics = float(args.beta_physics)
    weights.novelty = float(args.novelty_weight)

    used = set()
    first_idx = choose_first_unit(pool, used)
    used.add(str(pool["unit_ids"][first_idx]))
    motion = pool["motions"][first_idx].copy()
    plan = [{
        "slot": 0,
        "unit_id": str(pool["unit_ids"][first_idx]),
        "unit_index": int(first_idx),
        "start_frame": 0,
        "cut_frame": 0,
        "desired_onset": 0,
        "onset_strength": 0.0,
        "fallback": False,
    }]
    boundaries: List[int] = []

    for slot, (desired, strength) in enumerate(onset_events, start=1):
        if len(motion) >= args.length:
            break
        min_remaining = max(8, args.length - len(motion))
        idx, cut, debug = choose_candidate(
            pool=pool,
            current_motion=motion,
            desired_frame=min(desired, len(motion) - 2),
            onset_strength=strength,
            used=used,
            weights=weights,
            tolerance=args.transition_tolerance,
            min_remaining=min_remaining,
        )
        uid = str(pool["unit_ids"][idx])
        used.add(uid)
        blend_frames = blend_window_from_strength(strength, args.min_blend, args.max_blend)
        before_len = len(motion)
        motion, blend_debug = blend_motions(
            motion,
            pool["motions"][idx],
            cut_frame=cut,
            blend_frames=blend_frames,
            inplace_root=bool(args.inplace),
        )
        boundary = max(0, min(len(motion) - 1, cut))
        boundaries.append(boundary)
        plan.append({
            "slot": int(slot),
            "unit_id": uid,
            "unit_index": int(idx),
            "desired_onset": int(desired),
            "actual_cut_frame": int(cut),
            "boundary_after_stitch": int(boundary),
            "onset_strength": float(strength),
            "motion_len_before": int(before_len),
            "motion_len_after": int(len(motion)),
            "blend": blend_debug,
            **debug,
        })

    # If still short, append safe units without requiring exact onsets.
    while len(motion) < args.length:
        idx, cut, debug = choose_candidate(
            pool=pool,
            current_motion=motion,
            desired_frame=max(1, len(motion) - args.min_tail_context),
            onset_strength=0.0,
            used=used,
            weights=weights,
            tolerance=args.transition_tolerance,
            min_remaining=args.length - len(motion),
        )
        used.add(str(pool["unit_ids"][idx]))
        motion, blend_debug = blend_motions(
            motion, pool["motions"][idx], cut_frame=cut, blend_frames=args.max_blend, inplace_root=bool(args.inplace)
        )
        boundaries.append(int(cut))
        plan.append({
            "slot": int(len(plan)),
            "unit_id": str(pool["unit_ids"][idx]),
            "unit_index": int(idx),
            "desired_onset": None,
            "actual_cut_frame": int(cut),
            "onset_strength": 0.0,
            "blend": blend_debug,
            **debug,
        })
        if len(plan) > args.max_phrases + 4:
            break

    motion = motion[: args.length].astype(np.float32)
    if args.inplace:
        motion[:, ROOT_X_IDX] -= motion[0, ROOT_X_IDX]
        motion[:, ROOT_Z_IDX] -= motion[0, ROOT_Z_IDX]
        motion[:, ROOT_X_IDX] *= float(args.root_drift_keep)
        motion[:, ROOT_Z_IDX] *= float(args.root_drift_keep)

    metrics = local_jump_metrics(motion, boundaries)
    report = {
        "audio": audio_meta,
        "length": int(args.length),
        "fps": int(args.fps),
        "onsets": [{"frame": int(f), "strength": float(s)} for f, s in onset_events],
        "plan": plan,
        "boundaries": [int(x) for x in boundaries],
        "weights": weights.__dict__,
        "metrics": metrics,
        "notes": "Physics-first onset phrase prior: exact onset can be delayed inside transition_tolerance to avoid broken support/contact states.",
    }
    return motion, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="Physics-aware prior pool .npz")
    ap.add_argument("--out", required=True, help="Output long prior .npy")
    ap.add_argument("--audio_wav", default="", help="Optional wav file for onset extraction")
    ap.add_argument("--audio_feature", default="", help="Optional [T,C] audio feature .npy")
    ap.add_argument("--length", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_phrases", type=int, default=4)
    ap.add_argument("--onset_threshold", type=float, default=0.55)
    ap.add_argument("--min_onset_gap", type=int, default=24)
    ap.add_argument("--transition_tolerance", type=int, default=14)
    ap.add_argument("--min_blend", type=int, default=6)
    ap.add_argument("--max_blend", type=int, default=16)
    ap.add_argument("--min_tail_context", type=int, default=16)
    ap.add_argument("--alpha_music", type=float, default=float(os.environ.get("EDGE_ONSET_ALPHA_MUSIC", "0.20")))
    ap.add_argument("--beta_physics", type=float, default=float(os.environ.get("EDGE_ONSET_BETA_PHYSICS", "1.20")))
    ap.add_argument("--novelty_weight", type=float, default=float(os.environ.get("EDGE_ONSET_NOVELTY", "0.35")))
    ap.add_argument("--root_drift_keep", type=float, default=float(os.environ.get("EDGE_ONSET_ROOT_DRIFT_KEEP", "0.02")))
    ap.add_argument("--inplace", action="store_true", help="Keep root X/Z nearly fixed for in-place demo")
    ap.add_argument("--export_pkl_dir", default="", help="Optional directory for DunhuangDataset-compatible .pkl export")
    args = ap.parse_args()

    motion, report = compose_phrase(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, motion.astype(np.float32))
    report_path = out.with_suffix(".report.json")
    write_json(report, report_path)
    print(f"✅ saved onset phrase prior: {out} shape={motion.shape}")
    print(f"✅ report: {report_path}")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))

    if args.export_pkl_dir:
        pkl_dir = Path(args.export_pkl_dir)
        pkl_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = pkl_dir / f"{out.stem}.pkl"
        save_motion_pkl(motion, pkl_path, metadata={
            "original_filename": out.stem,
            "source_file": str(out),
            "generation_report": str(report_path),
            "note": "Generated by generate_onset_phrase_prior.py; can be used for v3_unit_recon fine-tuning.",
        })
        print(f"✅ exported training pkl: {pkl_path}")


if __name__ == "__main__":
    main()
