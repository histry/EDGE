#!/usr/bin/env python3
"""
V16C Visual-First Phrase Scheduler for EDGE-Dunhuang.

This is a drop-in replacement for tools/schedule_visual_first_phrase.py.

Why this version exists
-----------------------
The old scheduler selected visually strong 45-frame units, but it pasted every
selected unit from frame 0. When the visual-first pool contains near-duplicate
sliding windows, the 150-frame demo can look like this:

    start a new phrase -> immediately reset to the unit entry pose

V16C keeps the successful visual-first idea, but adds:
  1) min_source_gap diversity to reject adjacent sliding windows;
  2) entry_reset_penalty for static / entry-pose-like first 8 frames;
  3) transition cost over a short trend window, not only entry frame;
  4) internal offset candidates, so later slots may start inside a unit;
  5) longer cross-fade around overlaps / boundaries.

The script is intentionally inference-only. It does not touch training,
diffusion, checkpoint loading, or the RAG pool builder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_starts(text: str) -> List[int]:
    return [int(float(x.strip())) for x in str(text).replace(";", ",").split(",") if x.strip()]


def parse_offsets(text: str) -> Optional[List[int]]:
    text = str(text or "").strip()
    if not text:
        return None
    vals = []
    for x in text.replace(";", ",").split(","):
        x = x.strip()
        if x:
            vals.append(max(0, int(float(x))))
    return sorted(set(vals))


def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def load_pkl_motion(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    for k in ["motion", "motion_151", "poses"]:
        if k in obj:
            x = np.asarray(obj[k], dtype=np.float32)
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            if x.ndim == 2 and x.shape[-1] == 151:
                return x.astype(np.float32), obj

    raise ValueError(f"No [T,151] motion in {path}")


def is_nonzero_frame(frame: np.ndarray, eps: float = 1e-8) -> bool:
    return bool(np.any(np.abs(frame) > eps))


def has_content(frames: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.any(np.abs(frames) > eps, axis=-1)


def rot_velocity(rot: np.ndarray) -> np.ndarray:
    if len(rot) <= 1:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(rot[1:] - rot[:-1], axis=-1).astype(np.float32)


def safe_percentile(x: np.ndarray, p: float, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return float(default)
    return float(np.percentile(x, p))


def metrics(m: np.ndarray, starts: Optional[Sequence[int]] = None, boundary_radius: int = 2) -> Dict[str, float]:
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT]

    droot = np.linalg.norm(root[1:] - root[:-1], axis=1) if len(m) > 1 else np.zeros(1, dtype=np.float32)
    drot = np.linalg.norm(rot[1:] - rot[:-1], axis=1) if len(m) > 1 else np.zeros(1, dtype=np.float32)

    out: Dict[str, float] = {
        "frames": int(len(m)),
        "root_max_radius": float(np.linalg.norm(root - root[:1], axis=1).max()) if len(m) else 0.0,
        "global_root_jump_p95": safe_percentile(droot, 95),
        "global_rot_jump_p95": safe_percentile(drot, 95),
        "global_rot_jump_max": float(drot.max()) if len(drot) else 0.0,
        "segment_activity_mean": float(drot.mean()) if len(drot) else 0.0,
        "nonzero_frame_ratio": float(has_content(m).mean()) if len(m) else 0.0,
    }

    # Report both old common boundaries and actual starts.
    boundaries = set([35, 45, 70, 74, 90, 96, 105, 108, 128, 135, 140, 142])
    if starts:
        boundaries.update(int(x) for x in starts if int(x) > 0)

    for b in sorted(boundaries):
        if 2 <= b < len(m) - 2:
            lo = max(1, b - boundary_radius)
            hi = min(len(m), b + boundary_radius + 1)
            local = np.linalg.norm(rot[lo:hi] - rot[lo - 1:hi - 1], axis=1)
            out[f"boundary_{b}_local_rot_jump_max"] = float(local.max()) if len(local) else 0.0

    return out


def localize_root(m: np.ndarray, root_origin: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.asarray(m, dtype=np.float32).copy()
    if len(out) == 0:
        return out
    if root_origin is None:
        root_origin = out[0, [ROOT_X, ROOT_Z]].copy()
    out[:, ROOT_X] -= float(root_origin[0])
    out[:, ROOT_Z] -= float(root_origin[1])
    return out


def candidate_offsets(
    unit_len: int,
    slot: int,
    max_internal_offset: int,
    offset_step: int,
    min_remaining: int,
    offset_list: Optional[Sequence[int]] = None,
    allow_first_offset: bool = False,
) -> List[int]:
    if unit_len <= 0:
        return []

    if offset_list is not None:
        offsets = [int(o) for o in offset_list]
    else:
        max_off = max(0, min(int(max_internal_offset), unit_len - 1))
        step = max(1, int(offset_step))
        offsets = list(range(0, max_off + 1, step))

    offsets = [o for o in offsets if 0 <= o < unit_len and (unit_len - o) >= int(min_remaining)]
    if not offsets:
        offsets = [0]

    if slot == 0 and not allow_first_offset:
        return [0]

    return sorted(set(offsets))


def extract_segment(unit: np.ndarray, offset: int, max_len: int) -> np.ndarray:
    offset = max(0, min(int(offset), len(unit) - 1))
    seg = unit[offset: offset + max_len].astype(np.float32)
    return localize_root(seg)


def early_motion_stats(seg: np.ndarray, reset_window: int) -> Dict[str, float]:
    w = max(2, min(int(reset_window), len(seg)))
    rot = seg[:w, ROT]
    vel = rot_velocity(rot)
    disp = np.linalg.norm(rot - rot[:1], axis=-1) if len(rot) else np.zeros(1, dtype=np.float32)

    return {
        "early_activity": float(vel.mean()) if len(vel) else 0.0,
        "early_displacement": float(disp.max()) if len(disp) else 0.0,
        "early_final_displacement": float(disp[-1]) if len(disp) else 0.0,
    }


def entry_reset_penalty(
    seg: np.ndarray,
    reset_window: int,
    min_early_activity: float,
    min_early_displacement: float,
) -> Tuple[float, Dict[str, float]]:
    st = early_motion_stats(seg, reset_window=reset_window)

    act_deficit = max(0.0, float(min_early_activity) - st["early_activity"]) / max(float(min_early_activity), 1e-8)
    disp_deficit = max(0.0, float(min_early_displacement) - st["early_displacement"]) / max(float(min_early_displacement), 1e-8)

    # If the first window barely moves, this candidate probably starts from a
    # static / entry pose and may create a visible "reset".
    penalty = 0.65 * act_deficit + 0.35 * disp_deficit
    return float(penalty), st


def canvas_transition_cost(
    canvas: np.ndarray,
    start: int,
    seg: np.ndarray,
    trend_window: int,
    entry_weight: float,
    trend_weight: float,
    overlap_weight: float,
    root_weight: float,
) -> Tuple[float, Dict[str, float]]:
    T = len(canvas)
    start = int(start)
    w = max(2, int(trend_window))

    parts: Dict[str, float] = {
        "entry_rot_jump": 0.0,
        "entry_root_jump": 0.0,
        "trend_vel_jump": 0.0,
        "overlap_rot_mse": 0.0,
        "overlap_root_mse": 0.0,
    }

    if len(seg) == 0:
        return 1e9, parts

    # Entry cost: compare with the last existing frame before start.
    prev_idx = None
    for t in range(min(start - 1, T - 1), -1, -1):
        if is_nonzero_frame(canvas[t]):
            prev_idx = t
            break

    if prev_idx is not None:
        parts["entry_rot_jump"] = float(np.linalg.norm(canvas[prev_idx, ROT] - seg[0, ROT]))
        parts["entry_root_jump"] = float(np.linalg.norm(canvas[prev_idx, [ROOT_X, ROOT_Z]] - seg[0, [ROOT_X, ROOT_Z]]))

        # Trend cost: compare the previous rotational velocity trend with the
        # candidate's first few-frame velocity trend.
        prev_lo = max(0, prev_idx - w + 1)
        prev_chunk = canvas[prev_lo:prev_idx + 1]
        prev_chunk = prev_chunk[has_content(prev_chunk)]
        cand_chunk = seg[: min(w, len(seg))]

        pv = rot_velocity(prev_chunk[:, ROT]) if len(prev_chunk) >= 2 else np.zeros((0,), dtype=np.float32)
        cv = rot_velocity(cand_chunk[:, ROT]) if len(cand_chunk) >= 2 else np.zeros((0,), dtype=np.float32)
        if len(pv) and len(cv):
            parts["trend_vel_jump"] = float(abs(float(pv.mean()) - float(cv.mean())))

    # Overlap cost: if the new segment overlaps already-filled frames, compare
    # the whole short window rather than only one boundary frame.
    end = min(T, start + len(seg))
    if start < end:
        existing = canvas[start:end]
        mask = has_content(existing)
        if np.any(mask):
            n = min(int(w), len(existing), len(seg))
            mask_n = mask[:n]
            if np.any(mask_n):
                ex = existing[:n][mask_n]
                sg = seg[:n][mask_n]
                parts["overlap_rot_mse"] = float(np.mean((ex[:, ROT] - sg[:, ROT]) ** 2))
                parts["overlap_root_mse"] = float(np.mean((ex[:, [ROOT_X, ROOT_Z]] - sg[:, [ROOT_X, ROOT_Z]]) ** 2))

    total = (
        float(entry_weight) * parts["entry_rot_jump"]
        + float(root_weight) * parts["entry_root_jump"]
        + float(trend_weight) * parts["trend_vel_jump"]
        + float(overlap_weight) * (parts["overlap_rot_mse"] ** 0.5)
        + float(root_weight) * (parts["overlap_root_mse"] ** 0.5)
    )
    return float(total), parts


def paste_with_blend(canvas: np.ndarray, unit: np.ndarray, start: int, blend_radius: int, offset: int = 0) -> np.ndarray:
    T = canvas.shape[0]
    start = int(start)
    offset = max(0, int(offset))
    unit = np.asarray(unit, dtype=np.float32)

    if start >= T or len(unit) == 0:
        return canvas

    seg = extract_segment(unit, offset=offset, max_len=T - start)
    end = min(T, start + len(seg))
    length = end - start
    if length <= 0:
        return canvas

    seg = seg[:length]
    existing = canvas[start:end]
    existing_mask = has_content(existing)

    # If there is no overlap, paste directly.
    if not np.any(existing_mask):
        canvas[start:end] = seg
        return canvas

    # Boundary-aware long cross-fade. The blend weight ramps in slowly for the
    # overlapping boundary, then becomes full candidate after blend_radius.
    br = max(1, int(blend_radius))
    for i in range(length):
        t = start + i
        existing_nonzero = is_nonzero_frame(canvas[t])

        if not existing_nonzero:
            canvas[t] = seg[i]
            continue

        a = float(i) / float(br)
        a = float(smoothstep(a))
        a = max(0.0, min(1.0, a))

        canvas[t, CONTACT] = seg[i, CONTACT]
        canvas[t, ROOT_X] = 0.0
        canvas[t, ROOT_Z] = 0.0
        canvas[t, ROOT_Y] = (1.0 - a) * canvas[t, ROOT_Y] + a * seg[i, ROOT_Y]
        canvas[t, ROT] = (1.0 - a) * canvas[t, ROT] + a * seg[i, ROT]

    return canvas


def build_candidates(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    for item in items:
        pkl = item["pkl"]
        motion, obj = load_pkl_motion(pkl)

        metadata = {}
        if isinstance(obj, dict):
            metadata = obj.get("metadata", {}) or {}

        candidates.append({
            "motion": motion,
            "item": item,
            "pkl": pkl,
            "source_index": int(item.get("source_index", metadata.get("source_index", -1))),
            "source_file": str(item.get("source_file", metadata.get("source_file", ""))),
            "original_filename": str(item.get("original_filename", metadata.get("original_filename", ""))),
            "final_score": float(item.get("final_score", item.get("visual_score", 0.0))),
            "visual_score": float(item.get("visual_score", 0.0)),
            "activity": float(item.get("activity", 0.0)),
        })
    return candidates


def source_gap_ok(cand: Dict[str, Any], chosen: Sequence[Dict[str, Any]], min_source_gap: int) -> bool:
    if int(min_source_gap) <= 0:
        return True
    idx = int(cand.get("source_index", -1))
    if idx < 0:
        return True
    for c in chosen:
        cidx = int(c.get("source_index", -1))
        if cidx >= 0 and abs(idx - cidx) < int(min_source_gap):
            return False
    return True


def select_for_slot(
    slot: int,
    start: int,
    canvas: np.ndarray,
    candidates: Sequence[Dict[str, Any]],
    chosen: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    allow_relax_source_gap: bool = False,
) -> Dict[str, Any]:
    best = None
    best_score = -1e18
    best_debug: Dict[str, Any] = {}

    offset_list = parse_offsets(args.offsets)

    for cand in candidates:
        # Exact duplicate is always forbidden.
        if cand["source_index"] in [c["source_index"] for c in chosen]:
            continue

        if (not allow_relax_source_gap) and (not source_gap_ok(cand, chosen, args.min_source_gap)):
            continue

        offsets = candidate_offsets(
            unit_len=len(cand["motion"]),
            slot=slot,
            max_internal_offset=args.max_internal_offset,
            offset_step=args.offset_step,
            min_remaining=args.min_remaining_frames,
            offset_list=offset_list,
            allow_first_offset=bool(args.allow_first_offset),
        )

        for off in offsets:
            seg = extract_segment(cand["motion"], offset=off, max_len=args.num_frames - start)
            if len(seg) < args.min_remaining_frames:
                continue

            trans_cost, trans_parts = canvas_transition_cost(
                canvas=canvas,
                start=start,
                seg=seg,
                trend_window=args.trend_window,
                entry_weight=args.entry_cost_weight,
                trend_weight=args.trend_cost_weight,
                overlap_weight=args.overlap_cost_weight,
                root_weight=args.root_cost_weight,
            )

            if trans_cost > args.max_transition_cost:
                continue

            reset_cost, reset_parts = entry_reset_penalty(
                seg,
                reset_window=args.reset_window,
                min_early_activity=args.min_early_activity,
                min_early_displacement=args.min_early_displacement,
            )

            zero_offset_penalty = 0.0
            if slot > 0 and off == 0:
                zero_offset_penalty = float(args.zero_offset_penalty)

            offset_bonus = 0.0
            if slot > 0 and off > 0:
                # Small bonus: prefer starting inside a unit when cost is similar,
                # but never overwhelm visual quality or transition safety.
                offset_bonus = float(args.internal_offset_bonus) * min(off / max(args.max_internal_offset, 1), 1.0)

            sc = (
                args.visual_weight * cand["final_score"]
                + args.activity_weight * cand["activity"]
                - args.transition_weight * trans_cost
                - args.entry_reset_weight * reset_cost
                - zero_offset_penalty
                + offset_bonus
            )

            if sc > best_score:
                best_score = float(sc)
                best = {
                    **cand,
                    "slot": int(slot),
                    "start": int(start),
                    "offset": int(off),
                    "transition_cost": float(trans_cost),
                    "entry_reset_penalty": float(reset_cost),
                    "zero_offset_penalty": float(zero_offset_penalty),
                    "offset_bonus": float(offset_bonus),
                    "slot_score": float(sc),
                    "relaxed_source_gap": bool(allow_relax_source_gap),
                }
                best_debug = {
                    "transition_parts": trans_parts,
                    "entry_reset_parts": reset_parts,
                    "candidate_segment_len": int(len(seg)),
                }

    if best is None:
        if allow_relax_source_gap:
            raise RuntimeError(
                f"No candidate available for slot={slot}, start={start}. "
                "Try increasing --candidate_top_k or loosening --max_transition_cost."
            )
        return select_for_slot(
            slot=slot,
            start=start,
            canvas=canvas,
            candidates=candidates,
            chosen=chosen,
            args=args,
            allow_relax_source_gap=True,
        )

    best.update(best_debug)
    return best


def write_human_summary(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# V16C Visual-First Scheduler Report",
        "",
        "## Output",
        "",
        f"- Motion: `{report['out']}`",
        f"- Source report: `{report['source_report']}`",
        f"- Starts: `{report['starts']}`",
        "",
        "## Schedule",
        "",
        "| slot | start | offset | source_index | slot_score | transition | reset | relaxed_gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in report["schedule"]:
        lines.append(
            "| {slot} | {start} | {offset} | {source_index} | {slot_score:.4f} | "
            "{transition_cost:.4f} | {entry_reset_penalty:.4f} | {relaxed_source_gap} |".format(**s)
        )

    lines += [
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(report["metrics"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "- `offset > 0` means the scheduler skipped the retrieved unit entry pose to avoid phrase reset.",
        "- `relaxed_gap=true` means source-index diversity was relaxed only because no valid candidate remained.",
        "- Large boundary jump still indicates a stitching problem, not a failure of the visual-first prior pool.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--starts", default=os.environ.get("EDGE_V16C_STARTS", "0,32,64,96"))
    ap.add_argument("--candidate_top_k", type=int, default=env_int("EDGE_V16C_CANDIDATE_TOP_K", 240))

    # Original scoring weights, still supported.
    ap.add_argument("--transition_weight", type=float, default=env_float("EDGE_V16C_TRANSITION_WEIGHT", 0.65))
    ap.add_argument("--activity_weight", type=float, default=env_float("EDGE_V16C_ACTIVITY_WEIGHT", 0.08))
    ap.add_argument("--visual_weight", type=float, default=env_float("EDGE_V16C_VISUAL_WEIGHT", 1.0))
    ap.add_argument("--blend_radius", type=int, default=env_int("EDGE_V16C_BLEND_RADIUS", 14))
    ap.add_argument("--max_transition_cost", type=float, default=env_float("EDGE_V16C_MAX_TRANSITION_COST", 999.0))

    # V16C additions.
    ap.add_argument("--min_source_gap", type=int, default=env_int("EDGE_V16C_MIN_SOURCE_GAP", 90))
    ap.add_argument("--reset_window", type=int, default=env_int("EDGE_V16C_RESET_WINDOW", 8))
    ap.add_argument("--entry_reset_weight", type=float, default=env_float("EDGE_V16C_ENTRY_RESET_WEIGHT", 0.45))
    ap.add_argument("--min_early_activity", type=float, default=env_float("EDGE_V16C_MIN_EARLY_ACTIVITY", 0.020))
    ap.add_argument("--min_early_displacement", type=float, default=env_float("EDGE_V16C_MIN_EARLY_DISPLACEMENT", 0.065))

    ap.add_argument("--trend_window", type=int, default=env_int("EDGE_V16C_TREND_WINDOW", 10))
    ap.add_argument("--entry_cost_weight", type=float, default=env_float("EDGE_V16C_ENTRY_COST_WEIGHT", 1.0))
    ap.add_argument("--trend_cost_weight", type=float, default=env_float("EDGE_V16C_TREND_COST_WEIGHT", 0.35))
    ap.add_argument("--overlap_cost_weight", type=float, default=env_float("EDGE_V16C_OVERLAP_COST_WEIGHT", 0.75))
    ap.add_argument("--root_cost_weight", type=float, default=env_float("EDGE_V16C_ROOT_COST_WEIGHT", 3.0))

    ap.add_argument("--max_internal_offset", type=int, default=env_int("EDGE_V16C_MAX_INTERNAL_OFFSET", 10))
    ap.add_argument("--offset_step", type=int, default=env_int("EDGE_V16C_OFFSET_STEP", 2))
    ap.add_argument("--offsets", default=os.environ.get("EDGE_V16C_OFFSETS", ""))
    ap.add_argument("--allow_first_offset", type=int, default=env_int("EDGE_V16C_ALLOW_FIRST_OFFSET", 0))
    ap.add_argument("--min_remaining_frames", type=int, default=env_int("EDGE_V16C_MIN_REMAINING_FRAMES", 24))
    ap.add_argument("--zero_offset_penalty", type=float, default=env_float("EDGE_V16C_ZERO_OFFSET_PENALTY", 0.15))
    ap.add_argument("--internal_offset_bonus", type=float, default=env_float("EDGE_V16C_INTERNAL_OFFSET_BONUS", 0.05))

    args = ap.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = report["items"][: args.candidate_top_k]
    starts = parse_starts(args.starts)

    if not starts:
        raise ValueError("--starts cannot be empty")

    candidates = build_candidates(items)
    if not candidates:
        raise RuntimeError("No candidates loaded.")

    canvas = np.zeros((args.num_frames, 151), dtype=np.float32)
    chosen: List[Dict[str, Any]] = []

    for slot, st in enumerate(starts):
        if st < 0 or st >= args.num_frames:
            raise ValueError(f"Invalid start={st} for num_frames={args.num_frames}")

        best = select_for_slot(
            slot=slot,
            start=st,
            canvas=canvas,
            candidates=candidates,
            chosen=chosen,
            args=args,
        )

        chosen.append(best)
        canvas = paste_with_blend(
            canvas,
            best["motion"],
            start=st,
            blend_radius=args.blend_radius,
            offset=best["offset"],
        )

    # Fill any empty tail with last valid frame.
    nz = np.where(has_content(canvas))[0]
    if len(nz):
        last = int(nz[-1])
        for t in range(last + 1, args.num_frames):
            canvas[t] = canvas[last]

    # If there are accidental early gaps, hold the previous valid frame.
    nz = np.where(has_content(canvas))[0]
    if len(nz):
        first = int(nz[0])
        for t in range(0, first):
            canvas[t] = canvas[first]
        for t in range(first + 1, args.num_frames):
            if not is_nonzero_frame(canvas[t]):
                canvas[t] = canvas[t - 1]

    # Keep root X/Z exactly in-place for the first-paper stationary setting.
    canvas[:, ROOT_X] = 0.0
    canvas[:, ROOT_Z] = 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, canvas[None].astype(np.float32))

    schedule = []
    for c in chosen:
        schedule.append({
            "slot": int(c["slot"]),
            "start": int(c["start"]),
            "offset": int(c["offset"]),
            "source_index": int(c["source_index"]),
            "pkl": c["pkl"],
            "final_score": float(c["final_score"]),
            "visual_score": float(c["visual_score"]),
            "activity": float(c["activity"]),
            "transition_cost": float(c["transition_cost"]),
            "entry_reset_penalty": float(c["entry_reset_penalty"]),
            "zero_offset_penalty": float(c["zero_offset_penalty"]),
            "offset_bonus": float(c["offset_bonus"]),
            "slot_score": float(c["slot_score"]),
            "relaxed_source_gap": bool(c["relaxed_source_gap"]),
            "transition_parts": c.get("transition_parts", {}),
            "entry_reset_parts": c.get("entry_reset_parts", {}),
            "candidate_segment_len": int(c.get("candidate_segment_len", 0)),
        })

    out_report = {
        "scheduler": "v16c_boundary_aware_diverse_visual_first",
        "out": str(out),
        "source_report": str(report_path),
        "num_frames": int(args.num_frames),
        "starts": starts,
        "config": {
            "candidate_top_k": int(args.candidate_top_k),
            "transition_weight": float(args.transition_weight),
            "activity_weight": float(args.activity_weight),
            "visual_weight": float(args.visual_weight),
            "blend_radius": int(args.blend_radius),
            "max_transition_cost": float(args.max_transition_cost),
            "min_source_gap": int(args.min_source_gap),
            "reset_window": int(args.reset_window),
            "entry_reset_weight": float(args.entry_reset_weight),
            "min_early_activity": float(args.min_early_activity),
            "min_early_displacement": float(args.min_early_displacement),
            "trend_window": int(args.trend_window),
            "max_internal_offset": int(args.max_internal_offset),
            "offset_step": int(args.offset_step),
            "offsets": args.offsets,
            "allow_first_offset": bool(args.allow_first_offset),
            "min_remaining_frames": int(args.min_remaining_frames),
            "zero_offset_penalty": float(args.zero_offset_penalty),
            "internal_offset_bonus": float(args.internal_offset_bonus),
        },
        "schedule": schedule,
        "metrics": metrics(canvas, starts=starts),
    }

    json_path = out.with_suffix(".schedule_report.json")
    md_path = out.with_suffix(".schedule_report.md")
    json_path.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_human_summary(md_path, out_report)

    print(json.dumps(out_report, ensure_ascii=False, indent=2))
    print(f"saved: {out}")
    print(f"report_json: {json_path}")
    print(f"report_md: {md_path}")


if __name__ == "__main__":
    main()
