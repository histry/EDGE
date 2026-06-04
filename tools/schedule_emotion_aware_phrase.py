#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion-aware Visual-First Scheduler

用途：
- 在 V16C 边界感知调度基础上加入音乐情感语义匹配 cost；
- 不替换你原来的 tools/schedule_visual_first_phrase.py，作为并行实验脚本使用；
- 适合先做无训练的 emotion-aware scheduler ablation。

核心：
score = visual_score + activity
        - transition_weight * transition_cost
        - entry_reset_weight * reset_penalty
        - emotion_weight * emotion_distance(music_slot, motion_unit)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def json_safe(obj):
    """Convert numpy / nested objects into JSON-serializable Python objects."""
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    return obj


CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)


def parse_starts(text: str) -> List[int]:
    return [int(float(x.strip())) for x in str(text).replace(";", ",").split(",") if x.strip()]


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


def has_content(frames: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.any(np.abs(frames) > eps, axis=-1)


def localize_root(m: np.ndarray) -> np.ndarray:
    out = np.asarray(m, dtype=np.float32).copy()
    if len(out):
        out[:, ROOT_X] -= out[0, ROOT_X]
        out[:, ROOT_Z] -= out[0, ROOT_Z]
    return out


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def rot_velocity(rot: np.ndarray) -> np.ndarray:
    if len(rot) <= 1:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(rot[1:] - rot[:-1], axis=-1).astype(np.float32)


def motion_descriptor(seg: np.ndarray) -> np.ndarray:
    # 输出 [arousal, tension, calmness]，后续按候选集 minmax 归一化。
    rot = seg[:, ROT].reshape(len(seg), 24, 6)
    d = rot[1:] - rot[:-1] if len(rot) > 1 else np.zeros((0, 24, 6), dtype=np.float32)
    if len(d) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    upper = np.linalg.norm(d[:, 14:24].reshape(len(d), -1), axis=-1).mean()
    torso = np.linalg.norm(d[:, 8:14].reshape(len(d), -1), axis=-1).mean()
    full = np.linalg.norm(d.reshape(len(d), -1), axis=-1)
    rot_range = np.std(rot.reshape(len(rot), -1), axis=0).mean()
    smoothness = 1.0 / (1.0 + float(np.var(full)))
    arousal = float(0.6 * upper + 0.4 * torso)
    tension = float(0.5 * upper + 0.3 * torso + 0.2 * rot_range)
    calmness = float(smoothness)
    return np.array([arousal, tension, calmness], dtype=np.float32)


def normalize_descriptors(cands: List[Dict[str, Any]]) -> None:
    X = np.stack([c["raw_motion_desc"] for c in cands], axis=0)
    lo = X.min(axis=0, keepdims=True)
    hi = X.max(axis=0, keepdims=True)
    Xn = (X - lo) / (hi - lo + 1e-8)
    for c, d in zip(cands, Xn):
        c["motion_desc"] = d.astype(np.float32)


def music_slot_vector(music: Optional[np.ndarray], start: int, length: int = 45) -> np.ndarray:
    if music is None or len(music) == 0:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    T = len(music)
    lo = max(0, min(int(start), T - 1))
    hi = max(lo + 1, min(T, lo + int(length)))
    w = music[lo:hi]
    # music dims: arousal=4, tension=6, calmness=7
    if w.shape[-1] >= 8:
        return np.array([w[:, 4].mean(), w[:, 6].mean(), w[:, 7].mean()], dtype=np.float32)
    return np.array([0.5, 0.5, 0.5], dtype=np.float32)


def transition_cost(prev: np.ndarray, seg: np.ndarray, w: int = 10) -> float:
    if prev is None or len(prev) == 0 or len(seg) == 0:
        return 0.0
    rot_jump = np.linalg.norm(prev[-1, ROT] - seg[0, ROT])
    pv = rot_velocity(prev[max(0, len(prev)-w):, ROT])
    cv = rot_velocity(seg[:min(w, len(seg)), ROT])
    trend = abs(float(pv.mean()) - float(cv.mean())) if len(pv) and len(cv) else 0.0
    return float(rot_jump + 0.35 * trend)


def entry_reset_penalty(seg: np.ndarray, reset_window: int = 8) -> float:
    w = max(2, min(reset_window, len(seg)))
    v = rot_velocity(seg[:w, ROT])
    disp = np.linalg.norm(seg[:w, ROT] - seg[:1, ROT], axis=-1).max() if w else 0.0
    act_def = max(0.0, 0.02 - float(v.mean() if len(v) else 0.0)) / 0.02
    disp_def = max(0.0, 0.065 - float(disp)) / 0.065
    return float(0.65 * act_def + 0.35 * disp_def)


def paste(canvas: np.ndarray, unit: np.ndarray, start: int, offset: int, blend_radius: int) -> np.ndarray:
    T = len(canvas)
    seg = localize_root(unit[offset: offset + max(0, T-start)])
    end = min(T, start + len(seg))
    if end <= start:
        return canvas
    seg = seg[: end-start]
    existing = canvas[start:end]
    mask = has_content(existing)
    if not np.any(mask):
        canvas[start:end] = seg
    else:
        br = max(1, int(blend_radius))
        for i in range(len(seg)):
            t = start + i
            if not np.any(np.abs(canvas[t]) > 1e-8):
                canvas[t] = seg[i]
                continue
            a = smoothstep(i / br)
            canvas[t, CONTACT] = seg[i, CONTACT]
            canvas[t, ROOT_X] = 0.0
            canvas[t, ROOT_Z] = 0.0
            canvas[t, ROOT_Y] = (1-a) * canvas[t, ROOT_Y] + a * seg[i, ROOT_Y]
            canvas[t, ROT] = (1-a) * canvas[t, ROT] + a * seg[i, ROT]
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--music_npy", default="")
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--starts", default="0,32,64,96")
    ap.add_argument("--candidate_top_k", type=int, default=240)
    ap.add_argument("--min_source_gap", type=int, default=90)
    ap.add_argument("--blend_radius", type=int, default=14)
    ap.add_argument("--max_internal_offset", type=int, default=10)
    ap.add_argument("--offset_step", type=int, default=2)
    ap.add_argument("--transition_weight", type=float, default=0.55)
    ap.add_argument("--entry_reset_weight", type=float, default=0.35)
    ap.add_argument("--emotion_weight", type=float, default=0.45)
    ap.add_argument("--visual_weight", type=float, default=1.0)
    ap.add_argument("--activity_weight", type=float, default=0.08)
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    items = report["items"][:args.candidate_top_k]
    music = np.load(args.music_npy).astype(np.float32) if args.music_npy else None
    starts = parse_starts(args.starts)

    cands = []
    for item in items:
        motion, obj = load_pkl_motion(item["pkl"])
        cands.append({
            "motion": motion,
            "pkl": item["pkl"],
            "source_index": int(item.get("source_index", -1)),
            "final_score": float(item.get("final_score", item.get("visual_score", 0.0))),
            "visual_score": float(item.get("visual_score", 0.0)),
            "activity": float(item.get("activity", 0.0)),
            "raw_motion_desc": motion_descriptor(motion),
        })
    normalize_descriptors(cands)

    canvas = np.zeros((args.num_frames, 151), dtype=np.float32)
    chosen = []
    prev_seg = None

    for slot, st in enumerate(starts):
        target = music_slot_vector(music, st, 45)
        best = None
        best_score = -1e18
        for cand in cands:
            if any(c["source_index"] == cand["source_index"] for c in chosen):
                continue
            if any(cand["source_index"] >= 0 and abs(cand["source_index"] - c["source_index"]) < args.min_source_gap for c in chosen):
                continue
            offsets = [0] if slot == 0 else list(range(0, args.max_internal_offset + 1, args.offset_step))
            for off in offsets:
                if len(cand["motion"]) - off < 24:
                    continue
                seg = localize_root(cand["motion"][off:])
                tc = transition_cost(prev_seg, seg)
                rp = entry_reset_penalty(seg)
                ed = float(np.linalg.norm(cand["motion_desc"] - target))
                score = (
                    args.visual_weight * cand["final_score"]
                    + args.activity_weight * cand["activity"]
                    - args.transition_weight * tc
                    - args.entry_reset_weight * rp
                    - args.emotion_weight * ed
                    + (0.03 * off / max(args.max_internal_offset, 1) if slot > 0 else 0.0)
                )
                if score > best_score:
                    best_score = score
                    best = {**cand, "slot": slot, "start": st, "offset": off, "transition_cost": tc,
                            "entry_reset_penalty": rp, "emotion_distance": ed, "music_target": target.tolist(),
                            "slot_score": score}
        if best is None:
            raise RuntimeError(f"No candidate found for slot {slot}. Try lowering --min_source_gap.")
        chosen.append(best)
        canvas = paste(canvas, best["motion"], best["start"], best["offset"], args.blend_radius)
        prev_seg = localize_root(best["motion"][best["offset"]:])

    # fill tail
    nz = np.where(has_content(canvas))[0]
    if len(nz):
        for t in range(int(nz[-1]) + 1, args.num_frames):
            canvas[t] = canvas[int(nz[-1])]
    canvas[:, ROOT_X] = 0.0
    canvas[:, ROOT_Z] = 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, canvas[None].astype(np.float32))

    sched = []
    for c in chosen:
        sched.append({k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                      for k, v in c.items() if k not in ["motion", "raw_motion_desc"]})
    rep = {"scheduler": "emotion_aware_visual_first", "out": str(out), "music_npy": args.music_npy,
           "starts": starts, "schedule": sched}
    out.with_suffix(".schedule_report.json").write_text(json.dumps(json_safe(rep), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(rep), ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
