#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V20C Event-Graph Phrase Scheduler

修复点：
1. 支持 Dynamic Rhythm Event-RAG variable-length event units；
2. 修复 source_id gap 误杀问题：只在同一 source_id 内比较 source_start；
3. 加入 Music-to-Motion Event Semantic Mapping：
   accent -> high_tension / arm_flourish / support_shift
   section_change -> support_shift / build_up / release
   calm_flow -> calm_flow / pose_hold
4. 支持 beam search；
5. 可自动应用 common start pose，保证三首音乐首帧一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_motion(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        x = np.load(p, allow_pickle=True)
    elif p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        for k in ["motion", "motion_151", "canonical_motion", "arr_0"]:
            if k in z:
                x = z[k]
                break
        else:
            raise ValueError(f"No motion key in {p}")
    else:
        obj = pickle.load(open(p, "rb"))
        if isinstance(obj, dict):
            for k in ["motion", "motion_151", "canonical_motion"]:
                if k in obj:
                    x = obj[k]
                    break
            else:
                raise ValueError(f"No motion key in {p}, keys={list(obj.keys())[:20]}")
        else:
            x = obj

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"{p} should be [T,151] or [1,T,151], got {x.shape}")
    return x.astype(np.float32)


def localize_root(m: np.ndarray) -> np.ndarray:
    y = np.asarray(m, dtype=np.float32).copy()
    if len(y):
        y[:, ROOT_X] -= y[0, ROOT_X]
        y[:, ROOT_Z] -= y[0, ROOT_Z]
    return y


def rot_velocity(m: np.ndarray) -> np.ndarray:
    if len(m) <= 1:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(m[1:, ROT] - m[:-1, ROT], axis=-1).astype(np.float32)


def transition_cost(prev: np.ndarray | None, cur: np.ndarray, w: int = 8) -> float:
    if prev is None or len(prev) == 0 or len(cur) == 0:
        return 0.0
    pose_jump = float(np.linalg.norm(prev[-1, ROT] - cur[0, ROT]))
    pv = rot_velocity(prev[max(0, len(prev) - w):])
    cv = rot_velocity(cur[:min(w, len(cur))])
    trend = abs(float(pv.mean()) - float(cv.mean())) if len(pv) and len(cv) else 0.0
    root_y_jump = abs(float(prev[-1, ROOT_Y] - cur[0, ROOT_Y]))
    return pose_jump + 0.35 * trend + 0.5 * root_y_jump


def make_transition(prev: np.ndarray, cur: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((0, 151), dtype=np.float32)
    out = np.zeros((k, 151), dtype=np.float32)
    a0 = prev[-1]
    a1 = cur[0]
    for i in range(k):
        a = smoothstep((i + 1) / (k + 1))
        out[i, CONTACT] = a1[CONTACT]
        out[i, ROOT_X] = 0.0
        out[i, ROOT_Z] = 0.0
        out[i, ROOT_Y] = (1 - a) * a0[ROOT_Y] + a * a1[ROOT_Y]
        out[i, ROT] = (1 - a) * a0[ROT] + a * a1[ROT]
    return out


def get_desc(item: Dict[str, Any], key: str, default=0.0) -> float:
    if key in item:
        try:
            return float(item[key])
        except Exception:
            pass
    d = item.get("descriptor", {})
    if isinstance(d, dict) and key in d:
        try:
            return float(d[key])
        except Exception:
            pass
    return float(default)


def get_event_type(item: Dict[str, Any]) -> str:
    et = item.get("event_type", None)
    if et:
        return str(et)
    d = item.get("descriptor", {})
    if isinstance(d, dict) and d.get("event_type", None):
        return str(d["event_type"])
    return "neutral_flow"


def event_semantic_compatibility(music_event: str, motion_event: str) -> float:
    """Soft mapping from music events to motion events."""
    m = str(music_event or "neutral_flow")
    e = str(motion_event or "neutral_flow")

    table = {
        "accent": {
            "high_tension": 1.00,
            "arm_flourish": 0.98,
            "support_shift": 0.75,
            "build_up": 0.70,
            "neutral_flow": 0.15,
            "calm_flow": -0.15,
            "pose_hold": -0.25,
        },
        "climax": {
            "high_tension": 1.00,
            "arm_flourish": 1.00,
            "build_up": 0.75,
            "support_shift": 0.60,
            "neutral_flow": 0.10,
            "pose_hold": -0.30,
        },
        "section_change": {
            "support_shift": 1.00,
            "build_up": 0.85,
            "release": 0.75,
            "arm_flourish": 0.60,
            "high_tension": 0.55,
            "neutral_flow": 0.20,
            "pose_hold": -0.10,
        },
        "build_up": {
            "build_up": 1.00,
            "high_tension": 0.85,
            "arm_flourish": 0.70,
            "support_shift": 0.50,
            "neutral_flow": 0.20,
        },
        "release": {
            "release": 1.00,
            "calm_flow": 0.80,
            "pose_hold": 0.60,
            "neutral_flow": 0.30,
            "high_tension": -0.20,
        },
        "calm_flow": {
            "calm_flow": 1.00,
            "pose_hold": 0.70,
            "neutral_flow": 0.50,
            "release": 0.35,
            "high_tension": -0.35,
            "arm_flourish": -0.25,
        },
        "neutral_flow": {
            "neutral_flow": 0.75,
            "calm_flow": 0.55,
            "pose_hold": 0.35,
            "build_up": 0.25,
        },
    }

    if m == e:
        return 1.0
    return float(table.get(m, {}).get(e, 0.0))


def load_event_db(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    if not isinstance(items, list):
        raise ValueError(f"Invalid event db: {path}")
    return items


def classify_music_from_vec(vec: np.ndarray) -> str:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if len(v) >= 8:
        onset = float(v[1]) if len(v) > 1 else 0.0
        beat = float(v[2]) if len(v) > 2 else 0.0
        arousal = float(v[4])
        tension = float(v[6])
        calmness = float(v[7])
    else:
        onset = float(v[1]) if len(v) > 1 else 0.0
        beat = float(v[2]) if len(v) > 2 else 0.0
        arousal = float(v[0]) if len(v) > 0 else 0.5
        tension = float(v[1]) if len(v) > 1 else 0.5
        calmness = float(v[2]) if len(v) > 2 else 0.5

    if onset > 0.55 or beat > 0.55:
        return "accent"
    if tension > 0.72 and arousal > 0.55:
        return "climax"
    if calmness > 0.68 and tension < 0.55:
        return "calm_flow"
    if arousal > 0.62 or tension > 0.62:
        return "build_up"
    return "neutral_flow"


def load_music_events(path: str) -> Tuple[np.ndarray, List[str]]:
    p = Path(path)
    arr = np.load(p, allow_pickle=True)

    json_path = p.with_suffix(".json")
    json_events = None
    if json_path.is_file():
        try:
            js = json.loads(json_path.read_text(encoding="utf-8"))
            for key in ["events", "frame_events", "music_events", "frames"]:
                if key in js and isinstance(js[key], list):
                    json_events = js[key]
                    break
        except Exception:
            json_events = None

    if arr.dtype == object:
        rows = list(arr.reshape(-1))
        events = []
        vecs = []
        for r in rows:
            if isinstance(r, dict):
                events.append(str(r.get("event", r.get("event_type", "neutral_flow"))))
                vals = []
                for k in ["energy", "onset", "beat", "tempo", "arousal", "delta_arousal", "tension", "calmness"]:
                    vals.append(float(r.get(k, 0.0)))
                vecs.append(vals)
        if vecs:
            return np.asarray(vecs, dtype=np.float32), events

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]

    events = []
    for i in range(arr.shape[0]):
        ev = None
        if json_events is not None and i < len(json_events):
            r = json_events[i]
            if isinstance(r, dict):
                ev = r.get("event", r.get("event_type", None))
            elif isinstance(r, str):
                ev = r
        events.append(str(ev) if ev else classify_music_from_vec(arr[i]))

    return arr, events



def _norm_by_track(music: np.ndarray, dim: int, val: float) -> float:
    if music.ndim != 2 or music.shape[1] <= dim:
        return float(val)
    col = np.asarray(music[:, dim], dtype=np.float32)
    lo, hi = np.percentile(col, [10, 90])
    if hi - lo < 1e-6:
        return 0.5
    return float(np.clip((val - lo) / (hi - lo + 1e-6), 0.0, 1.0))


def music_slot(music: np.ndarray, events: List[str], start: int, length: int = 45) -> Dict[str, Any]:
    """V20D phrase-level music event.

    不再让逐帧 accent 主导整个 slot。
    accent 只作为 rhythm density；phrase event 主要由 arousal/tension/calmness/delta/section_change 决定。
    """
    T = len(music)
    lo = max(0, min(int(start), max(T - 1, 0)))
    hi = max(lo + 1, min(T, lo + int(length)))
    win = music[lo:hi]

    if len(win) == 0:
        vec = np.zeros((music.shape[-1],), dtype=np.float32)
    else:
        vec = win.mean(axis=0).astype(np.float32)

    evs = events[lo:hi] if events else []
    n = max(len(evs), 1)
    accent_density = evs.count("accent") / n
    section_density = evs.count("section_change") / n
    release_density = evs.count("release") / n
    calm_density = evs.count("calm_flow") / n
    climax_density = evs.count("climax") / n
    buildup_density = evs.count("build_up") / n

    # feature convention from extract_music_event_stream:
    # [energy, onset, beat, tempo, arousal, delta_arousal, tension, calmness]
    energy = float(vec[0]) if len(vec) > 0 else 0.5
    onset = float(vec[1]) if len(vec) > 1 else 0.0
    beat = float(vec[2]) if len(vec) > 2 else 0.0
    arousal = float(vec[4]) if len(vec) > 4 else 0.5
    delta_arousal = float(vec[5]) if len(vec) > 5 else 0.0
    tension = float(vec[6]) if len(vec) > 6 else 0.5
    calmness = float(vec[7]) if len(vec) > 7 else 0.5

    energy_n = _norm_by_track(music, 0, energy)
    arousal_n = _norm_by_track(music, 4, arousal)
    tension_n = _norm_by_track(music, 6, tension)
    calm_n = _norm_by_track(music, 7, calmness)

    # delta over the current phrase window
    if hi - lo > 2 and music.shape[1] > 6:
        da = float(np.mean(music[max(lo+1, lo):hi, 4] - music[lo:hi-1, 4]))
        dt = float(np.mean(music[max(lo+1, lo):hi, 6] - music[lo:hi-1, 6]))
    else:
        da = delta_arousal
        dt = 0.0

    # phrase-level event decision
    if section_density >= 0.08 and (tension_n >= 0.45 or arousal_n >= 0.45):
        ev = "section_change"
    elif release_density >= 0.04 or (da < -0.02 and dt < -0.02 and tension_n < 0.65):
        ev = "release"
    elif calm_n >= 0.68 and tension_n <= 0.55:
        ev = "calm_flow"
    elif climax_density >= 0.03 or (tension_n >= 0.75 and arousal_n >= 0.55):
        ev = "climax"
    elif buildup_density >= 0.04 or da > 0.02 or dt > 0.02 or tension_n >= 0.62:
        ev = "build_up"
    elif accent_density >= 0.45 and energy_n >= 0.55 and (arousal_n >= 0.55 or tension_n >= 0.55):
        ev = "accent"
    else:
        ev = "neutral_flow"

    # normalized affect target for matching motion descriptors
    affect = np.array([arousal_n, tension_n, calm_n], dtype=np.float32)

    return {
        "event": ev,
        "vector": vec,
        "affect": affect,
        "start": lo,
        "end": hi,
        "accent_density": float(accent_density),
        "section_density": float(section_density),
        "release_density": float(release_density),
        "calm_density": float(calm_density),
        "energy_norm": float(energy_n),
        "arousal_norm": float(arousal_n),
        "tension_norm": float(tension_n),
        "calm_norm": float(calm_n),
        "delta_arousal": float(da),
        "delta_tension": float(dt),
    }


def motion_affect(item: Dict[str, Any]) -> np.ndarray:
    upper = get_desc(item, "upper_activity", 0.0)
    torso = get_desc(item, "torso_activity", 0.0)
    tension = get_desc(item, "style_tension", 0.0)
    smooth = get_desc(item, "smoothness", 1.0)
    arousal = 0.65 * upper + 0.35 * torso
    return np.array([arousal, tension, smooth], dtype=np.float32)


def normalize_affects(items: List[Dict[str, Any]]) -> None:
    X = np.stack([motion_affect(x) for x in items], axis=0)
    lo = X.min(axis=0, keepdims=True)
    hi = X.max(axis=0, keepdims=True)
    Xn = (X - lo) / (hi - lo + 1e-8)
    for item, v in zip(items, Xn):
        item["_affect_norm"] = v.astype(np.float32)


def transition_len_rule(prev_event: str, next_event: str, music_event: str) -> int:
    if music_event in {"accent", "climax"}:
        return 6 if next_event in {"high_tension", "arm_flourish"} else 8
    if music_event == "section_change":
        return 12 if next_event in {"support_shift", "build_up"} else 10
    if next_event in {"calm_flow", "pose_hold", "release"}:
        return 14
    return 10



def music_specific_bias(music_key: str, event_id: str) -> float:
    """Small deterministic tie-breaker for high-quality candidates.

    It should not dominate event matching, only prevents different music from selecting
    exactly the same top-quality units when scores are nearly tied.
    """
    key = f"{music_key}::{event_id}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    v = int(h[:8], 16) / float(0xFFFFFFFF)
    return float(2.0 * v - 1.0)


def candidate_score(
    cand: Dict[str, Any],
    prev: Dict[str, Any] | None,
    chosen: List[Dict[str, Any]],
    music_ev: Dict[str, Any],
    args,
) -> Tuple[float, Dict[str, Any]]:
    visual = get_desc(cand, "visual_score", get_desc(cand, "quality_score", 0.0))
    quality = get_desc(cand, "quality_score", visual)
    safety = get_desc(cand, "safety_score", quality)
    activity = (
        get_desc(cand, "upper_activity", 0.0)
        + 0.6 * get_desc(cand, "torso_activity", 0.0)
        + 0.3 * get_desc(cand, "lower_activity", 0.0)
    )

    motion_event = get_event_type(cand)
    ematch = event_semantic_compatibility(music_ev["event"], motion_event)

    aff = cand.get("_affect_norm", np.array([0.5, 0.5, 0.5], dtype=np.float32))
    target = np.asarray(music_ev.get("affect", [0.5, 0.5, 0.5]), dtype=np.float32)
    ed = float(np.linalg.norm(aff - target))
    mbias = music_specific_bias(str(music_ev.get("music_key", "")), str(cand.get("event_id", cand.get("pkl", ""))))

    tc = 0.0
    if prev is not None:
        tc = transition_cost(prev["_motion"], cand["_motion"])

    # diversity bonus: encourage different event type / source
    used_events = {c.get("event_type") for c in chosen}
    used_sources = {int(c.get("source_id", -1)) for c in chosen}
    div = 0.0
    if motion_event not in used_events:
        div += 1.0
    if int(cand.get("source_id", -1)) not in used_sources:
        div += 0.35

    # V20E: avoid over-static schedules.
    # Low-activity pose_hold/calm_flow units are safe but can freeze the last seconds.
    static_penalty = 0.0
    if activity < args.min_activity:
        static_penalty += (args.min_activity - activity) / max(args.min_activity, 1e-6)
    if motion_event == "pose_hold":
        static_penalty += 0.65
    if motion_event == "calm_flow" and activity < args.min_activity * 0.75:
        static_penalty += 0.35

    active_bonus = 0.0
    if motion_event in {"arm_flourish", "high_tension", "support_shift", "build_up", "neutral_flow"}:
        active_bonus += min(activity / max(args.min_activity, 1e-6), 3.0) * 0.15

    score = (
        args.visual_weight * visual
        + args.quality_weight * quality
        + args.safety_weight * safety
        + args.activity_weight * activity
        + active_bonus
        + args.event_weight * ematch
        + args.diversity_weight * div
        + args.music_distinct_weight * mbias
        - args.static_penalty_weight * static_penalty
        - args.emotion_weight * ed
        - args.transition_weight * tc
    )

    # explicit boost: force accent / section_change to activate expressive events
    if music_ev["event"] in {"accent", "climax"} and motion_event in {"high_tension", "arm_flourish"}:
        score += 0.35 * args.event_weight
    if music_ev["event"] == "section_change" and motion_event in {"support_shift", "build_up", "arm_flourish"}:
        score += 0.35 * args.event_weight

    parts = {
        "visual": visual,
        "quality": quality,
        "safety": safety,
        "activity": activity,
        "event_match": ematch,
        "emotion_distance": ed,
        "transition_cost": tc,
        "diversity": div,
        "music_event": music_ev["event"],
        "music_bias": mbias,
        "static_penalty": static_penalty,
        "active_bonus": active_bonus,
        "music_slot_start": music_ev.get("start"),
        "music_slot_end": music_ev.get("end"),
        "accent_density": music_ev.get("accent_density"),
        "section_density": music_ev.get("section_density"),
        "release_density": music_ev.get("release_density"),
        "arousal_norm": music_ev.get("arousal_norm"),
        "tension_norm": music_ev.get("tension_norm"),
        "calm_norm": music_ev.get("calm_norm"),
    }
    return float(score), parts


def is_too_close_same_source(cand: Dict[str, Any], chosen: List[Dict[str, Any]], gap: int) -> bool:
    if gap <= 0:
        return False
    sid = int(cand.get("source_id", -1))
    st = int(cand.get("source_start", cand.get("start", -10**9)))
    for c in chosen:
        if sid != int(c.get("source_id", -2)):
            continue
        pst = int(c.get("source_start", c.get("start", 10**9)))
        if abs(st - pst) < gap:
            return True
    return False


def build_canvas(chosen: List[Dict[str, Any]], num_frames: int) -> np.ndarray:
    pieces = []
    prev_motion = None
    prev_event = "neutral_flow"

    for c in chosen:
        m = c["_motion"]
        ev = c.get("event_type", "neutral_flow")
        k = int(c.get("transition_len", 0))
        if prev_motion is not None and k > 0:
            pieces.append(make_transition(prev_motion, m, k))
        pieces.append(m)
        prev_motion = m
        prev_event = ev

    if pieces:
        y = np.concatenate(pieces, axis=0).astype(np.float32)
    else:
        y = np.zeros((num_frames, 151), dtype=np.float32)

    if len(y) < num_frames:
        pad = np.repeat(y[-1:, :], num_frames - len(y), axis=0) if len(y) else np.zeros((num_frames, 151), dtype=np.float32)
        y = np.concatenate([y, pad], axis=0)
    y = y[:num_frames].copy()
    y[:, ROOT_X] = 0.0
    y[:, ROOT_Z] = 0.0
    return y.astype(np.float32)


def apply_start_anchor(y: np.ndarray, start_pose_path: str = "", blend_frames: int = 8) -> np.ndarray:
    if not start_pose_path:
        default = Path("data/dunhuang_dynamic_event_rag/v20_common_start_pose.npy")
        if default.is_file():
            start_pose_path = str(default)
    if not start_pose_path or not Path(start_pose_path).is_file():
        return y

    s = np.load(start_pose_path).astype(np.float32).reshape(-1)
    if s.shape[0] != 151:
        return y

    z = y.copy()
    bf = max(1, min(int(blend_frames), len(z)))
    z[0, CONTACT] = s[CONTACT]
    z[0, ROOT_Y] = s[ROOT_Y]
    z[0, ROT] = s[ROT]
    z[:, ROOT_X] = 0.0
    z[:, ROOT_Z] = 0.0
    for t in range(1, bf):
        a = smoothstep(t / max(bf - 1, 1))
        z[t, ROOT_Y] = (1 - a) * s[ROOT_Y] + a * z[t, ROOT_Y]
        z[t, ROT] = (1 - a) * s[ROT] + a * z[t, ROT]
    return z.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_db", required=True)
    ap.add_argument("--music_events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--beam_size", type=int, default=16)
    ap.add_argument("--target_event_count", type=int, default=4)
    ap.add_argument("--candidate_top_k", type=int, default=5000)
    ap.add_argument("--min_source_gap", type=int, default=240)

    ap.add_argument("--visual_weight", type=float, default=0.75)
    ap.add_argument("--quality_weight", type=float, default=0.80)
    ap.add_argument("--safety_weight", type=float, default=0.25)
    ap.add_argument("--event_weight", type=float, default=1.20)
    ap.add_argument("--emotion_weight", type=float, default=1.00)
    ap.add_argument("--transition_weight", type=float, default=0.35)
    ap.add_argument("--diversity_weight", type=float, default=0.45)
    ap.add_argument("--activity_weight", type=float, default=0.18)
    ap.add_argument("--min_activity", type=float, default=0.035)
    ap.add_argument("--static_penalty_weight", type=float, default=1.20)
    ap.add_argument("--late_motion_bonus", type=float, default=0.45)
    ap.add_argument("--coverage_weight", type=float, default=0.35)
    ap.add_argument("--music_distinct_weight", type=float, default=0.08)

    ap.add_argument("--allow_variable_len", type=int, default=1)
    ap.add_argument("--start_pose", default="")
    ap.add_argument("--start_anchor_blend", type=int, default=8)
    args = ap.parse_args()

    items = load_event_db(args.event_db)

    # sort by quality first, but keep many candidates to allow expressive events
    items = sorted(
        items,
        key=lambda x: (
            get_desc(x, "quality_score", get_desc(x, "visual_score", 0.0))
            + 0.15 * get_desc(x, "upper_activity", 0.0)
            + 0.10 * get_desc(x, "style_tension", 0.0)
        ),
        reverse=True,
    )[: max(1, args.candidate_top_k)]

    normalize_affects(items)

    # load motions only for candidates
    usable = []
    for it in items:
        p = it.get("pkl", it.get("path", it.get("motion_path", "")))
        if not p:
            continue
        try:
            m = localize_root(load_motion(p))
            if len(m) < 8:
                continue
            it = dict(it)
            it["_motion"] = m
            it["event_type"] = get_event_type(it)
            usable.append(it)
        except Exception:
            continue

    if not usable:
        raise RuntimeError("No usable motion candidates loaded from event db.")

    music, music_events = load_music_events(args.music_events)

    # beam item: (score, chosen, frame_est, prev_candidate)
    beam = [(0.0, [], 0, None)]

    for slot in range(args.target_event_count):
        new_beam = []
        for base_score, chosen, frame_est, prev in beam:
            slot_len = max(1, int(np.ceil(len(music) / max(args.target_event_count, 1))))
            slot_start = int(round(slot * len(music) / max(args.target_event_count, 1)))
            music_ev = music_slot(music, music_events, slot_start, slot_len)
            music_ev["music_key"] = Path(args.music_events).stem

            for cand in usable:
                p = cand.get("pkl", cand.get("path", ""))
                if any(c.get("pkl", c.get("path", "")) == p for c in chosen):
                    continue
                if is_too_close_same_source(cand, chosen, args.min_source_gap):
                    continue

                # V20E: prevent middle/late slots from becoming static.
                ev_type = cand.get("event_type", "neutral_flow")
                cand_activity = (
                    get_desc(cand, "upper_activity", 0.0)
                    + 0.6 * get_desc(cand, "torso_activity", 0.0)
                    + 0.3 * get_desc(cand, "lower_activity", 0.0)
                )

                # After the first slot, avoid very static pose holds unless music is clearly calm/release.
                if slot >= 1 and ev_type == "pose_hold" and music_ev["event"] not in {"calm_flow", "release"}:
                    continue

                # In the last two slots, require enough visible activity.
                if slot >= max(1, args.target_event_count - 2) and cand_activity < args.min_activity:
                    continue

                sc, parts = candidate_score(cand, prev, chosen, music_ev, args)
                k = 0
                if prev is not None:
                    k = transition_len_rule(prev.get("event_type", "neutral_flow"), cand["event_type"], music_ev["event"])

                length = int(cand.get("length", len(cand["_motion"])))
                next_frame_raw = frame_est + k + length

                # Prefer schedules that actually place motion inside the 150-frame window.
                # If an event starts too late, it only contributes 0~1 frames and causes tail freeze.
                remaining = max(args.num_frames - frame_est, 1)
                visible_len = max(0, min(length, args.num_frames - frame_est - k))
                coverage = visible_len / max(length, 1)
                if coverage < 0.35:
                    continue

                sc = sc + args.coverage_weight * coverage

                # late slots should still move, not just pad last frame
                if slot >= 1 and ev_type in {"arm_flourish", "high_tension", "support_shift", "build_up", "neutral_flow"}:
                    sc += args.late_motion_bonus * min(cand_activity / max(args.min_activity, 1e-6), 2.0)

                next_frame = min(args.num_frames - 1, next_frame_raw)

                cand2 = dict(cand)
                cand2["slot"] = slot
                cand2["transition_len"] = int(k)
                cand2["score_parts"] = parts
                cand2["slot_score"] = float(sc)
                cand2["start_frame_est"] = int(frame_est)

                new_beam.append((base_score + sc, chosen + [cand2], next_frame, cand2))

        if not new_beam:
            raise RuntimeError(
                "Beam search failed. Try smaller min_source_gap or larger candidate_top_k."
            )
        new_beam.sort(key=lambda z: z[0], reverse=True)
        beam = new_beam[: args.beam_size]

    best_score, chosen, _, _ = beam[0]
    y = build_canvas(chosen, args.num_frames)
    y = apply_start_anchor(y, args.start_pose, args.start_anchor_blend)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, y[None].astype(np.float32))

    sched = []
    for c in chosen:
        rec = {}
        for k, v in c.items():
            if k in {"_motion", "_affect_norm"}:
                continue
            rec[k] = json_safe(v)
        sched.append(rec)

    report = {
        "scheduler": "v20c_event_graph_phrase",
        "out": str(out),
        "event_db": args.event_db,
        "music_events": args.music_events,
        "score": float(best_score),
        "schedule": sched,
    }
    out.with_suffix(".schedule_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
