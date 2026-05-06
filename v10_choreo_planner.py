#!/usr/bin/env python3
"""
V10 Choreo Planning utilities for EDGE.

Environment-controlled planning:
- EDGE_V10_MODE=dual_auto_mid | manual_multiunit | upperdance_rag | auto_multiunit
- EDGE_V10_MID_FRAMES=50,100
- EDGE_V10_AUTO_MID_COUNT=2 or 3
- EDGE_V10_RAG_DB=/path/to/rag.npz
- EDGE_V10_MANUAL_MID_POSES=/path/mid1.npy,/path/mid2.npy

Important compatibility:
- This version writes both V10-native fields:
    items, mid_poses, mid_pose_frames, unit_paths
  and legacy diagnostic fields:
    keyframes
- planner_edit_loop.py expects plan["keyframes"], so every V10 plan now
  directly reports transition_jerk/contact_phase_break without manual conversion.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CONTACT_SLICE = slice(0, 4)
ROOT_XZ_IDX = [4, 6]
ROT_START = 7
N_JOINTS = 24
ROT_DIM = 6

# Coarse SMPL-like joint groups.  The exact semantic mapping is imperfect, but
# this is consistent with the existing EDGE 151D representation and diagnostic usage.
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def parse_csv(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def parse_frames(text: str, count: int, num_frames: int = 150) -> List[int]:
    if text:
        raw = parse_csv(text)
        if len(raw) != count:
            raise ValueError(f"Expected {count} frame values, got {len(raw)} from: {text}")
        frames = []
        for item in raw:
            v = float(item)
            if 0.0 < v < 1.0:
                v = v * (num_frames - 1)
            frames.append(int(round(v)))
        return [max(1, min(num_frames - 2, f)) for f in frames]

    if count == 2 and num_frames <= 180:
        return [int(round(num_frames / 3)), int(round(2 * num_frames / 3))]

    return [
        int(round((i + 1) * (num_frames - 1) / (count + 1)))
        for i in range(count)
    ]


def default_dual_mid_frames(num_frames: int = 150) -> List[int]:
    text = os.environ.get("EDGE_V10_MID_FRAMES", "")
    if text:
        return parse_frames(text, 2, num_frames=num_frames)

    f1 = env_int("EDGE_V10_MID_FRAME1", 50 if num_frames == 150 else int(round(num_frames / 3)))
    f2 = env_int("EDGE_V10_MID_FRAME2", 100 if num_frames == 150 else int(round(2 * num_frames / 3)))
    min_gap = env_int("EDGE_V10_MIN_GAP", 35)

    f1 = max(1, min(num_frames - 2, f1))
    f2 = max(1, min(num_frames - 2, f2))
    if f2 - f1 < min_gap:
        f1 = max(1, int(round(num_frames / 3)))
        f2 = min(num_frames - 2, max(f1 + min_gap, int(round(2 * num_frames / 3))))

    return [f1, f2]


def as_unit_t151(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected unit [T,151] or [151,T], got {arr.shape}")
    if arr.shape[-1] == 151:
        return arr
    if arr.shape[0] == 151:
        return arr.T
    raise ValueError(f"Expected one dim to be 151, got {arr.shape}")


def rot_view(unit: np.ndarray) -> np.ndarray:
    unit = as_unit_t151(unit)
    rot = unit[:, ROT_START : ROT_START + N_JOINTS * ROT_DIM]
    if rot.shape[-1] < N_JOINTS * ROT_DIM:
        pad = np.zeros((rot.shape[0], N_JOINTS * ROT_DIM - rot.shape[-1]), dtype=rot.dtype)
        rot = np.concatenate([rot, pad], axis=-1)
    return rot.reshape(rot.shape[0], N_JOINTS, ROT_DIM)


def mean_abs_velocity(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(x, axis=0), axis=-1)))


@dataclass
class UnitScore:
    index: int
    score: float
    upper_activity: float
    lower_activity: float
    root_speed: float
    spatial_range: float
    turning: float
    pose_diversity: float
    contact_change: float
    source_key: str


def score_unit(unit: np.ndarray, index: int = -1, source_key: str = "") -> UnitScore:
    unit = as_unit_t151(unit)

    root_xz = unit[:, ROOT_XZ_IDX]
    root_vel = np.diff(root_xz, axis=0) if len(unit) > 1 else np.zeros((0, 2), dtype=np.float32)
    root_speed = float(np.mean(np.linalg.norm(root_vel, axis=-1))) if len(root_vel) else 0.0
    spatial_range = float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0))) if len(unit) else 0.0

    rot = rot_view(unit)
    upper_activity = mean_abs_velocity(rot[:, UPPER_JOINTS, :])
    lower_activity = mean_abs_velocity(rot[:, LOWER_JOINTS, :])

    if len(rot) > 1:
        center = rot.mean(axis=0, keepdims=True)
        pose_diversity = float(
            np.mean(np.linalg.norm((rot - center).reshape(rot.shape[0], -1), axis=-1))
        )
    else:
        pose_diversity = 0.0

    if root_vel.shape[0] >= 2:
        dirs = root_vel / (np.linalg.norm(root_vel, axis=-1, keepdims=True) + 1e-8)
        dots = np.sum(dirs[1:] * dirs[:-1], axis=-1).clip(-1.0, 1.0)
        turning = float(np.mean(np.arccos(dots)))
    else:
        turning = 0.0

    contact = unit[:, CONTACT_SLICE]
    contact_change = float(np.mean(np.abs(np.diff(contact, axis=0)))) if len(unit) > 1 else 0.0

    # Upper-body-dominant dance score.
    score = (
        env_float("EDGE_V10_SCORE_UPPER_W", 0.45) * upper_activity
        + env_float("EDGE_V10_SCORE_TURN_W", 0.20) * turning
        + env_float("EDGE_V10_SCORE_DIVERSITY_W", 0.20) * pose_diversity
        - env_float("EDGE_V10_SCORE_CONTACT_W", 0.05) * contact_change
        - env_float("EDGE_V10_SCORE_ROOT_PENALTY_W", 0.15) * root_speed
        - env_float("EDGE_V10_SCORE_LOWER_PENALTY_W", 0.05) * max(0.0, lower_activity - upper_activity)
    )

    return UnitScore(
        index=int(index),
        score=float(score),
        upper_activity=float(upper_activity),
        lower_activity=float(lower_activity),
        root_speed=float(root_speed),
        spatial_range=float(spatial_range),
        turning=float(turning),
        pose_diversity=float(pose_diversity),
        contact_change=float(contact_change),
        source_key=str(source_key),
    )


def _candidate_arrays_from_npz(npz: np.lib.npyio.NpzFile) -> List[Tuple[str, np.ndarray]]:
    out = []
    for key in npz.files:
        try:
            arr = np.asarray(npz[key])
        except Exception:
            continue
        if arr.dtype == object:
            continue
        if arr.ndim == 3 and (arr.shape[-1] == 151 or arr.shape[1] == 151):
            out.append((key, arr))
        elif arr.ndim == 2 and (arr.shape[-1] == 151 or arr.shape[0] == 151):
            out.append((key, arr[None]))
    return out


def load_units_from_npz(
    rag_db: str,
    max_units: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    rag_path = Path(rag_db)
    if not rag_path.exists():
        raise FileNotFoundError(f"RAG DB not found: {rag_db}")

    npz = np.load(rag_path, allow_pickle=True)
    arrays = _candidate_arrays_from_npz(npz)
    if not arrays:
        raise ValueError(f"No [N,T,151] or [N,151,T] unit arrays found in {rag_db}. keys={npz.files}")

    def key_rank(item):
        key = item[0].lower()
        rank = 0
        for token in ["unit", "motion", "motions", "clip", "clips", "x"]:
            if token in key:
                rank -= 1
        return (rank, key)

    arrays = sorted(arrays, key=key_rank)

    units: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []
    for key, arr in arrays:
        for i in range(arr.shape[0]):
            try:
                unit = as_unit_t151(arr[i])
            except Exception:
                continue
            units.append(unit.astype(np.float32))
            meta.append({"source_key": key, "source_index": i})
            if max_units is not None and len(units) >= max_units:
                return units, meta

    if not units:
        raise ValueError(f"Could not coerce any units from {rag_db}")

    return units, meta


def choose_diverse_units(
    units: Sequence[np.ndarray],
    meta: Sequence[Dict[str, Any]],
    count: int,
) -> Tuple[List[int], List[UnitScore]]:
    scored = []
    min_upper = env_float("EDGE_V10_MIN_UPPER_ACTIVITY", 0.0)
    max_root = env_float("EDGE_V10_MAX_ROOT_SPEED", 999.0)

    for i, unit in enumerate(units):
        s = score_unit(unit, index=i, source_key=str(meta[i].get("source_key", "")))
        if s.upper_activity >= min_upper and s.root_speed <= max_root:
            scored.append(s)

    if not scored:
        scored = [
            score_unit(unit, index=i, source_key=str(meta[i].get("source_key", "")))
            for i, unit in enumerate(units)
        ]

    scored = sorted(scored, key=lambda x: x.score, reverse=True)

    selected: List[UnitScore] = []
    selected_indices: List[int] = []
    min_score_gap = env_float("EDGE_V10_MIN_SCORE_GAP", 0.0)

    for s in scored:
        too_similar = False
        for prev in selected:
            if (
                abs(s.upper_activity - prev.upper_activity) < min_score_gap
                and abs(s.turning - prev.turning) < min_score_gap
            ):
                too_similar = True
                break

        if too_similar and len(scored) > count:
            continue

        selected.append(s)
        selected_indices.append(s.index)

        if len(selected) >= count:
            break

    return selected_indices[:count], selected[:count]


def _score_parts_from_score(score: UnitScore) -> Dict[str, float | str]:
    return {
        "phase": "v10_upperdance",
        "expressiveness_score": float(score.upper_activity),
        "motion_energy_norm": float(score.score),
        "upper_activity": float(score.upper_activity),
        "lower_activity": float(score.lower_activity),
        "root_speed": float(score.root_speed),
        "spatial_range": float(score.spatial_range),
        "turning": float(score.turning),
        "pose_diversity": float(score.pose_diversity),
        "contact_change": float(score.contact_change),
        "source_key": str(score.source_key),
    }


def _keyframe_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    score = item.get("score", {}) or {}
    if isinstance(score, UnitScore):
        score_parts = _score_parts_from_score(score)
        score_value = float(score.score)
    elif isinstance(score, dict):
        score_parts = {
            "phase": item.get("phase", "v10_upperdance"),
            "expressiveness_score": float(score.get("upper_activity", score.get("expressiveness_score", 0.0))),
            "motion_energy_norm": float(score.get("score", score.get("motion_energy_norm", 0.0))),
            "upper_activity": float(score.get("upper_activity", 0.0)),
            "lower_activity": float(score.get("lower_activity", 0.0)),
            "root_speed": float(score.get("root_speed", 0.0)),
            "spatial_range": float(score.get("spatial_range", 0.0)),
            "turning": float(score.get("turning", 0.0)),
            "pose_diversity": float(score.get("pose_diversity", 0.0)),
            "contact_change": float(score.get("contact_change", 0.0)),
            "source_key": str(score.get("source_key", item.get("source_key", ""))),
        }
        score_value = float(score.get("score", 0.0))
    else:
        score_parts = {"phase": item.get("phase", "v10_manual")}
        score_value = 0.0

    return {
        "frame": int(item.get("frame", 0)),
        "segment_id": int(item.get("rank", -1)),
        "phase": str(score_parts.get("phase", item.get("phase", "v10"))),
        "source": item.get("unit_path", item.get("pose_path", "")),
        "score": score_value,
        "score_parts": score_parts,
    }


def _attach_legacy_keyframes(plan: Dict[str, Any]) -> Dict[str, Any]:
    items = plan.get("items", []) or []
    keyframes = [_keyframe_from_item(item) for item in items]
    plan["keyframes"] = keyframes
    plan["auto_keyframes"] = keyframes
    return plan


def save_unit_mid_assets(
    units: Sequence[np.ndarray],
    scores: Sequence[UnitScore],
    frames: Sequence[int],
    out_prefix: str,
) -> Dict[str, Any]:
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    mid_paths: List[str] = []
    unit_paths: List[str] = []
    items: List[Dict[str, Any]] = []

    for k, (unit, score, frame) in enumerate(zip(units, scores, frames), start=1):
        unit = as_unit_t151(unit)

        # Use the center pose of the retrieved unit as the explicit mid pose.
        mid_idx = int(round((len(unit) - 1) / 2))
        pose = unit[mid_idx].astype(np.float32)

        pose_path = out_prefix_path.parent / f"{out_prefix_path.name}_v10_mid{k}_f{int(frame):03d}.npy"
        unit_path = out_prefix_path.parent / f"{out_prefix_path.name}_v10_mid{k}_f{int(frame):03d}_unit.npy"

        np.save(pose_path, pose)
        np.save(unit_path, unit.astype(np.float32))

        mid_paths.append(str(pose_path))
        unit_paths.append(str(unit_path))

        score_dict = asdict(score)
        items.append(
            {
                "rank": k,
                "frame": int(frame),
                "phase": "v10_upperdance",
                "pose_path": str(pose_path),
                "unit_path": str(unit_path),
                "source_key": score.source_key,
                "score": score_dict,
            }
        )

    plan: Dict[str, Any] = {
        "mode": "upperdance_rag",
        "mid_poses": mid_paths,
        "mid_pose_frames": [int(f) for f in frames],
        "unit_paths": unit_paths,
        "items": items,
    }
    plan = _attach_legacy_keyframes(plan)

    plan_path = out_prefix_path.parent / f"{out_prefix_path.name}_v10_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    plan["plan_path"] = str(plan_path)
    print(f"✅ V10 plan saved with keyframes: {plan_path}")
    return plan


def plan_upperdance_from_rag_db(
    rag_db: str,
    out_prefix: str,
    num_frames: int = 150,
    count: int = 2,
    frames: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    units, meta = load_units_from_npz(
        rag_db,
        max_units=env_int("EDGE_V10_MAX_RAG_UNITS", 20000),
    )
    selected_indices, selected_scores = choose_diverse_units(units, meta, count=count)
    selected_units = [units[i] for i in selected_indices]

    if frames is None:
        if count == 2:
            frames = default_dual_mid_frames(num_frames)
        else:
            frames = parse_frames(os.environ.get("EDGE_V10_MID_FRAMES", ""), count=count, num_frames=num_frames)

    plan = save_unit_mid_assets(selected_units, selected_scores, frames, out_prefix)
    plan["rag_db"] = str(rag_db)
    plan["mode"] = "upperdance_rag"

    # Rewrite with final mode/rag_db included.
    plan_path = Path(plan["plan_path"])
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    return plan


def plan_manual_from_env(
    out_prefix: str,
    num_frames: int = 150,
) -> Optional[Dict[str, Any]]:
    paths = parse_csv(os.environ.get("EDGE_V10_MANUAL_MID_POSES", ""))
    if not paths:
        return None

    frames = parse_frames(
        os.environ.get("EDGE_V10_MANUAL_MID_FRAMES", ""),
        count=len(paths),
        num_frames=num_frames,
    )

    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Manual mid pose not found: {p}")

    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    items = [
        {
            "rank": i + 1,
            "frame": int(frame),
            "phase": "v10_manual_multiunit",
            "pose_path": path,
            "source_key": "manual",
            "score": {
                "score": 0.0,
                "upper_activity": 0.0,
                "lower_activity": 0.0,
                "root_speed": 0.0,
                "spatial_range": 0.0,
                "turning": 0.0,
                "pose_diversity": 0.0,
                "contact_change": 0.0,
                "source_key": "manual",
            },
        }
        for i, (path, frame) in enumerate(zip(paths, frames))
    ]

    plan: Dict[str, Any] = {
        "mode": "manual_multiunit",
        "mid_poses": paths,
        "mid_pose_frames": [int(f) for f in frames],
        "unit_paths": [],
        "items": items,
    }
    plan = _attach_legacy_keyframes(plan)

    plan_path = out_prefix_path.parent / f"{out_prefix_path.name}_v10_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    plan["plan_path"] = str(plan_path)
    print(f"✅ V10 manual plan saved with keyframes: {plan_path}")
    return plan
