#!/usr/bin/env python3
"""Unified V10 Choreo Planner for EDGE.

Patch purpose
-------------
This version permanently fixes environment leakage from Step2 Manual Unit mode.

Bug fixed:
    If EDGE_V10_MANUAL_UNITS was exported for Step2, then Step1/Step3/Step4
    could accidentally reuse those two manual units. Step4 has 3 frames, so it
    crashed with:

        ValueError: planned_units count 2 != frames count 3

Fix:
    Manual unit / manual pose environment variables are honored only when
    EDGE_V10_MODE=manual_multiunit. In all other modes they are ignored even if
    still present in the parent shell.

It also keeps the Unified Planner v3 features:
- Unified Unit representation.
- Config-driven Step1/Step3/Step4.
- Manual Unit IDs for fair Manual-vs-Auto ablation.
- Greedy or Beam Search global planning.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CONTACT_SLICE = slice(0, 4)
ROOT_XZ_IDX = [4, 6]
ROT_START = 7
N_JOINTS = 24
ROT_DIM = 6
ROT_SLICE = slice(7, 151)

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


def env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_csv(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def parse_frames(text: str, count: int, num_frames: int = 150) -> List[int]:
    if text:
        raw = parse_csv(text)
        if len(raw) != count:
            raise ValueError(f"Expected {count} frame values, got {len(raw)} from: {text}")
        frames: List[int] = []
        for item in raw:
            v = float(item)
            if 0.0 < v < 1.0:
                v = v * (num_frames - 1)
            frames.append(int(round(v)))
        return [max(1, min(num_frames - 2, f)) for f in frames]

    return [int(round((i + 1) * (num_frames - 1) / (count + 1))) for i in range(count)]


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


def default_frames_for_mode(mode: str, num_frames: int = 150) -> List[int]:
    mode = str(mode).strip().lower()

    explicit = os.environ.get("EDGE_V10_MID_FRAMES", "")
    if explicit:
        raw = parse_csv(explicit)
        return parse_frames(explicit, count=len(raw), num_frames=num_frames)

    if mode in {"dual_auto_mid", "upperdance_rag"}:
        return default_dual_mid_frames(num_frames=num_frames)

    if mode == "auto_multiunit":
        return [40, 75, 110] if num_frames == 150 else parse_frames("", count=3, num_frames=num_frames)

    if mode == "manual_multiunit":
        manual_units = parse_csv(os.environ.get("EDGE_V10_MANUAL_UNITS", ""))
        manual_poses = parse_csv(os.environ.get("EDGE_V10_MANUAL_MID_POSES", ""))
        manual_count = len(manual_units) or len(manual_poses) or 2
        manual_frames = os.environ.get("EDGE_V10_MANUAL_MID_FRAMES", "")
        return parse_frames(manual_frames, count=manual_count, num_frames=num_frames)

    return default_dual_mid_frames(num_frames=num_frames)


@dataclass
class PlannerConfig:
    mode: str
    num_frames: int
    mid_frames: List[int]
    score_weights: Dict[str, float] = field(default_factory=dict)
    transition_weights: Dict[str, float] = field(default_factory=dict)
    manual_units: List[str] = field(default_factory=list)
    manual_mid_poses: List[str] = field(default_factory=list)
    search_method: str = "greedy"
    top_k: int = 64
    beam_width: int = 8
    candidate_cap: int = 600
    min_upper_activity: float = 0.0
    max_root_speed: float = 999.0
    rag_summary_mode: str = "mean"

    @property
    def count(self) -> int:
        return len(self.mid_frames)


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
    source_index: int = -1
    unit_id: str = ""
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    transition_cost: float = 0.0
    transition_score: float = 0.0
    path_score: float = 0.0
    emission_score: float = 0.0


@dataclass
class PlannedUnit:
    unit: np.ndarray
    score: UnitScore
    meta: Dict[str, Any]


def build_config_from_env(num_frames: int = 150) -> PlannerConfig:
    mode = env_str("EDGE_V10_MODE", "dual_auto_mid").lower()
    mid_frames = default_frames_for_mode(mode=mode, num_frames=num_frames)

    score_weights = {
        "upper": env_float("EDGE_V10_SCORE_UPPER_W", 0.45),
        "turn": env_float("EDGE_V10_SCORE_TURN_W", 0.15),
        "diversity": env_float("EDGE_V10_SCORE_DIVERSITY_W", 0.20),
        "contact": env_float("EDGE_V10_SCORE_CONTACT_W", 0.08),
        "root_penalty": env_float("EDGE_V10_SCORE_ROOT_PENALTY_W", 0.12),
        "lower_penalty": env_float("EDGE_V10_SCORE_LOWER_PENALTY_W", 0.05),
    }

    transition_weights = {
        "start": env_float("EDGE_V10_START_COMPAT_W", env_float("EDGE_V10_ENTRY_COMPAT_W", 0.35)),
        "end": env_float("EDGE_V10_END_COMPAT_W", env_float("EDGE_V10_EXIT_COMPAT_W", 0.30)),
        "pose": env_float("EDGE_V10_TRANS_POSE_W", 0.35),
        "contact": env_float("EDGE_V10_TRANS_CONTACT_W", 0.10),
        "root": env_float("EDGE_V10_TRANS_ROOT_W", 0.05),
        "same_source": env_float("EDGE_V10_SAME_SOURCE_PENALTY", 0.03),
    }

    if mode == "upperdance_rag":
        score_weights["upper"] = env_float("EDGE_V10_SCORE_UPPER_W", 0.55)
        score_weights["turn"] = env_float("EDGE_V10_SCORE_TURN_W", 0.20)
        score_weights["diversity"] = env_float("EDGE_V10_SCORE_DIVERSITY_W", 0.20)
        score_weights["root_penalty"] = env_float("EDGE_V10_SCORE_ROOT_PENALTY_W", 0.10)

    # Permanent env-isolation fix:
    # Manual controls are meaningful only in manual_multiunit mode.  This makes
    # Step1/Step3/Step4 robust even if the user's shell still has
    # EDGE_V10_MANUAL_UNITS exported from Step2.
    manual_units = parse_csv(os.environ.get("EDGE_V10_MANUAL_UNITS", "")) if mode == "manual_multiunit" else []
    manual_mid_poses = parse_csv(os.environ.get("EDGE_V10_MANUAL_MID_POSES", "")) if mode == "manual_multiunit" else []

    if mode != "manual_multiunit" and (
        os.environ.get("EDGE_V10_MANUAL_UNITS") or os.environ.get("EDGE_V10_MANUAL_MID_POSES")
    ):
        print(
            f"ℹ️ Ignoring manual unit/pose env because EDGE_V10_MODE={mode}. "
            "Manual controls are only active in manual_multiunit mode."
        )

    search_default = "manual" if mode == "manual_multiunit" else ("beam" if mode == "auto_multiunit" else "greedy")
    return PlannerConfig(
        mode=mode,
        num_frames=num_frames,
        mid_frames=mid_frames,
        score_weights=score_weights,
        transition_weights=transition_weights,
        manual_units=manual_units,
        manual_mid_poses=manual_mid_poses,
        search_method=env_str("EDGE_V10_SEARCH_METHOD", search_default).lower(),
        top_k=env_int("EDGE_V10_TOP_K", 64),
        beam_width=env_int("EDGE_V10_BEAM_WIDTH", 8),
        candidate_cap=env_int("EDGE_V10_CANDIDATE_CAP", 600),
        min_upper_activity=env_float("EDGE_V10_MIN_UPPER_ACTIVITY", 0.0),
        max_root_speed=env_float("EDGE_V10_MAX_ROOT_SPEED", 999.0),
        rag_summary_mode=env_str("EDGE_RAG_SUMMARY_MODE", "mean").lower(),
    )


def as_unit_t151(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected unit [T,151] or [151,T], got {arr.shape}")
    if arr.shape[-1] == 151:
        return arr
    if arr.shape[0] == 151:
        return arr.T
    raise ValueError(f"Expected one dim to be 151, got {arr.shape}")


def load_151_pose(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None

    arr = np.load(str(p), allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        if "motion" in d:
            arr = d["motion"]
        elif "pose" in d:
            arr = d["pose"]

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[-1] == 151:
            return arr[0].astype(np.float32)
        if arr.shape[0] == 151:
            return arr[:, 0].astype(np.float32)
    arr = arr.reshape(-1)
    if arr.shape[0] != 151:
        raise ValueError(f"Expected 151D pose: {path}, got {arr.shape}")
    return arr.astype(np.float32)


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


def pose_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape[0] != 151 or b.shape[0] != 151:
        return 0.0

    rot = float(np.sqrt(np.mean((a[ROT_SLICE] - b[ROT_SLICE]) ** 2)))
    root_y = float(abs(a[5] - b[5]))
    contact = float(np.mean(np.abs(a[CONTACT_SLICE] - b[CONTACT_SLICE])))
    return rot + 0.20 * root_y + 0.10 * contact


def contact_phase_distance(unit_a: np.ndarray, unit_b: np.ndarray) -> float:
    a = as_unit_t151(unit_a)
    b = as_unit_t151(unit_b)
    return float(np.mean(np.abs(a[-1, CONTACT_SLICE] - b[0, CONTACT_SLICE])))


def root_direction_distance(unit_a: np.ndarray, unit_b: np.ndarray) -> float:
    a = as_unit_t151(unit_a)
    b = as_unit_t151(unit_b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va = a[-1, ROOT_XZ_IDX] - a[-2, ROOT_XZ_IDX]
    vb = b[1, ROOT_XZ_IDX] - b[0, ROOT_XZ_IDX]
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    cos = float(np.dot(va, vb) / (na * nb + 1e-8))
    cos = float(np.clip(cos, -1.0, 1.0))
    return 1.0 - cos


def transition_cost(unit_a: np.ndarray, unit_b: np.ndarray, weights: Dict[str, float]) -> float:
    pose = pose_distance(as_unit_t151(unit_a)[-1], as_unit_t151(unit_b)[0])
    contact = contact_phase_distance(unit_a, unit_b)
    root = root_direction_distance(unit_a, unit_b)
    return (
        float(weights.get("pose", 0.35)) * pose
        + float(weights.get("contact", 0.10)) * contact
        + float(weights.get("root", 0.05)) * root
    )


def score_unit(
    unit: np.ndarray,
    index: int = -1,
    source_key: str = "",
    source_index: int = -1,
    weights: Optional[Dict[str, float]] = None,
) -> UnitScore:
    unit = as_unit_t151(unit)
    weights = weights or {
        "upper": env_float("EDGE_V10_SCORE_UPPER_W", 0.45),
        "turn": env_float("EDGE_V10_SCORE_TURN_W", 0.15),
        "diversity": env_float("EDGE_V10_SCORE_DIVERSITY_W", 0.20),
        "contact": env_float("EDGE_V10_SCORE_CONTACT_W", 0.08),
        "root_penalty": env_float("EDGE_V10_SCORE_ROOT_PENALTY_W", 0.12),
        "lower_penalty": env_float("EDGE_V10_SCORE_LOWER_PENALTY_W", 0.05),
    }

    root_xz = unit[:, ROOT_XZ_IDX]
    root_vel = np.diff(root_xz, axis=0) if len(unit) > 1 else np.zeros((0, 2), dtype=np.float32)
    root_speed = float(np.mean(np.linalg.norm(root_vel, axis=-1))) if len(root_vel) else 0.0
    spatial_range = float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0))) if len(unit) else 0.0

    rot = rot_view(unit)
    upper_activity = mean_abs_velocity(rot[:, UPPER_JOINTS, :])
    lower_activity = mean_abs_velocity(rot[:, LOWER_JOINTS, :])

    if len(rot) > 1:
        center = rot.mean(axis=0, keepdims=True)
        pose_diversity = float(np.mean(np.linalg.norm((rot - center).reshape(rot.shape[0], -1), axis=-1)))
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

    score = (
        float(weights.get("upper", 0.45)) * upper_activity
        + float(weights.get("turn", 0.15)) * turning
        + float(weights.get("diversity", 0.20)) * pose_diversity
        - float(weights.get("contact", 0.08)) * contact_change
        - float(weights.get("root_penalty", 0.12)) * root_speed
        - float(weights.get("lower_penalty", 0.05)) * max(0.0, lower_activity - upper_activity)
    )

    unit_id = f"{source_key}:{int(source_index)}" if source_key else f"unit_{int(index)}"
    return UnitScore(
        index=int(index),
        score=float(score),
        emission_score=float(score),
        upper_activity=float(upper_activity),
        lower_activity=float(lower_activity),
        root_speed=float(root_speed),
        spatial_range=float(spatial_range),
        turning=float(turning),
        pose_diversity=float(pose_diversity),
        contact_change=float(contact_change),
        source_key=str(source_key),
        source_index=int(source_index),
        unit_id=unit_id,
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
    global_index = 0
    for key, arr in arrays:
        for i in range(arr.shape[0]):
            try:
                unit = as_unit_t151(arr[i])
            except Exception:
                continue
            units.append(unit.astype(np.float32))
            meta.append(
                {
                    "source_key": key,
                    "source_index": int(i),
                    "global_index": int(global_index),
                    "unit_id": f"{key}:{int(i)}",
                    "alt_unit_id": f"unit_{int(global_index):04d}",
                }
            )
            global_index += 1
            if max_units is not None and len(units) >= max_units:
                return units, meta

    if not units:
        raise ValueError(f"Could not coerce any units from {rag_db}")

    return units, meta


class RAGUnitDB:
    def __init__(self, rag_db: str, max_units: Optional[int] = None):
        self.rag_db = rag_db
        self.units, self.meta = load_units_from_npz(rag_db, max_units=max_units)

    def resolve_unit_ids(self, unit_ids: Sequence[str]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        resolved_units: List[np.ndarray] = []
        resolved_meta: List[Dict[str, Any]] = []

        id_to_index: Dict[str, int] = {}
        for i, m in enumerate(self.meta):
            source_key = str(m.get("source_key", ""))
            source_index = int(m.get("source_index", i))
            global_index = int(m.get("global_index", i))
            keys = {
                str(global_index),
                f"unit_{global_index}",
                f"unit_{global_index:04d}",
                f"{source_key}:{source_index}",
                f"{source_key}:{source_index:04d}",
                str(m.get("unit_id", "")),
                str(m.get("alt_unit_id", "")),
            }
            for key in keys:
                if key:
                    id_to_index[key] = i

        for raw in unit_ids:
            token = str(raw).strip()
            if not token:
                continue

            idx: Optional[int] = id_to_index.get(token)

            if idx is None:
                m = re.fullmatch(r"unit_0*(\d+)", token)
                if m:
                    idx = id_to_index.get(str(int(m.group(1))))

            if idx is None and ":" in token:
                source_key, source_idx_text = token.rsplit(":", 1)
                m = re.fullmatch(r"(?:unit_)?0*(\d+)", source_idx_text.strip())
                if m:
                    idx = id_to_index.get(f"{source_key}:{int(m.group(1))}")

            if idx is None:
                examples = [str(m.get("unit_id", "")) for m in self.meta[:5]]
                raise KeyError(
                    f"Cannot resolve manual unit id {token!r}. "
                    f"Use global index like 1042/unit_1042 or source_key:index. "
                    f"Examples: {examples}"
                )

            meta = dict(self.meta[idx])
            meta["requested_unit_id"] = token
            resolved_units.append(self.units[idx])
            resolved_meta.append(meta)

        return resolved_units, resolved_meta


class UnifiedChoreoPlanner:
    def __init__(self, config: PlannerConfig, rag_db: Optional[str] = None):
        self.config = config
        self.rag_db = rag_db
        self.db: Optional[RAGUnitDB] = None
        if rag_db:
            self.db = RAGUnitDB(
                rag_db,
                max_units=env_int("EDGE_V10_MAX_RAG_UNITS", 20000),
            )

    def _score_all(self) -> List[UnitScore]:
        if self.db is None:
            raise RuntimeError("RAG DB is required for auto/manual-unit planning.")

        scores: List[UnitScore] = []
        for i, unit in enumerate(self.db.units):
            m = self.db.meta[i]
            s = score_unit(
                unit,
                index=i,
                source_key=str(m.get("source_key", "")),
                source_index=int(m.get("source_index", i)),
                weights=self.config.score_weights,
            )
            if s.upper_activity >= self.config.min_upper_activity and s.root_speed <= self.config.max_root_speed:
                scores.append(s)

        if not scores:
            for i, unit in enumerate(self.db.units):
                m = self.db.meta[i]
                scores.append(
                    score_unit(
                        unit,
                        index=i,
                        source_key=str(m.get("source_key", "")),
                        source_index=int(m.get("source_index", i)),
                        weights=self.config.score_weights,
                    )
                )

        return sorted(scores, key=lambda x: x.score, reverse=True)[: max(self.config.candidate_cap, self.config.top_k)]

    def _annotate_path(
        self,
        indices: Sequence[int],
        base_scores: Dict[int, UnitScore],
        start_pose: Optional[np.ndarray],
        end_pose: Optional[np.ndarray],
        path_score: float = 0.0,
    ) -> List[PlannedUnit]:
        if self.db is None:
            raise RuntimeError("RAG DB is required for path annotation.")

        planned: List[PlannedUnit] = []
        prev_unit: Optional[np.ndarray] = None

        for rank, idx in enumerate(indices):
            unit = self.db.units[idx]
            base = base_scores[idx]
            s = UnitScore(**asdict(base))

            if rank == 0:
                s.entry_cost = pose_distance(start_pose, unit[0])
                s.transition_cost = float(self.config.transition_weights.get("start", 0.35)) * s.entry_cost
            else:
                assert prev_unit is not None
                s.entry_cost = pose_distance(prev_unit[-1], unit[0])
                s.transition_cost = transition_cost(prev_unit, unit, self.config.transition_weights)

            if rank == len(indices) - 1:
                s.exit_cost = pose_distance(unit[-1], end_pose)
            else:
                next_unit = self.db.units[indices[rank + 1]]
                s.exit_cost = pose_distance(unit[-1], next_unit[0])

            s.transition_score = s.score - s.transition_cost
            s.path_score = float(path_score)

            meta = dict(self.db.meta[idx])
            meta["rank"] = rank + 1
            planned.append(PlannedUnit(unit=unit, score=s, meta=meta))
            prev_unit = unit

        return planned

    def _select_manual_units(
        self,
        start_pose: Optional[np.ndarray],
        end_pose: Optional[np.ndarray],
    ) -> List[PlannedUnit]:
        if self.db is None:
            raise RuntimeError("RAG DB is required when EDGE_V10_MANUAL_UNITS is set.")

        units, meta = self.db.resolve_unit_ids(self.config.manual_units)
        base_scores: Dict[int, UnitScore] = {}
        indices: List[int] = []

        for unit, m in zip(units, meta):
            idx = int(m["global_index"])
            indices.append(idx)
            base_scores[idx] = score_unit(
                unit,
                index=idx,
                source_key=str(m.get("source_key", "")),
                source_index=int(m.get("source_index", idx)),
                weights=self.config.score_weights,
            )

        return self._annotate_path(indices, base_scores, start_pose, end_pose, path_score=0.0)

    def _select_greedy(
        self,
        start_pose: Optional[np.ndarray],
        end_pose: Optional[np.ndarray],
    ) -> List[PlannedUnit]:
        if self.db is None:
            raise RuntimeError("RAG DB is required for greedy planning.")

        base_list = self._score_all()
        base_scores = {s.index: s for s in base_list}
        selected: List[int] = []
        used_sources: set[str] = set()
        prev_unit: Optional[np.ndarray] = None

        for slot in range(self.config.count):
            scored_slot: List[Tuple[float, int]] = []
            is_last = slot == self.config.count - 1

            for base in base_list:
                if base.index in selected:
                    continue

                unit = self.db.units[base.index]
                if prev_unit is None:
                    cost = float(self.config.transition_weights.get("start", 0.35)) * pose_distance(start_pose, unit[0])
                else:
                    cost = transition_cost(prev_unit, unit, self.config.transition_weights)

                if is_last:
                    cost += float(self.config.transition_weights.get("end", 0.30)) * pose_distance(unit[-1], end_pose)

                if base.source_key in used_sources:
                    cost += float(self.config.transition_weights.get("same_source", 0.03))

                scored_slot.append((base.score - cost, base.index))

            if not scored_slot:
                break

            _, best_idx = max(scored_slot, key=lambda x: x[0])
            selected.append(best_idx)
            used_sources.add(base_scores[best_idx].source_key)
            prev_unit = self.db.units[best_idx]

        path_score = float(sum(base_scores[i].score for i in selected))
        return self._annotate_path(selected, base_scores, start_pose, end_pose, path_score=path_score)

    def _select_beam(
        self,
        start_pose: Optional[np.ndarray],
        end_pose: Optional[np.ndarray],
    ) -> List[PlannedUnit]:
        if self.db is None:
            raise RuntimeError("RAG DB is required for beam planning.")

        base_list = self._score_all()[: max(1, self.config.top_k)]
        base_scores = {s.index: s for s in base_list}
        if not base_list:
            raise RuntimeError("No V10 unit candidates found.")

        beams: List[Tuple[float, List[int], set[str]]] = [(0.0, [], set())]

        for slot in range(self.config.count):
            new_beams: List[Tuple[float, List[int], set[str]]] = []
            is_last = slot == self.config.count - 1

            for cur_score, path, used_sources in beams:
                prev_unit = self.db.units[path[-1]] if path else None

                for cand in base_list:
                    idx = cand.index
                    if idx in path:
                        continue

                    unit = self.db.units[idx]
                    step_score = cand.score

                    if prev_unit is None:
                        step_score -= float(self.config.transition_weights.get("start", 0.35)) * pose_distance(start_pose, unit[0])
                    else:
                        step_score -= transition_cost(prev_unit, unit, self.config.transition_weights)

                    if is_last:
                        step_score -= float(self.config.transition_weights.get("end", 0.30)) * pose_distance(unit[-1], end_pose)

                    if cand.source_key in used_sources:
                        step_score -= float(self.config.transition_weights.get("same_source", 0.03))

                    next_used = set(used_sources)
                    next_used.add(cand.source_key)
                    new_beams.append((cur_score + step_score, path + [idx], next_used))

            if not new_beams:
                break

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[: max(1, self.config.beam_width)]

        if not beams:
            raise RuntimeError("Beam search produced no valid path.")

        best_score, best_path, _ = max(beams, key=lambda x: x[0])
        return self._annotate_path(best_path, base_scores, start_pose, end_pose, path_score=float(best_score))

    def plan(
        self,
        out_prefix: str,
        start_pose_path: str = "",
        end_pose_path: str = "",
    ) -> Dict[str, Any]:
        start_pose = load_151_pose(start_pose_path)
        end_pose = load_151_pose(end_pose_path)

        if self.config.mode == "manual_multiunit" and self.config.manual_units:
            planned_units = self._select_manual_units(start_pose=start_pose, end_pose=end_pose)
        elif self.config.mode == "manual_multiunit" and self.config.manual_mid_poses:
            return save_manual_pose_plan(self.config, out_prefix=out_prefix)
        elif self.config.search_method == "beam":
            planned_units = self._select_beam(start_pose=start_pose, end_pose=end_pose)
        else:
            planned_units = self._select_greedy(start_pose=start_pose, end_pose=end_pose)

        return save_planned_units(
            planned_units=planned_units,
            frames=self.config.mid_frames,
            out_prefix=out_prefix,
            config=self.config,
            rag_db=self.rag_db,
            start_pose_path=start_pose_path,
            end_pose_path=end_pose_path,
        )


def _score_parts_from_score(score: UnitScore) -> Dict[str, Any]:
    return {
        "phase": "v10_unified_planner",
        "unit_id": score.unit_id,
        "source_key": score.source_key,
        "source_index": int(score.source_index),
        "global_index": int(score.index),
        "emission_score": float(score.emission_score),
        "motion_energy_norm": float(score.score),
        "transition_score": float(score.transition_score),
        "path_score": float(score.path_score),
        "entry_cost": float(score.entry_cost),
        "exit_cost": float(score.exit_cost),
        "transition_cost": float(score.transition_cost),
        "upper_activity": float(score.upper_activity),
        "lower_activity": float(score.lower_activity),
        "root_speed": float(score.root_speed),
        "spatial_range": float(score.spatial_range),
        "turning": float(score.turning),
        "pose_diversity": float(score.pose_diversity),
        "contact_change": float(score.contact_change),
    }


def _keyframe_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    score_parts = item.get("score_parts", {}) or {}
    return {
        "frame": int(item.get("frame", 0)),
        "segment_id": int(item.get("rank", -1)),
        "phase": str(item.get("phase", "v10_unified_planner")),
        "source": item.get("unit_path", item.get("pose_path", "")),
        "score": float(score_parts.get("transition_score", score_parts.get("emission_score", 0.0))),
        "score_parts": score_parts,
    }


def _attach_legacy_keyframes(plan: Dict[str, Any]) -> Dict[str, Any]:
    items = plan.get("items", []) or []
    keyframes = [_keyframe_from_item(item) for item in items]
    plan["keyframes"] = keyframes
    plan["auto_keyframes"] = keyframes
    return plan


def save_planned_units(
    planned_units: Sequence[PlannedUnit],
    frames: Sequence[int],
    out_prefix: str,
    config: PlannerConfig,
    rag_db: Optional[str] = None,
    start_pose_path: str = "",
    end_pose_path: str = "",
) -> Dict[str, Any]:
    if len(planned_units) != len(frames):
        raise ValueError(f"planned_units count {len(planned_units)} != frames count {len(frames)}")

    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    mid_paths: List[str] = []
    unit_paths: List[str] = []
    unit_ids: List[str] = []
    items: List[Dict[str, Any]] = []

    for k, (planned, frame) in enumerate(zip(planned_units, frames), start=1):
        unit = as_unit_t151(planned.unit)
        score = planned.score
        meta = planned.meta

        mid_idx = int(round((len(unit) - 1) / 2))
        pose = unit[mid_idx].astype(np.float32)

        pose_path = out_prefix_path.parent / f"{out_prefix_path.name}_v10_mid{k}_f{int(frame):03d}.npy"
        unit_path = out_prefix_path.parent / f"{out_prefix_path.name}_v10_mid{k}_f{int(frame):03d}_unit.npy"

        np.save(pose_path, pose)
        np.save(unit_path, unit.astype(np.float32))

        mid_paths.append(str(pose_path))
        unit_paths.append(str(unit_path))
        unit_ids.append(str(score.unit_id))

        score_parts = _score_parts_from_score(score)
        items.append(
            {
                "rank": k,
                "frame": int(frame),
                "phase": "v10_unified_planner",
                "pose_path": str(pose_path),
                "unit_path": str(unit_path),
                "unit_id": str(score.unit_id),
                "requested_unit_id": str(meta.get("requested_unit_id", "")),
                "source_key": str(meta.get("source_key", score.source_key)),
                "source_index": int(meta.get("source_index", score.source_index)),
                "global_index": int(meta.get("global_index", score.index)),
                "score_parts": score_parts,
            }
        )

    plan: Dict[str, Any] = {
        "version": "v10_unified_planner_v3_env_isolation_fix",
        "mode": config.mode,
        "search_method": config.search_method,
        "mid_poses": mid_paths,
        "mid_pose_frames": [int(f) for f in frames],
        "unit_paths": unit_paths,
        "unit_ids": unit_ids,
        "items": items,
        "rag_db": str(rag_db or ""),
        "start_pose_path": str(start_pose_path or ""),
        "end_pose_path": str(end_pose_path or ""),
        "transition_aware": True,
        "v9_rag_summary_unit_paths": unit_paths,
        "planner_config": asdict(config),
        "conditions": {
            "use_rag_summary": bool(unit_paths),
            "rag_summary_mode": config.rag_summary_mode,
            "use_unit_prior": env_bool("EDGE_UNIT_SOFT_PRIOR", True),
            "unit_prior_strength": env_float("EDGE_UNIT_PRIOR_STRENGTH", 0.012),
            "mid_keyframe_strength": env_float("EDGE_V10_MID_STRENGTH", 0.035),
        },
    }
    plan = _attach_legacy_keyframes(plan)

    plan_path = out_prefix_path.parent / f"{out_prefix_path.name}_v10_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    plan["plan_path"] = str(plan_path)
    print(f"✅ V10 Unified plan saved: {plan_path}")
    print(
        "✅ V10 Unified planner selected units: "
        + ", ".join(f"{uid}@f{frame}" for uid, frame in zip(unit_ids, frames))
    )
    return plan


def save_manual_pose_plan(config: PlannerConfig, out_prefix: str) -> Dict[str, Any]:
    paths = config.manual_mid_poses
    if not paths:
        raise ValueError("manual_mid_poses is empty.")

    if len(paths) != len(config.mid_frames):
        raise ValueError(
            f"Manual pose count {len(paths)} must match frame count {len(config.mid_frames)}."
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
            "phase": "v10_manual_pose_legacy",
            "pose_path": path,
            "unit_path": "",
            "unit_id": "",
            "source_key": "manual_pose",
            "source_index": -1,
            "global_index": -1,
            "score_parts": {
                "phase": "v10_manual_pose_legacy",
                "warning": "No unit_paths: V9 rag_summary and DCT unit prior are unavailable in legacy manual-pose mode.",
                "emission_score": 0.0,
                "transition_score": 0.0,
                "entry_cost": 0.0,
                "exit_cost": 0.0,
            },
        }
        for i, (path, frame) in enumerate(zip(paths, config.mid_frames))
    ]

    plan: Dict[str, Any] = {
        "version": "v10_unified_planner_v3_env_isolation_fix",
        "mode": "manual_pose_legacy",
        "search_method": "manual_pose",
        "mid_poses": paths,
        "mid_pose_frames": [int(f) for f in config.mid_frames],
        "unit_paths": [],
        "unit_ids": [],
        "items": items,
        "transition_aware": False,
        "planner_config": asdict(config),
        "conditions": {
            "use_rag_summary": False,
            "use_unit_prior": False,
            "warning": "Prefer EDGE_V10_MANUAL_UNITS for fair Manual vs Auto comparison.",
        },
    }
    plan = _attach_legacy_keyframes(plan)

    plan_path = out_prefix_path.parent / f"{out_prefix_path.name}_v10_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    plan["plan_path"] = str(plan_path)
    print(f"✅ V10 legacy manual-pose plan saved: {plan_path}")
    print("⚠️ Legacy manual-pose mode has no unit_paths; prefer EDGE_V10_MANUAL_UNITS.")
    return plan


def plan_choreo_from_rag_db(
    rag_db: str,
    out_prefix: str,
    config: Optional[PlannerConfig] = None,
    start_pose_path: str = "",
    end_pose_path: str = "",
) -> Dict[str, Any]:
    config = config or build_config_from_env()
    planner = UnifiedChoreoPlanner(config=config, rag_db=rag_db)
    return planner.plan(out_prefix=out_prefix, start_pose_path=start_pose_path, end_pose_path=end_pose_path)


# Backward-compatible wrappers used by older scripts.
def plan_upperdance_from_rag_db(
    rag_db: str,
    out_prefix: str,
    num_frames: int = 150,
    count: int = 2,
    frames: Optional[Sequence[int]] = None,
    start_pose_path: str = "",
    end_pose_path: str = "",
) -> Dict[str, Any]:
    config = build_config_from_env(num_frames=num_frames)
    if frames is not None:
        config.mid_frames = [int(x) for x in frames]
    elif len(config.mid_frames) != int(count):
        config.mid_frames = parse_frames("", count=int(count), num_frames=num_frames)
    return plan_choreo_from_rag_db(
        rag_db=rag_db,
        out_prefix=out_prefix,
        config=config,
        start_pose_path=start_pose_path,
        end_pose_path=end_pose_path,
    )


def plan_manual_from_env(
    out_prefix: str,
    num_frames: int = 150,
) -> Optional[Dict[str, Any]]:
    config = build_config_from_env(num_frames=num_frames)
    if config.mode != "manual_multiunit" or (not config.manual_units and not config.manual_mid_poses):
        return None

    rag_db = os.environ.get("EDGE_V10_RAG_DB") or os.environ.get("RAG_DB", "")
    if config.manual_units and not rag_db:
        raise RuntimeError("EDGE_V10_MANUAL_UNITS requires EDGE_V10_RAG_DB or RAG_DB.")

    planner = UnifiedChoreoPlanner(config=config, rag_db=rag_db or None)
    return planner.plan(out_prefix=out_prefix)
