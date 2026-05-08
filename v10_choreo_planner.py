#!/usr/bin/env python3
"""Energy/Expressiveness-aware Unified V10 Choreo Planner for EDGE.

Drop-in replacement for v10_choreo_planner.py.

Key additions
-------------
1. Energy-aware / expressiveness-aware hard filtering:
   EDGE_UNIT_MIN_ENERGY
   EDGE_UNIT_MIN_EXPRESSIVENESS
   EDGE_UNIT_BAN_LOW_ENERGY
   EDGE_UNIT_LOW_ENERGY_THRESHOLD

2. Soft score bonuses:
   EDGE_UNIT_ENERGY_BONUS
   EDGE_UNIT_EXPRESSIVENESS_BONUS
   EDGE_V10_TEXT_SCORE_W

3. Optional stats cache:
   EDGE_RAG_STATS_CACHE=/path/to/index_stats.npz
   If unset, planner auto-searches <rag_db_stem>_stats.npz.
   If no cache exists, it computes needed stats in memory.

4. Full plan audit output:
   <out_prefix>_v10_plan.json
   <out_prefix>_v10_score_parts.json
   <out_prefix>_midXX.npy
   <out_prefix>_midXX_unit.npy

Compatibility
-------------
generate_v10_choreo.py imports:
    build_config_from_env, env_int, plan_choreo_from_rag_db
These names are preserved.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPR_DIM = 151
CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROOT_XZ_IDX = [ROOT_X_IDX, ROOT_Z_IDX]
ROT_START = 7
ROT_SLICE = slice(7, 151)
N_JOINTS = 24
ROT_DIM = 6
TORSO_JOINTS = [3, 6, 9]
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
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_csv(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def parse_frames(text: str, count: int, num_frames: int = 150) -> List[int]:
    if text:
        raw = parse_csv(text)
        if len(raw) != count:
            raise ValueError(f"Expected {count} frame values, got {len(raw)} from: {text}")
        frames: List[int] = []
        for item in raw:
            value = float(item)
            if 0.0 < value < 1.0:
                value = value * (num_frames - 1)
            frames.append(int(round(value)))
        return [max(1, min(num_frames - 2, f)) for f in frames]
    return [int(round((i + 1) * (num_frames - 1) / (count + 1))) for i in range(count)]


def default_dual_mid_frames(num_frames: int = 150) -> List[int]:
    explicit = os.environ.get("EDGE_V10_MID_FRAMES", "")
    if explicit:
        raw = parse_csv(explicit)
        return parse_frames(explicit, len(raw), num_frames=num_frames)
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
    mode = str(mode or "dual_auto_mid").strip().lower()
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
    unit_energy: float = 0.0
    unit_energy_norm: float = 0.0
    expressiveness_score: float = 0.0
    upper_activity: float = 0.0
    upper_activity_norm: float = 0.0
    lower_activity: float = 0.0
    lower_activity_norm: float = 0.0
    root_speed: float = 0.0
    root_speed_norm: float = 0.0
    spatial_range: float = 0.0
    spatial_range_norm: float = 0.0
    turning: float = 0.0
    turning_norm: float = 0.0
    pose_diversity: float = 0.0
    pose_diversity_norm: float = 0.0
    contact_change: float = 0.0
    contact_stability: float = 1.0
    text_score: float = 0.0
    source_key: str = ""
    source_index: int = -1
    unit_id: str = ""
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    transition_cost: float = 0.0
    transition_score: float = 0.0
    path_score: float = 0.0
    emission_score: float = 0.0
    score_parts: Dict[str, float] = field(default_factory=dict)


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
        "text": env_float("EDGE_V10_TEXT_SCORE_W", env_float("EDGE_TEXT_SCORE_WEIGHT", 0.15)),
        "energy_bonus": env_float("EDGE_UNIT_ENERGY_BONUS", 0.0),
        "expressiveness_bonus": env_float("EDGE_UNIT_EXPRESSIVENESS_BONUS", 0.0),
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

    manual_units = parse_csv(os.environ.get("EDGE_V10_MANUAL_UNITS", "")) if mode == "manual_multiunit" else []
    manual_mid_poses = parse_csv(os.environ.get("EDGE_V10_MANUAL_MID_POSES", "")) if mode == "manual_multiunit" else []
    if mode != "manual_multiunit" and (os.environ.get("EDGE_V10_MANUAL_UNITS") or os.environ.get("EDGE_V10_MANUAL_MID_POSES")):
        print(f"ℹ️ Ignoring manual unit/pose env because EDGE_V10_MODE={mode}.")

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
        raise ValueError(f"Expected [T,151] or [151,T], got {arr.shape}")
    if arr.shape[-1] == REPR_DIM:
        return arr
    if arr.shape[0] == REPR_DIM:
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
        arr = d.get("motion", d.get("pose", arr))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[-1] == REPR_DIM:
            return arr[0].astype(np.float32)
        if arr.shape[0] == REPR_DIM:
            return arr[:, 0].astype(np.float32)
    arr = arr.reshape(-1)
    if arr.shape[0] != REPR_DIM:
        raise ValueError(f"Expected 151D pose: {path}, got {arr.shape}")
    return arr.astype(np.float32)


def _joint_rot_indices(joints: Iterable[int]) -> np.ndarray:
    idx = []
    for j in joints:
        j = int(j)
        idx.extend(range(ROT_START + ROT_DIM * j, ROT_START + ROT_DIM * j + ROT_DIM))
    return np.asarray([i for i in idx if 0 <= i < REPR_DIM], dtype=np.int64)


UPPER_ROT_INDEX = _joint_rot_indices(list(TORSO_JOINTS) + list(UPPER_JOINTS))
LOWER_ROT_INDEX = _joint_rot_indices(LOWER_JOINTS)


def robust_norm(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values, 0.0, 1.0
    lo, hi = np.percentile(values, [10, 90])
    if float(hi - lo) <= 1e-8:
        lo, hi = float(values.min()), float(values.max() + 1e-6)
    return np.clip((values - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0).astype(np.float32), float(lo), float(hi)


def compute_unit_stats(unit: np.ndarray) -> Dict[str, float]:
    unit = as_unit_t151(unit)
    if len(unit) <= 1:
        return dict(unit_energy=0.0, upper_activity=0.0, lower_activity=0.0, root_speed=0.0, spatial_range=0.0, turning=0.0, pose_diversity=0.0, contact_change=0.0, contact_stability=1.0)
    diff = unit[1:] - unit[:-1]
    rot_diff = diff[:, ROT_SLICE]
    root_xz = unit[:, ROOT_XZ_IDX]
    root_vel = root_xz[1:] - root_xz[:-1]
    rot = unit[:, ROT_SLICE]
    contact = (unit[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_change = float(np.mean(np.abs(contact[1:] - contact[:-1]))) if len(contact) > 1 else 0.0
    if len(root_vel) >= 2:
        v1, v2 = root_vel[:-1], root_vel[1:]
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        cos = np.sum(v1 * v2, axis=-1) / np.clip(n1 * n2, 1e-8, None)
        turning = float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))
    else:
        turning = 0.0
    return dict(
        unit_energy=float(np.mean(np.linalg.norm(rot_diff, axis=-1))),
        upper_activity=float(np.sqrt(np.mean(diff[:, UPPER_ROT_INDEX] ** 2))) if UPPER_ROT_INDEX.size else 0.0,
        lower_activity=float(np.sqrt(np.mean(diff[:, LOWER_ROT_INDEX] ** 2))) if LOWER_ROT_INDEX.size else 0.0,
        root_speed=float(np.mean(np.linalg.norm(root_vel, axis=-1))) if len(root_vel) else 0.0,
        spatial_range=float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0))),
        turning=float(max(0.0, turning)),
        pose_diversity=float(np.mean(np.linalg.norm(rot - rot.mean(axis=0, keepdims=True), axis=-1))),
        contact_change=float(np.clip(contact_change, 0.0, 1.0)),
        contact_stability=float(np.clip(1.0 - contact_change, 0.0, 1.0)),
    )


def pose_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape[0] != REPR_DIM or b.shape[0] != REPR_DIM:
        return 0.0
    rot = float(np.sqrt(np.mean((a[ROT_SLICE] - b[ROT_SLICE]) ** 2)))
    root_y = float(abs(a[ROOT_Y_IDX] - b[ROOT_Y_IDX]))
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
    return float(1.0 - np.clip(np.dot(va, vb) / (na * nb + 1e-8), -1.0, 1.0))


def transition_cost(unit_a: np.ndarray, unit_b: np.ndarray, weights: Dict[str, float]) -> float:
    pose = pose_distance(as_unit_t151(unit_a)[-1], as_unit_t151(unit_b)[0])
    contact = contact_phase_distance(unit_a, unit_b)
    root = root_direction_distance(unit_a, unit_b)
    return float(weights.get("pose", 0.35)) * pose + float(weights.get("contact", 0.10)) * contact + float(weights.get("root", 0.05)) * root


def _safe_l2_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _encode_query_text(query: str, dim_hint: int) -> Optional[np.ndarray]:
    if not query:
        return None
    try:
        from text_context_rag_utils import encode_texts
        emb = encode_texts([query], fallback_dim=dim_hint)[0]
        return np.asarray(emb, dtype=np.float32)
    except Exception as exc:
        if env_bool("EDGE_TEXT_SCORE_REQUIRED", False):
            raise RuntimeError(f"Text scoring requested but text encoder failed: {exc}") from exc
        print(f"⚠️ Text score disabled: failed to encode query ({exc})")
        return None


def default_query_for_mode(mode: str) -> str:
    custom = env_str("EDGE_TEXT_QUERY", "")
    if custom:
        return custom
    mode = str(mode or "").lower()
    if "upper" in mode:
        return "敦煌舞，飞天风格，上肢大幅舒展，手臂展开，身体线条清晰，中高能量，优雅转身"
    if "manual" in mode:
        return "敦煌舞，人工精选动作单元，上肢舒展，姿态清晰，重心稳定"
    return "敦煌舞，飞天风格，高能量，上肢大幅舒展，流动转身，空间展开，短段落编舞"


def _candidate_arrays_from_npz(npz: np.lib.npyio.NpzFile) -> List[Tuple[str, np.ndarray]]:
    preferred = ["unit_motions", "unit_motions_physical", "motions", "motion_units", "clips", "units", "x"]
    out: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for key in preferred + list(npz.files):
        if key in seen or key not in npz.files:
            continue
        seen.add(key)
        try:
            arr = np.asarray(npz[key])
        except Exception:
            continue
        if arr.dtype == object:
            continue
        if arr.ndim == 3 and (arr.shape[-1] == REPR_DIM or arr.shape[1] == REPR_DIM):
            out.append((key, arr))
        elif arr.ndim == 2 and (arr.shape[-1] == REPR_DIM or arr.shape[0] == REPR_DIM):
            out.append((key, arr[None]))
    return out


def _stats_cache_candidates(rag_db: str) -> List[Path]:
    p = Path(rag_db)
    out = []
    if os.environ.get("EDGE_RAG_STATS_CACHE"):
        out.append(Path(os.environ["EDGE_RAG_STATS_CACHE"]))
    out.append(p.with_name(p.stem + "_stats.npz"))
    if p.name.endswith(".npz"):
        out.append(p.with_suffix(".stats.npz"))
    return out


class RAGUnitDB:
    def __init__(self, rag_db: str, max_units: Optional[int] = None):
        self.rag_db = str(rag_db)
        self.units, self.meta, self.raw_npz = self._load_units(self.rag_db, max_units=max_units)
        self.stats_npz = self._load_stats_cache()
        self.stats = self._load_or_compute_stats()
        self.text_embeddings = self._load_text_embeddings()
        self.text_scores = self._compute_text_scores()
        self._id_to_index = self._build_id_map()

    @staticmethod
    def _coerce_unit_array(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == REPR_DIM:
            return arr
        if arr.ndim == 3 and arr.shape[1] == REPR_DIM:
            return np.transpose(arr, (0, 2, 1)).astype(np.float32)
        raise ValueError(f"Expected [N,T,151] or [N,151,T], got {arr.shape}")

    def _load_units(self, rag_db: str, max_units: Optional[int]) -> Tuple[List[np.ndarray], List[Dict[str, Any]], np.lib.npyio.NpzFile]:
        path = Path(rag_db)
        if not path.exists():
            raise FileNotFoundError(f"RAG DB not found: {rag_db}")
        npz = np.load(str(path), allow_pickle=True)
        arrays = _candidate_arrays_from_npz(npz)
        if not arrays:
            raise ValueError(f"No [N,T,151] unit arrays found in {rag_db}. keys={npz.files}")
        key, arr = arrays[0]
        units_arr = self._coerce_unit_array(arr)
        if max_units is not None and max_units > 0:
            units_arr = units_arr[: int(max_units)]
        units = [units_arr[i].astype(np.float32) for i in range(units_arr.shape[0])]

        source = np.asarray(npz["source"]) if "source" in npz.files and len(npz["source"]) >= len(units) else None
        unit_start = np.asarray(npz["unit_start"]) if "unit_start" in npz.files and len(npz["unit_start"]) >= len(units) else None
        unit_center = np.asarray(npz["unit_center"]) if "unit_center" in npz.files and len(npz["unit_center"]) >= len(units) else None
        unit_end = np.asarray(npz["unit_end"]) if "unit_end" in npz.files and len(npz["unit_end"]) >= len(units) else None
        meta = []
        for i in range(len(units)):
            m = {
                "source_key": key,
                "source_index": int(i),
                "global_index": int(i),
                "unit_id": f"{key}:{i}",
                "alt_unit_id": f"unit_{i:04d}",
            }
            if source is not None:
                m["source"] = str(source[i])
            if unit_start is not None:
                m["unit_start"] = int(unit_start[i])
            if unit_center is not None:
                m["unit_center"] = int(unit_center[i])
            if unit_end is not None:
                m["unit_end"] = int(unit_end[i])
            meta.append(m)
        return units, meta, npz

    def _load_stats_cache(self) -> Optional[np.lib.npyio.NpzFile]:
        for path in _stats_cache_candidates(self.rag_db):
            if path and path.exists():
                try:
                    print(f"✅ V10 planner using stats cache: {path}")
                    return np.load(str(path), allow_pickle=True)
                except Exception as exc:
                    print(f"⚠️ Failed to load stats cache {path}: {exc}")
        return None

    def _field(self, key: str) -> Optional[np.ndarray]:
        n = len(self.units)
        for src in [self.stats_npz, self.raw_npz]:
            if src is not None and key in src.files:
                arr = np.asarray(src[key])
                if arr.shape[0] >= n:
                    return arr[:n]
        return None

    def _load_or_compute_stats(self) -> Dict[str, np.ndarray]:
        n = len(self.units)
        raw_keys = ["unit_energy", "motion_energy", "upper_activity", "lower_activity", "root_speed", "spatial_range", "turning", "pose_diversity", "contact_change", "contact_stability"]
        stats: Dict[str, np.ndarray] = {}
        for key in raw_keys:
            arr = self._field(key)
            if arr is not None:
                stats[key] = np.asarray(arr, dtype=np.float32)
        if "unit_energy" not in stats and "motion_energy" in stats:
            stats["unit_energy"] = stats["motion_energy"]
        if any(k not in stats for k in ["unit_energy", "upper_activity", "root_speed", "contact_change", "pose_diversity"]):
            rows = [compute_unit_stats(u) for u in self.units]
            for k in raw_keys:
                if k not in stats:
                    stats[k] = np.asarray([r.get(k, 0.0) for r in rows], dtype=np.float32)
        if "contact_stability" not in stats:
            stats["contact_stability"] = np.clip(1.0 - stats.get("contact_change", np.zeros(n, dtype=np.float32)), 0.0, 1.0)
        for key in ["unit_energy", "upper_activity", "lower_activity", "root_speed", "spatial_range", "turning", "pose_diversity", "contact_change"]:
            norm_key = f"{key}_norm"
            arr = self._field(norm_key)
            if arr is not None:
                stats[norm_key] = np.asarray(arr, dtype=np.float32)
            else:
                stats[norm_key] = robust_norm(stats[key])[0]
        expr = self._field("expressiveness_score")
        if expr is not None:
            stats["expressiveness_score"] = np.asarray(expr, dtype=np.float32)
        else:
            stats["expressiveness_score"] = np.clip(
                0.30 * stats["unit_energy_norm"]
                + 0.30 * stats["upper_activity_norm"]
                + 0.20 * stats["spatial_range_norm"]
                + 0.15 * stats["turning_norm"]
                + 0.05 * stats["root_speed_norm"],
                0.0,
                1.0,
            ).astype(np.float32) * np.clip(0.50 + 0.50 * stats["contact_stability"], 0.0, 1.0)
        for key, arr in list(stats.items()):
            stats[key] = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)[:n]
        return stats

    def _load_text_embeddings(self) -> Optional[np.ndarray]:
        n = len(self.units)
        for key in ["motion_text_embedding", "motion_embedding", "text_embedding"]:
            arr = self._field(key)
            if arr is not None and arr.ndim == 2 and arr.shape[0] >= n:
                return np.asarray(arr[:n], dtype=np.float32)
        return None

    def _compute_text_scores(self) -> np.ndarray:
        n = len(self.units)
        if self.text_embeddings is None or self.text_embeddings.ndim != 2:
            return np.zeros((n,), dtype=np.float32)
        query = default_query_for_mode(env_str("EDGE_V10_MODE", "auto_multiunit"))
        q = _encode_query_text(query, dim_hint=self.text_embeddings.shape[-1])
        if q is None:
            return np.zeros((n,), dtype=np.float32)
        d = min(q.shape[-1], self.text_embeddings.shape[-1])
        scores = np.dot(_safe_l2_norm(self.text_embeddings[:, :d]), _safe_l2_norm(q[:d].reshape(1, -1))[0])
        # map cosine [-1,1] -> [0,1]
        return np.clip((scores.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)

    def _build_id_map(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
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
                    out[key] = i
        return out

    def resolve_unit_ids(self, unit_ids: Sequence[str]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        resolved_units: List[np.ndarray] = []
        resolved_meta: List[Dict[str, Any]] = []
        for raw in unit_ids:
            token = str(raw).strip()
            if not token:
                continue
            idx = self._id_to_index.get(token)
            if idx is None:
                m = re.fullmatch(r"unit_0*(\d+)", token)
                if m:
                    idx = self._id_to_index.get(str(int(m.group(1))))
            if idx is None and ":" in token:
                source_key, source_idx_text = token.rsplit(":", 1)
                m = re.fullmatch(r"(?:unit_)?0*(\d+)", source_idx_text.strip())
                if m:
                    idx = self._id_to_index.get(f"{source_key}:{int(m.group(1))}")
            if idx is None:
                examples = [str(m.get("unit_id", "")) for m in self.meta[:5]]
                raise KeyError(f"Cannot resolve manual unit id {token!r}. Examples: {examples}")
            meta = dict(self.meta[idx])
            meta["requested_unit_id"] = token
            resolved_units.append(self.units[idx])
            resolved_meta.append(meta)
        return resolved_units, resolved_meta

    def score_unit(self, index: int, weights: Dict[str, float]) -> UnitScore:
        i = int(index)
        s = self.stats
        text_score = float(self.text_scores[i]) if self.text_scores is not None else 0.0
        parts = {
            "text": float(weights.get("text", 0.15)) * text_score,
            "upper": float(weights.get("upper", 0.45)) * float(s["upper_activity_norm"][i]),
            "turn": float(weights.get("turn", 0.15)) * float(s["turning_norm"][i]),
            "diversity": float(weights.get("diversity", 0.20)) * float(s["pose_diversity_norm"][i]),
            "energy_bonus": float(weights.get("energy_bonus", 0.0)) * float(s["unit_energy_norm"][i]),
            "expressiveness_bonus": float(weights.get("expressiveness_bonus", 0.0)) * float(s["expressiveness_score"][i]),
            "contact_penalty": -float(weights.get("contact", 0.08)) * float(s["contact_change_norm"][i]),
            "root_penalty": -float(weights.get("root_penalty", 0.12)) * float(s["root_speed_norm"][i]),
            "lower_penalty": -float(weights.get("lower_penalty", 0.05)) * max(0.0, float(s["lower_activity_norm"][i]) - float(s["upper_activity_norm"][i])),
        }
        score = float(sum(parts.values()))
        m = self.meta[i]
        unit_id = f"{m.get('source_key', '')}:{int(m.get('source_index', i))}" if m.get("source_key") else f"unit_{i}"
        return UnitScore(
            index=i,
            score=score,
            emission_score=score,
            unit_energy=float(s["unit_energy"][i]),
            unit_energy_norm=float(s["unit_energy_norm"][i]),
            expressiveness_score=float(s["expressiveness_score"][i]),
            upper_activity=float(s["upper_activity"][i]),
            upper_activity_norm=float(s["upper_activity_norm"][i]),
            lower_activity=float(s["lower_activity"][i]),
            lower_activity_norm=float(s["lower_activity_norm"][i]),
            root_speed=float(s["root_speed"][i]),
            root_speed_norm=float(s["root_speed_norm"][i]),
            spatial_range=float(s["spatial_range"][i]),
            spatial_range_norm=float(s["spatial_range_norm"][i]),
            turning=float(s["turning"][i]),
            turning_norm=float(s["turning_norm"][i]),
            pose_diversity=float(s["pose_diversity"][i]),
            pose_diversity_norm=float(s["pose_diversity_norm"][i]),
            contact_change=float(s["contact_change"][i]),
            contact_stability=float(s["contact_stability"][i]),
            text_score=text_score,
            source_key=str(m.get("source_key", "")),
            source_index=int(m.get("source_index", i)),
            unit_id=unit_id,
            score_parts=parts,
        )


class UnifiedChoreoPlanner:
    def __init__(self, config: PlannerConfig, rag_db: Optional[str] = None):
        self.config = config
        self.rag_db = rag_db
        self.db: Optional[RAGUnitDB] = None
        if rag_db:
            # Default now favors full DB scanning. Set EDGE_V10_MAX_RAG_UNITS for smoke tests.
            max_units = env_int("EDGE_V10_MAX_RAG_UNITS", 1000000)
            self.db = RAGUnitDB(rag_db, max_units=max_units)

    def _passes_hard_filters(self, score: UnitScore) -> bool:
        min_energy = env_float("EDGE_UNIT_MIN_ENERGY", 0.0)
        min_expr = env_float("EDGE_UNIT_MIN_EXPRESSIVENESS", 0.0)
        ban_low = env_bool("EDGE_UNIT_BAN_LOW_ENERGY", False)
        low_thr = env_float("EDGE_UNIT_LOW_ENERGY_THRESHOLD", 0.20)
        energy_value = score.unit_energy_norm if env_bool("EDGE_UNIT_FILTER_USE_NORM", True) else score.unit_energy
        expr_value = score.expressiveness_score
        if energy_value < min_energy:
            return False
        if expr_value < min_expr:
            return False
        if ban_low and energy_value < low_thr:
            return False
        if score.upper_activity < self.config.min_upper_activity:
            return False
        if score.root_speed > self.config.max_root_speed:
            return False
        return True

    def _score_all(self) -> List[UnitScore]:
        if self.db is None:
            raise RuntimeError("RAG DB is required for auto/manual-unit planning.")
        scores = [self.db.score_unit(i, self.config.score_weights) for i in range(len(self.db.units))]
        filtered = [s for s in scores if self._passes_hard_filters(s)]
        if not filtered:
            print("⚠️ Energy/expressiveness filters removed all candidates; falling back to unfiltered scores.")
            filtered = scores
        filtered = sorted(filtered, key=lambda x: x.score, reverse=True)
        return filtered[: max(self.config.candidate_cap, self.config.top_k)]

    def _annotate_path(self, indices: Sequence[int], base_scores: Dict[int, UnitScore], start_pose: Optional[np.ndarray], end_pose: Optional[np.ndarray], path_score: float = 0.0) -> List[PlannedUnit]:
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
                s.transition_cost += float(self.config.transition_weights.get("end", 0.30)) * s.exit_cost
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

    def _select_manual_units(self, start_pose: Optional[np.ndarray], end_pose: Optional[np.ndarray]) -> List[PlannedUnit]:
        if self.db is None:
            raise RuntimeError("RAG DB is required when EDGE_V10_MANUAL_UNITS is set.")
        units, meta = self.db.resolve_unit_ids(self.config.manual_units)
        base_scores: Dict[int, UnitScore] = {}
        indices: List[int] = []
        for unit, m in zip(units, meta):
            idx = int(m["global_index"])
            indices.append(idx)
            base_scores[idx] = self.db.score_unit(idx, self.config.score_weights)
        return self._annotate_path(indices, base_scores, start_pose, end_pose, path_score=0.0)

    def _select_greedy(self, start_pose: Optional[np.ndarray], end_pose: Optional[np.ndarray]) -> List[PlannedUnit]:
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

    def _select_beam(self, start_pose: Optional[np.ndarray], end_pose: Optional[np.ndarray]) -> List[PlannedUnit]:
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
        return self._annotate_path(best_path, base_scores, start_pose, end_pose, path_score=best_score)

    def plan(self, start_pose: Optional[np.ndarray] = None, end_pose: Optional[np.ndarray] = None) -> List[PlannedUnit]:
        if self.config.manual_units:
            planned = self._select_manual_units(start_pose, end_pose)
        elif self.config.manual_mid_poses:
            return []
        elif self.config.search_method == "beam":
            planned = self._select_beam(start_pose, end_pose)
        else:
            planned = self._select_greedy(start_pose, end_pose)
        if len(planned) != self.config.count:
            raise ValueError(f"planned_units count {len(planned)} != frames count {self.config.count}")
        return planned


def _center_pose(unit: np.ndarray) -> np.ndarray:
    unit = as_unit_t151(unit)
    return unit[len(unit) // 2].astype(np.float32)


def _write_plan_files(planned: List[PlannedUnit], out_prefix: str, frames: Sequence[int]) -> Dict[str, Any]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mid_poses: List[str] = []
    unit_paths: List[str] = []
    rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    for rank, (p, frame) in enumerate(zip(planned, frames), start=1):
        pose_path = prefix.with_name(prefix.name + f"_mid{rank:02d}_f{int(frame)}.npy")
        unit_path = prefix.with_name(prefix.name + f"_mid{rank:02d}_f{int(frame)}_unit.npy")
        np.save(pose_path, _center_pose(p.unit).astype(np.float32))
        np.save(unit_path, as_unit_t151(p.unit).astype(np.float32))
        mid_poses.append(str(pose_path))
        unit_paths.append(str(unit_path))
        sdict = asdict(p.score)
        row = {
            "rank": rank,
            "frame": int(frame),
            "pose_path": str(pose_path),
            "unit_path": str(unit_path),
            "unit_index": int(p.score.index),
            "unit_id": p.score.unit_id,
            "source": p.meta.get("source", ""),
            "source_key": p.meta.get("source_key", ""),
            "source_index": int(p.meta.get("source_index", p.score.index)),
            "unit_start": p.meta.get("unit_start", None),
            "unit_center": p.meta.get("unit_center", None),
            "unit_end": p.meta.get("unit_end", None),
            "score": float(p.score.score),
            "transition_cost": float(p.score.transition_cost),
            "transition_score": float(p.score.transition_score),
            "path_score": float(p.score.path_score),
            "unit_energy_norm": float(p.score.unit_energy_norm),
            "expressiveness_score": float(p.score.expressiveness_score),
            "upper_activity_norm": float(p.score.upper_activity_norm),
            "text_score": float(p.score.text_score),
            "contact_stability": float(p.score.contact_stability),
            "score_parts": p.score.score_parts,
        }
        rows.append(row)
        score_rows.append({k: v for k, v in sdict.items() if k != "score_parts"} | {"score_parts": p.score.score_parts})
    plan_path = prefix.with_name(prefix.name + "_v10_plan.json")
    score_parts_path = prefix.with_name(prefix.name + "_v10_score_parts.json")
    plan = {
        "mid_poses": mid_poses,
        "mid_pose_frames": [int(x) for x in frames],
        "unit_paths": unit_paths,
        "records": rows,
        "plan_path": str(plan_path),
        "score_parts_path": str(score_parts_path),
        "env": {
            "EDGE_UNIT_MIN_ENERGY": os.environ.get("EDGE_UNIT_MIN_ENERGY", ""),
            "EDGE_UNIT_MIN_EXPRESSIVENESS": os.environ.get("EDGE_UNIT_MIN_EXPRESSIVENESS", ""),
            "EDGE_UNIT_BAN_LOW_ENERGY": os.environ.get("EDGE_UNIT_BAN_LOW_ENERGY", ""),
            "EDGE_UNIT_ENERGY_BONUS": os.environ.get("EDGE_UNIT_ENERGY_BONUS", ""),
            "EDGE_UNIT_EXPRESSIVENESS_BONUS": os.environ.get("EDGE_UNIT_EXPRESSIVENESS_BONUS", ""),
            "EDGE_V10_TEXT_SCORE_W": os.environ.get("EDGE_V10_TEXT_SCORE_W", ""),
            "EDGE_RAG_STATS_CACHE": os.environ.get("EDGE_RAG_STATS_CACHE", ""),
            "EDGE_V10_MAX_RAG_UNITS": os.environ.get("EDGE_V10_MAX_RAG_UNITS", ""),
        },
    }
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    score_parts_path.write_text(json.dumps(score_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def plan_choreo_from_rag_db(
    rag_db: str,
    out_prefix: str,
    config: PlannerConfig,
    start_pose_path: str = "",
    end_pose_path: str = "",
) -> Dict[str, Any]:
    if config.manual_mid_poses and not config.manual_units:
        plan = {
            "mid_poses": list(config.manual_mid_poses),
            "mid_pose_frames": list(config.mid_frames),
            "unit_paths": [],
            "records": [],
        }
        prefix = Path(out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        plan_path = prefix.with_name(prefix.name + "_v10_plan.json")
        plan["plan_path"] = str(plan_path)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return plan

    start_pose = load_151_pose(start_pose_path)
    end_pose = load_151_pose(end_pose_path)
    planner = UnifiedChoreoPlanner(config=config, rag_db=rag_db)
    planned = planner.plan(start_pose=start_pose, end_pose=end_pose)
    plan = _write_plan_files(planned, out_prefix=out_prefix, frames=config.mid_frames)
    print("✅ V10 Energy/Expressiveness-aware planner selected units:")
    for rec in plan["records"]:
        print(
            f"  rank={rec['rank']} frame={rec['frame']} idx={rec['unit_index']} "
            f"score={rec['score']:.4f} energy={rec['unit_energy_norm']:.3f} "
            f"expr={rec['expressiveness_score']:.3f} upper={rec['upper_activity_norm']:.3f} "
            f"text={rec['text_score']:.3f} trans={rec['transition_cost']:.4f}"
        )
    print(f"  plan={plan['plan_path']}")
    print(f"  score_parts={plan['score_parts_path']}")
    return plan


if __name__ == "__main__":
    # Lightweight CLI sanity check.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_db", required=True)
    parser.add_argument("--out_prefix", default="output/v10_eval/planner_debug")
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--start_pose", default="")
    parser.add_argument("--end_pose", default="")
    args = parser.parse_args()
    cfg = build_config_from_env(num_frames=args.num_frames)
    plan_choreo_from_rag_db(args.rag_db, args.out_prefix, cfg, args.start_pose, args.end_pose)

# ---------------------------------------------------------------------------
# Compatibility helper for text_bridge_planner_patch.py
# ---------------------------------------------------------------------------
def _score_parts_from_score(score):
    # Return a JSON-serializable score dictionary for a UnitScore-like object.
    # Older text_bridge_planner_patch.py expects this private helper to exist.
    # This helper is permissive and preserves dynamically added semantic fields.
    try:
        from dataclasses import asdict, is_dataclass
    except Exception:
        asdict = None
        is_dataclass = lambda _: False  # type: ignore

    out = {}
    if asdict is not None and is_dataclass(score):
        try:
            out.update(asdict(score))
        except Exception:
            pass

    keys = [
        "index", "unit_id", "source_key", "source_index",
        "score", "emission_score", "path_score", "transition_score",
        "transition_cost", "entry_cost", "exit_cost",
        "upper_activity", "lower_activity", "root_speed", "spatial_range",
        "turning", "pose_diversity", "contact_change", "contact_stability",
        "unit_energy", "motion_energy", "expressiveness_score", "text_score",
        "semantic_score", "semantic_score_norm", "semantic_query",
        "motion_text", "text_bridge_weight", "text_bridge_mode", "original_score",
    ]

    for key in keys:
        if not hasattr(score, key):
            continue
        value = getattr(score, key)
        try:
            import numpy as _np
            if isinstance(value, (_np.floating, _np.integer)):
                value = value.item()
            elif isinstance(value, _np.ndarray):
                value = value.tolist()
        except Exception:
            pass
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
        except Exception:
            pass
        out[key] = value

    if not out:
        for key in dir(score):
            if key.startswith("_"):
                continue
            try:
                value = getattr(score, key)
            except Exception:
                continue
            if callable(value):
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                out[key] = value

    return out

