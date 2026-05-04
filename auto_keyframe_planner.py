"""Tension / expressiveness-aware ChoreoRAG auto middle-keyframe planner for EDGE.

Drop-in replacement for auto_keyframe_planner.py.

Design goals:
- Default behavior remains close to the existing ChoreoRAG planner.
- All new behavior is activated by environment variables.
- Stage 1: expressiveness-aware retrieval.
- Stage 2: music tension / phase-aware dynamic weight modulation.
- Stage 3: homogeneity penalty for long sequences.
- Stage 4 support: save selected 45-frame unit priors for generate_controlled.py.

Key environment variables:
  # Stage 1: expressiveness filtering / reward
  EDGE_UNIT_MIN_EXPRESSIVENESS=-1       # >=0 filters candidates below this score
  EDGE_UNIT_EXPRESSIVENESS_BONUS=0.0    # subtract bonus * expressiveness from cost
  EDGE_UNIT_MIN_ENERGY=-1               # optional hard energy floor
  EDGE_UNIT_ENERGY_BONUS=0.0            # optional energy bonus

  # Stage 2: phase-aware planner
  EDGE_TENSION_AWARE_PLANNER=0          # enable attack/flow/pose dynamic weights

  # Stage 3: homogeneity penalty
  EDGE_UNIT_HOMOGENEITY_WEIGHT=0.0
  EDGE_UNIT_HOMOGENEITY_MIN_FRAMES=240

  # Existing / compatible vars
  EDGE_CHOREO_PLAN_JSON=/path/to/plan.json
  EDGE_CHOREO_STYLE_HINT=敦煌舞，飞天感，上肢舒展，重心稳定
  EDGE_TEXT_BRIDGE_MODEL=BAAI/bge-small-zh-v1.5
  EDGE_TEXT_BRIDGE_DEVICE=cpu
  EDGE_TEXT_BRIDGE_WEIGHT=0.50
  EDGE_UNIT_ENTRY_WEIGHT=0.60
  EDGE_UNIT_EXIT_WEIGHT=0.60
  EDGE_UNIT_CONTACT_PHASE_WEIGHT=0.85
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from dataset.quaternion import ax_to_6v
    from dataset.preprocess import vectorize_many, Normalizer
    from vis import SMPLSkeleton
except Exception:  # pragma: no cover
    ax_to_6v = None
    vectorize_many = None
    Normalizer = None
    SMPLSkeleton = None

try:
    from model.text_bridge_encoder import TextBridgeEncoder
except Exception:  # pragma: no cover
    try:
        from text_bridge_encoder import TextBridgeEncoder  # type: ignore
    except Exception:
        TextBridgeEncoder = None  # type: ignore

try:
    from music_choreo_planner import load_or_build_choreo_plan
except Exception:  # pragma: no cover
    load_or_build_choreo_plan = None


ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
CONTACT_SLICE = slice(0, 4)
ROT_SLICE = slice(7, 151)
POSE_FEATURE_INDEX = np.r_[ROOT_Y_IDX, np.arange(7, 151)]


@dataclass
class AutoKeyframe:
    frame: int
    pose: np.ndarray
    score: float
    source: str
    source_frame: int
    score_parts: Dict[str, float]
    unit_motion: Optional[np.ndarray] = None
    unit_start: int = -1
    unit_center: int = -1
    unit_end: int = -1
    motion_text: str = ""
    segment_id: int = -1
    segment_prompt: str = ""


@dataclass
class AutoKeyframePlan:
    keyframes: List[AutoKeyframe]
    frame_candidates: List[int]
    meta: Dict[str, object]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_motion_if_needed(motion: np.ndarray, normalizer, pose_space: str) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if pose_space == "normalized":
        return motion.astype(np.float32)
    if pose_space == "physical":
        if normalizer is None:
            raise ValueError("pose_space='physical' requires a checkpoint normalizer")
        mt = torch.from_numpy(motion).float()
        if mt.ndim == 1:
            return _to_numpy(normalizer.normalize(mt[None, None]))[0, 0].astype(np.float32)
        if mt.ndim == 2:
            return _to_numpy(normalizer.normalize(mt[None]))[0].astype(np.float32)
        if mt.ndim == 3:
            return _to_numpy(normalizer.normalize(mt)).astype(np.float32)
    raise ValueError(f"Unsupported pose_space: {pose_space}")


def convert_loaded_motion(motion: np.ndarray, normalizer, stored_pose_space: str, requested_pose_space: str) -> np.ndarray:
    stored_pose_space = str(stored_pose_space or requested_pose_space)
    requested_pose_space = str(requested_pose_space or "normalized")
    motion = np.asarray(motion, dtype=np.float32)
    if stored_pose_space == requested_pose_space:
        return motion.astype(np.float32)
    if stored_pose_space == "physical" and requested_pose_space == "normalized":
        return normalize_motion_if_needed(motion, normalizer, "physical")
    if stored_pose_space == "normalized" and requested_pose_space == "physical":
        if normalizer is None:
            raise ValueError("Cannot unnormalize without normalizer")
        mt = torch.from_numpy(motion).float()
        if mt.ndim == 1:
            return _to_numpy(normalizer.unnormalize(mt[None, None]))[0, 0].astype(np.float32)
        if mt.ndim == 2:
            return _to_numpy(normalizer.unnormalize(mt[None]))[0].astype(np.float32)
        if mt.ndim == 3:
            return _to_numpy(normalizer.unnormalize(mt)).astype(np.float32)
    raise ValueError(f"Unsupported pose-space conversion {stored_pose_space}->{requested_pose_space}")


def _pkl_to_motion_151(path: Path) -> Optional[np.ndarray]:
    if SMPLSkeleton is None or ax_to_6v is None or vectorize_many is None:
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "pos" not in data or "q" not in data:
        return None
    pos = torch.tensor(data["pos"], dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(data["q"], dtype=torch.float32).unsqueeze(0)
    q = q.reshape(q.shape[0], q.shape[1], -1, 3)
    smpl = SMPLSkeleton()
    with torch.no_grad():
        joints = smpl.forward(q, pos)
        feet = joints[:, :, [7, 8, 10, 11]]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(q)
        q_6v = ax_to_6v(q)
        motion = vectorize_many([contacts, pos, q_6v])
    return motion[0].detach().cpu().numpy().astype(np.float32)


def _load_npy_motion(path: Path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        for key in ("motion", "motion_151", "poses", "pose_seq", "pose"):
            if key in data:
                arr = data[key]
                break
        else:
            return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 151:
        arr = arr[None]
    if arr.ndim != 2 or arr.shape[1] != 151:
        return None
    return arr.astype(np.float32)


def _npz_str(data, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    try:
        return str(np.asarray(data[key]).reshape(-1)[0])
    except Exception:
        return default


def _field(data, *keys):
    for key in keys:
        if key in data.files:
            return data[key]
    return None


def load_rag_candidates(rag_db: str, normalizer=None, pose_space: str = "normalized", max_candidates: int = 5000, sample_stride: int = 1):
    """Load old pose RAG DB or new ChoreoRAG motion-unit DB."""
    path = Path(rag_db)
    candidates: List[Dict[str, object]] = []

    if path.is_file() and path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "poses" not in data.files:
            raise ValueError(f"{path} has no 'poses' field")
        stored_pose_space = _npz_str(data, "pose_space", pose_space)
        is_unit_db = ("unit_motions" in data.files) or ("entry_poses" in data.files and "exit_poses" in data.files)

        poses = convert_loaded_motion(data["poses"], normalizer, stored_pose_space, pose_space)
        source = data["source"] if "source" in data.files else np.array(["rag_index"] * len(poses))
        source_frame = data["source_frame"] if "source_frame" in data.files else np.arange(len(poses))
        root_vel = data["root_vel"] if "root_vel" in data.files else np.zeros((len(poses), 2), dtype=np.float32)

        entry = _field(data, "entry_poses")
        exitp = _field(data, "exit_poses")
        units = _field(data, "unit_motions")
        if entry is not None:
            entry = convert_loaded_motion(entry, normalizer, stored_pose_space, pose_space)
        if exitp is not None:
            exitp = convert_loaded_motion(exitp, normalizer, stored_pose_space, pose_space)
        if units is not None:
            units = convert_loaded_motion(units, normalizer, stored_pose_space, pose_space)

        text_emb = _field(data, "motion_text_embedding", "text_embedding", "motion_embedding", "motion_embeddings")
        mmr_emb = _field(data, "motion_mmr_embedding", "mmr_embedding", "mmr_embeddings")
        motion_text = _field(data, "motion_text")
        contact_entry = _field(data, "contact_entry")
        contact_exit = _field(data, "contact_exit")
        unit_start = _field(data, "unit_start")
        unit_center = _field(data, "unit_center")
        unit_end = _field(data, "unit_end")

        scalar_fields = {
            "motion_energy": _field(data, "motion_energy"),
            "motion_energy_norm": _field(data, "motion_energy_norm"),
            "root_speed": _field(data, "root_speed"),
            "root_speed_norm": _field(data, "root_speed_norm"),
            "upper_activity": _field(data, "upper_activity"),
            "upper_activity_norm": _field(data, "upper_activity_norm"),
            "lower_activity": _field(data, "lower_activity"),
            "lower_activity_norm": _field(data, "lower_activity_norm"),
            "spatial_range": _field(data, "spatial_range"),
            "spatial_range_norm": _field(data, "spatial_range_norm"),
            "turning": _field(data, "turning"),
            "turning_norm": _field(data, "turning_norm"),
            "contact_stability": _field(data, "contact_stability"),
            "expressiveness_score": _field(data, "expressiveness_score"),
        }

        for i in range(0, len(poses), max(1, int(sample_stride))):
            item = {
                "pose": np.asarray(poses[i], dtype=np.float32),
                "source": str(source[i]),
                "source_frame": int(source_frame[i]),
                "root_vel": np.asarray(root_vel[i], dtype=np.float32),
                "motion_text_embedding": None if text_emb is None else np.asarray(text_emb[i], dtype=np.float32),
                "motion_embedding": None if text_emb is None else np.asarray(text_emb[i], dtype=np.float32),
                "motion_mmr_embedding": None if mmr_emb is None else np.asarray(mmr_emb[i], dtype=np.float32),
                "is_motion_unit": bool(is_unit_db),
                "entry_pose": None if entry is None else np.asarray(entry[i], dtype=np.float32),
                "exit_pose": None if exitp is None else np.asarray(exitp[i], dtype=np.float32),
                "unit_motion": None if units is None else np.asarray(units[i], dtype=np.float32),
                "contact_entry": None if contact_entry is None else np.asarray(contact_entry[i], dtype=np.float32),
                "contact_exit": None if contact_exit is None else np.asarray(contact_exit[i], dtype=np.float32),
                "motion_text": "" if motion_text is None else str(motion_text[i]),
                "unit_start": -1 if unit_start is None else int(unit_start[i]),
                "unit_center": -1 if unit_center is None else int(unit_center[i]),
                "unit_end": -1 if unit_end is None else int(unit_end[i]),
            }
            for key, values in scalar_fields.items():
                item[key] = None if values is None else float(values[i])
            candidates.append(item)
            if len(candidates) >= max_candidates:
                break
        return candidates

    if path.is_dir():
        files = sorted(list(path.glob("*.npy")) + list(path.glob("*.npz")) + list(path.glob("*.pkl")))
        for file in files:
            if len(candidates) >= max_candidates:
                break
            if file.suffix == ".npz":
                candidates.extend(load_rag_candidates(str(file), normalizer, pose_space, max_candidates - len(candidates), sample_stride))
                continue
            motion = _pkl_to_motion_151(file) if file.suffix == ".pkl" else _load_npy_motion(file)
            if motion is None:
                continue
            motion = convert_loaded_motion(motion, normalizer, "physical", pose_space)
            root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
            root_vel = np.zeros_like(root)
            if len(root) > 1:
                root_vel[1:] = root[1:] - root[:-1]
            for i in range(0, len(motion), max(1, int(sample_stride))):
                candidates.append({
                    "pose": motion[i].astype(np.float32),
                    "source": str(file),
                    "source_frame": int(i),
                    "root_vel": root_vel[i].astype(np.float32),
                    "is_motion_unit": False,
                    "motion_text_embedding": None,
                    "motion_embedding": None,
                    "motion_energy": None,
                    "motion_energy_norm": None,
                    "contact_stability": None,
                    "expressiveness_score": None,
                })
                if len(candidates) >= max_candidates:
                    break
        return candidates

    raise ValueError(f"Invalid rag_db: {rag_db}")


def _robust_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0 or float(values.max() - values.min()) <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(values, [10, 90])
    if float(hi - lo) <= 1e-8:
        lo, hi = float(values.min()), float(values.max())
    return np.clip((values - float(lo)) / max(float(hi - lo), 1e-8), 0.0, 1.0).astype(np.float32)


def annotate_candidate_statistics(candidates):
    """Attach normalized stats and expressiveness fallback in-place."""
    if not candidates:
        return

    keys = ["motion_energy", "root_speed", "upper_activity", "lower_activity", "spatial_range", "turning"]
    for key in keys:
        norm_key = f"{key}_norm"
        if all(c.get(norm_key) is not None for c in candidates):
            continue
        vals = np.asarray([float(c.get(key) or 0.0) for c in candidates], dtype=np.float32)
        norm = _robust_norm(vals)
        for c, v in zip(candidates, norm):
            c[norm_key] = float(v)

    if all(c.get("expressiveness_score") is not None for c in candidates):
        return

    for c in candidates:
        # Same conservative composition as the DB builder.  Root speed is small.
        expr = (
            0.30 * float(c.get("motion_energy_norm") or 0.0)
            + 0.30 * float(c.get("upper_activity_norm") or 0.0)
            + 0.20 * float(c.get("spatial_range_norm") or 0.0)
            + 0.15 * float(c.get("turning_norm") or 0.0)
            + 0.05 * float(c.get("root_speed_norm") or 0.0)
        )
        contact = c.get("contact_stability")
        if contact is not None and float(contact) < 0.45:
            expr *= 0.75
        c["expressiveness_score"] = float(np.clip(expr, 0.0, 1.0))


def normalize_01(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    x = np.nan_to_num(x)
    x = x - float(x.min())
    m = float(x.max())
    return np.zeros_like(x) if m <= 1e-8 else (x / m).astype(np.float32)


def smooth_1d(x, window=5):
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < 3:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), np.ones(window, dtype=np.float32) / window, mode="valid").astype(np.float32)


def audio_onset_score(audio_feature, onset_index=768):
    audio_feature = np.asarray(audio_feature, dtype=np.float32)
    if audio_feature.ndim != 2 or len(audio_feature) == 0:
        return np.zeros((0,), dtype=np.float32)
    if audio_feature.shape[1] > onset_index:
        x = np.maximum(audio_feature[:, onset_index], 0.0)
    elif len(audio_feature) > 1:
        x = np.zeros((len(audio_feature),), dtype=np.float32)
        x[1:] = np.linalg.norm(audio_feature[1:] - audio_feature[:-1], axis=-1)
        x[0] = x[1]
    else:
        x = np.zeros((len(audio_feature),), dtype=np.float32)
    return normalize_01(smooth_1d(x, 5))


def trajectory_curvature_score(traj_physical):
    traj = np.asarray(traj_physical, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    traj = traj[:, :2]
    if len(traj) < 3:
        return np.zeros((len(traj),), dtype=np.float32)
    v1, v2 = traj[1:-1] - traj[:-2], traj[2:] - traj[1:-1]
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-8, None)
    out = np.zeros((len(traj),), dtype=np.float32)
    out[1:-1] = 1.0 - np.clip(cos, -1.0, 1.0)
    return normalize_01(smooth_1d(out, 5))


def adaptive_auto_mid_count(num_frames, requested_count, fps=30):
    requested_count = max(0, int(requested_count))
    if requested_count <= 0:
        return 0
    if num_frames <= int(5.0 * fps):
        cap = 1
    elif num_frames <= int(8.0 * fps):
        cap = 2
    else:
        cap = 3
    effective = min(requested_count, cap)
    if effective < requested_count:
        print(f"⚠️ auto_mid_count={requested_count} too dense for {num_frames} frames; using {effective}.")
    return effective


def adaptive_min_gap(num_frames, count, requested_min_gap=18, edge_margin=8):
    available = max(1, int(num_frames) - 2 * int(edge_margin))
    return int(max(1, min(int(requested_min_gap), available // (max(1, int(count)) + 2))))


def choose_auto_frames(audio_feature, traj_physical, num_frames, count, existing_frames, min_gap=18, edge_margin=8, music_weight=0.6, trajectory_weight=0.4, fps=30):
    count = adaptive_auto_mid_count(num_frames, count, fps=fps)
    if count <= 0:
        return []
    min_gap = adaptive_min_gap(num_frames, count, min_gap, edge_margin)
    onset = audio_onset_score(audio_feature)
    if len(onset) != num_frames:
        onset = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, max(1, len(onset))), onset if len(onset) else [0.0]).astype(np.float32)
    curv = trajectory_curvature_score(traj_physical)
    if len(curv) != num_frames:
        curv = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, max(1, len(curv))), curv if len(curv) else [0.0]).astype(np.float32)
    score = float(music_weight) * normalize_01(onset) + float(trajectory_weight) * normalize_01(curv)
    blocked = np.zeros((num_frames,), dtype=bool)
    for f in list(existing_frames) + [0, num_frames - 1]:
        blocked[max(0, int(f) - min_gap) : min(num_frames, int(f) + min_gap + 1)] = True
    blocked[:edge_margin] = True
    blocked[num_frames - edge_margin :] = True

    frames = []
    for _ in range(count):
        s = score.copy()
        s[blocked] = -1e9
        f = int(s.argmax())
        if s[f] < -1e8:
            break
        frames.append(f)
        blocked[max(0, f - min_gap) : min(num_frames, f + min_gap + 1)] = True

    if len(frames) < count:
        for i in range(count):
            f = int(round((i + 1) * (num_frames - 1) / (count + 1)))
            f = max(edge_margin, min(num_frames - edge_margin - 1, f))
            if all(abs(f - x) >= min_gap for x in frames + list(existing_frames) + [0, num_frames - 1]):
                frames.append(f)
            if len(frames) >= count:
                break
    return sorted(set(frames))[:count]


def pose_feature(pose):
    return np.asarray(pose, dtype=np.float32).reshape(-1)[POSE_FEATURE_INDEX].astype(np.float32)


def pose_distance(a, b):
    return float(np.sqrt(np.mean((pose_feature(a) - pose_feature(b)) ** 2)))


def interpolate_pose_at_frame(frame, anchors):
    anchors = sorted([(int(f), np.asarray(p, dtype=np.float32)) for f, p in anchors], key=lambda x: x[0])
    if frame <= anchors[0][0]:
        return anchors[0][1]
    if frame >= anchors[-1][0]:
        return anchors[-1][1]
    for (f0, p0), (f1, p1) in zip(anchors[:-1], anchors[1:]):
        if f0 <= frame <= f1:
            a = (frame - f0) / max(float(f1 - f0), 1.0)
            return ((1 - a) * p0 + a * p1).astype(np.float32)
    return anchors[-1][1]


def neighbor_anchors(frame, anchors):
    anchors = sorted([(int(f), np.asarray(p, dtype=np.float32)) for f, p in anchors], key=lambda x: x[0])
    prev_a, next_a = anchors[0], anchors[-1]
    for item in anchors:
        if item[0] <= frame:
            prev_a = item
        if item[0] >= frame:
            next_a = item
            break
    return prev_a, next_a


def target_traj_tangent(traj_physical, frame):
    traj = np.asarray(traj_physical, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    traj = traj[:, :2]
    if len(traj) < 2:
        return np.zeros((2,), dtype=np.float32)
    f0, f1 = max(0, frame - 1), min(len(traj) - 1, frame + 1)
    v = traj[f1] - traj[f0]
    n = float(np.linalg.norm(v))
    return np.zeros((2,), dtype=np.float32) if n <= 1e-8 else (v / n).astype(np.float32)


def direction_cost(v, t):
    if v is None:
        return 0.5
    v = np.asarray(v, dtype=np.float32).reshape(-1)[:2]
    t = np.asarray(t, dtype=np.float32).reshape(-1)[:2]
    nv, nt = np.linalg.norm(v), np.linalg.norm(t)
    if nv <= 1e-8 or nt <= 1e-8:
        return 0.5
    return 0.5 * (1.0 - np.clip(float(np.dot(v, t) / (nv * nt)), -1.0, 1.0))


def cosine_distance(a, b):
    if a is None or b is None:
        return 0.5
    a, b = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    if len(a) != len(b):
        return 0.5
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-8 or nb <= 1e-8:
        return 0.5
    return 0.5 * (1.0 - np.clip(float(np.dot(a, b) / (na * nb)), -1.0, 1.0))


def contact_pattern(pose):
    return (np.asarray(pose, dtype=np.float32).reshape(-1)[:4] > 0.5).astype(np.float32)


def contact_mismatch(a, b):
    if a is None or b is None:
        return 0.25
    a = (np.asarray(a, dtype=np.float32).reshape(-1)[:4] > 0.5)
    b = (np.asarray(b, dtype=np.float32).reshape(-1)[:4] > 0.5)
    if len(a) < 4 or len(b) < 4:
        return 0.25
    return float(np.mean(a != b))


def is_same_source_region(candidate, selected_keyframes, source_gap=120, disallow_same_source=False):
    src, sf = str(candidate.get("source", "")), int(candidate.get("source_frame", -1))
    for prev in selected_keyframes:
        if src and src == str(prev.source):
            if disallow_same_source:
                return True, "same_source"
            if sf >= 0 and abs(sf - int(prev.source_frame)) < int(source_gap):
                return True, "same_source_region"
    return False, ""


def encode_text_query(text, model_name="", device="cpu"):
    text = str(text or "").strip()
    if not text or TextBridgeEncoder is None:
        return None
    enc = TextBridgeEncoder(
        model_name=model_name or os.environ.get("EDGE_TEXT_BRIDGE_MODEL", "BAAI/bge-small-zh-v1.5"),
        device=device or os.environ.get("EDGE_TEXT_BRIDGE_DEVICE", "cpu"),
    )
    z = enc.encode([text])[0].astype(np.float32)
    return z / max(float(np.linalg.norm(z)), 1e-8)


def get_choreo_plan(audio_feature, num_frames):
    plan_json = os.environ.get("EDGE_CHOREO_PLAN_JSON", "").strip()
    style_hint = os.environ.get("EDGE_CHOREO_STYLE_HINT", "敦煌舞，飞天感，上肢舒展，重心稳定")
    if load_or_build_choreo_plan is not None:
        return load_or_build_choreo_plan(
            plan_json=plan_json,
            audio_feature=audio_feature,
            num_frames=num_frames,
            style_hint=style_hint,
            max_segments=5,
        )
    if plan_json and Path(plan_json).is_file():
        with open(plan_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "planner": "fallback",
        "global_style": style_hint,
        "segments": [{
            "id": 0,
            "start": 0,
            "end": num_frames - 1,
            "center": num_frames // 2,
            "query_text": f"中等能量，平稳移动，上肢舒展，重心稳定，{style_hint}",
            "motion_prompt": f"中等能量，平稳移动，上肢舒展，重心稳定，{style_hint}",
            "energy_target": 0.55,
            "tension_target": 0.55,
            "phase": "flow",
        }],
    }


def segment_for_frame(plan, frame):
    segments = list(plan.get("segments", []))
    if not segments:
        return {"id": -1, "start": 0, "end": 10**9, "query_text": "", "energy_target": 0.55, "tension_target": 0.55, "phase": "flow"}
    for seg in segments:
        if int(seg.get("start", 0)) <= frame <= int(seg.get("end", 10**9)):
            return seg
    return min(segments, key=lambda s: abs(int(s.get("center", (int(s.get("start", 0)) + int(s.get("end", 0))) // 2)) - frame))


def _candidate_expr(cand: Dict[str, object]) -> float:
    return float(np.clip(float(cand.get("expressiveness_score") or 0.0), 0.0, 1.0))


def _candidate_energy(cand: Dict[str, object]) -> float:
    return float(np.clip(float(cand.get("motion_energy_norm") or 0.0), 0.0, 1.0))


def _phase_modulation(segment, weights: Dict[str, float]) -> Dict[str, float]:
    out = dict(weights)
    if not _env_bool("EDGE_TENSION_AWARE_PLANNER", False):
        return out

    segment = segment or {}
    phase = str(segment.get("phase", "flow")).lower()
    tension = float(np.clip(float(segment.get("tension_target", segment.get("energy_target", 0.55))), 0.0, 1.0))

    if phase == "attack":
        out["energy_target"] = max(out["energy_target"], float(segment.get("energy_target", 0.75)), 0.72)
        out["min_expr"] = max(out["min_expr"], float(segment.get("min_expressiveness", 0.50)), 0.48 + 0.20 * tension)
        out["min_energy"] = max(out["min_energy"], float(segment.get("min_energy", 0.40)))
        out["expr_bonus"] += 0.25 + 0.20 * tension
        out["energy_bonus"] += 0.10 + 0.10 * tension
        out["w_contact"] *= 0.65
        out["w_contact_phase"] *= 0.70
        out["w_entry"] *= 0.75
        out["w_exit"] *= 0.75
        out["w_energy"] *= 0.65  # do not over-penalize non-medium-energy units
    elif phase == "flow":
        out["min_expr"] = max(out["min_expr"], float(segment.get("min_expressiveness", 0.30)))
        out["expr_bonus"] += 0.10 + 0.10 * tension
        out["w_traj"] *= 1.15
        out["w_diversity"] *= 1.15
    elif phase == "pose":
        out["min_expr"] = min(out["min_expr"], float(segment.get("min_expressiveness", -1.0)))
        out["min_energy"] = min(out["min_energy"], float(segment.get("min_energy", -1.0)))
        out["expr_bonus"] *= 0.35
        out["energy_bonus"] *= 0.35
        out["w_contact"] *= 1.40
        out["w_contact_phase"] *= 1.35
        out["w_entry"] *= 1.25
        out["w_exit"] *= 1.25
    return out


def _homogeneity_cost(expr: float, energy: float, selected_keyframes: List[AutoKeyframe], num_frames: int) -> float:
    weight = _env_float("EDGE_UNIT_HOMOGENEITY_WEIGHT", 0.0)
    min_frames = _env_int("EDGE_UNIT_HOMOGENEITY_MIN_FRAMES", 240)
    if weight <= 0.0 or num_frames < min_frames or not selected_keyframes:
        return 0.0
    prev = selected_keyframes[-1]
    prev_expr = float(prev.score_parts.get("expressiveness_score", prev.score_parts.get("motion_energy_norm", 0.0)))
    prev_energy = float(prev.score_parts.get("motion_energy_norm", 0.0))
    expr_sim = 1.0 - min(1.0, abs(float(expr) - prev_expr) / 0.35)
    energy_sim = 1.0 - min(1.0, abs(float(energy) - prev_energy) / 0.35)
    return float(weight) * max(0.0, 0.5 * expr_sim + 0.5 * energy_sim)


def choose_candidate_for_frame(frame, candidates, anchors, traj_physical, selected_poses, selected_keyframes=None, text_embedding=None, segment=None,
                               w_text=0.50, w_pose=1.0, w_traj=0.30, w_diversity=0.25, w_energy=0.35,
                               energy_target=0.55, energy_band=0.25, w_contact=0.50, w_contact_diversity=0.30,
                               w_end=0.25, w_entry=0.60, w_exit=0.60, w_contact_phase=0.85,
                               source_gap=120, disallow_same_source=False, energy_rerank_top_k=80, energy_rerank_weight=0.25):
    selected_keyframes = selected_keyframes or []
    ref_pose = interpolate_pose_at_frame(frame, anchors)
    prev_a, next_a = neighbor_anchors(frame, anchors)
    tangent = target_traj_tangent(traj_physical, frame)
    scored = []
    rejected_by_source = 0
    rejected_by_expr = 0
    rejected_by_energy = 0
    num_frames = int(anchors[-1][0]) + 1 if anchors else 0

    weights = _phase_modulation(segment, {
        "w_text": float(w_text),
        "w_pose": float(w_pose),
        "w_traj": float(w_traj),
        "w_diversity": float(w_diversity),
        "w_energy": float(w_energy),
        "energy_target": float(energy_target),
        "w_contact": float(w_contact),
        "w_contact_diversity": float(w_contact_diversity),
        "w_end": float(w_end),
        "w_entry": float(w_entry),
        "w_exit": float(w_exit),
        "w_contact_phase": float(w_contact_phase),
        "min_expr": _env_float("EDGE_UNIT_MIN_EXPRESSIVENESS", -1.0),
        "expr_bonus": _env_float("EDGE_UNIT_EXPRESSIVENESS_BONUS", 0.0),
        "min_energy": _env_float("EDGE_UNIT_MIN_ENERGY", -1.0),
        "energy_bonus": _env_float("EDGE_UNIT_ENERGY_BONUS", 0.0),
    })

    for cand in candidates:
        blocked, _ = is_same_source_region(cand, selected_keyframes, source_gap, disallow_same_source)
        if blocked:
            rejected_by_source += 1
            continue

        en = _candidate_energy(cand)
        expr = _candidate_expr(cand)
        if weights["min_expr"] >= 0.0 and expr < weights["min_expr"]:
            rejected_by_expr += 1
            continue
        if weights["min_energy"] >= 0.0 and en < weights["min_energy"]:
            rejected_by_energy += 1
            continue

        pose = np.asarray(cand["pose"], dtype=np.float32)
        entry = pose if cand.get("entry_pose") is None else np.asarray(cand["entry_pose"], dtype=np.float32)
        exitp = pose if cand.get("exit_pose") is None else np.asarray(cand["exit_pose"], dtype=np.float32)

        text_cost = cosine_distance(text_embedding, cand.get("motion_text_embedding", cand.get("motion_embedding")))
        pose_cost = pose_distance(pose, ref_pose)
        traj_cost = direction_cost(cand.get("root_vel"), tangent)
        diversity_cost = 0.0 if not selected_poses else 1.0 / (1.0 + min(pose_distance(pose, p) for p in selected_poses))

        e_target = float(np.clip(weights["energy_target"], 0.0, 1.0))
        energy_cost = float(min(abs(en - e_target) / max(float(energy_band), 1e-6), 2.0))

        contact_stability = cand.get("contact_stability")
        contact_cost = 0.25 if contact_stability is None else 1.0 - float(np.clip(contact_stability, 0.0, 1.0))
        contact_div = 0.0 if not selected_poses else max(float((contact_pattern(pose) == contact_pattern(p)).mean()) for p in selected_poses)

        entry_cost = pose_distance(entry, prev_a[1])
        exit_cost = pose_distance(exitp, next_a[1])
        ce = cand.get("contact_entry", contact_pattern(entry))
        cx = cand.get("contact_exit", contact_pattern(exitp))
        contact_phase = 0.5 * (contact_mismatch(contact_pattern(prev_a[1]), ce) + contact_mismatch(cx, contact_pattern(next_a[1])))

        end_alpha = float(frame) / max(float(anchors[-1][0]), 1.0)
        end_cost = end_alpha * pose_distance(pose, anchors[-1][1])
        homogeneity = _homogeneity_cost(expr, en, selected_keyframes, num_frames)

        score = (
            weights["w_text"] * text_cost
            + weights["w_pose"] * pose_cost
            + weights["w_traj"] * traj_cost
            + weights["w_diversity"] * diversity_cost
            + weights["w_energy"] * energy_cost
            + weights["w_contact"] * contact_cost
            + weights["w_contact_diversity"] * contact_div
            + weights["w_end"] * end_cost
            + weights["w_entry"] * entry_cost
            + weights["w_exit"] * exit_cost
            + weights["w_contact_phase"] * contact_phase
            + homogeneity
            - weights["expr_bonus"] * expr
            - weights["energy_bonus"] * en
        )

        kf = AutoKeyframe(
            frame=int(frame),
            pose=pose.astype(np.float32),
            score=float(score),
            source=str(cand.get("source", "")),
            source_frame=int(cand.get("source_frame", -1)),
            unit_motion=None if cand.get("unit_motion") is None else np.asarray(cand["unit_motion"], dtype=np.float32),
            unit_start=int(cand.get("unit_start", -1)),
            unit_center=int(cand.get("unit_center", cand.get("source_frame", -1))),
            unit_end=int(cand.get("unit_end", -1)),
            motion_text=str(cand.get("motion_text", "")),
            segment_id=int(segment.get("id", -1)) if segment else -1,
            segment_prompt=str(segment.get("query_text", segment.get("motion_prompt", ""))) if segment else "",
            score_parts={
                "score_before_rerank": float(score),
                "text_cost": float(text_cost),
                "pose_cost": float(pose_cost),
                "trajectory_direction_cost": float(traj_cost),
                "diversity_cost": float(diversity_cost),
                "energy_cost": float(energy_cost),
                "motion_energy_norm": float(en),
                "expressiveness_score": float(expr),
                "expressiveness_bonus": float(weights["expr_bonus"]),
                "energy_bonus": float(weights["energy_bonus"]),
                "min_expressiveness": float(weights["min_expr"]),
                "min_energy": float(weights["min_energy"]),
                "contact_stability_cost": float(contact_cost),
                "contact_diversity_cost": float(contact_div),
                "entry_compat_cost": float(entry_cost),
                "exit_compat_cost": float(exit_cost),
                "contact_phase_cost": float(contact_phase),
                "end_compat_cost": float(end_cost),
                "homogeneity_cost": float(homogeneity),
                "phase": str(segment.get("phase", "")) if segment else "",
                "tension_target": float(segment.get("tension_target", -1.0)) if segment else -1.0,
                "rejected_by_source_before_pool": int(rejected_by_source),
                "rejected_by_expressiveness_before_pool": int(rejected_by_expr),
                "rejected_by_energy_before_pool": int(rejected_by_energy),
            },
        )
        scored.append((float(score), float(expr), float(en), kf))

    if not scored:
        # Avoid hard failure when thresholds are too strict.  Relax new filters,
        # then source diversity if necessary.  The metadata will reveal this.
        if weights["min_expr"] >= 0.0 or weights["min_energy"] >= 0.0:
            old_min_expr = os.environ.get("EDGE_UNIT_MIN_EXPRESSIVENESS")
            old_min_energy = os.environ.get("EDGE_UNIT_MIN_ENERGY")
            os.environ["EDGE_UNIT_MIN_EXPRESSIVENESS"] = "-1"
            os.environ["EDGE_UNIT_MIN_ENERGY"] = "-1"
            try:
                kf = choose_candidate_for_frame(
                    frame, candidates, anchors, traj_physical, selected_poses, selected_keyframes, text_embedding, segment,
                    w_text, w_pose, w_traj, w_diversity, w_energy, energy_target, energy_band, w_contact,
                    w_contact_diversity, w_end, w_entry, w_exit, w_contact_phase, source_gap, disallow_same_source,
                    energy_rerank_top_k, energy_rerank_weight,
                )
                kf.score_parts["threshold_relaxed_fallback"] = 1.0
                return kf
            finally:
                if old_min_expr is None:
                    os.environ.pop("EDGE_UNIT_MIN_EXPRESSIVENESS", None)
                else:
                    os.environ["EDGE_UNIT_MIN_EXPRESSIVENESS"] = old_min_expr
                if old_min_energy is None:
                    os.environ.pop("EDGE_UNIT_MIN_ENERGY", None)
                else:
                    os.environ["EDGE_UNIT_MIN_ENERGY"] = old_min_energy
        if selected_keyframes and (source_gap > 0 or disallow_same_source):
            return choose_candidate_for_frame(
                frame, candidates, anchors, traj_physical, selected_poses, [], text_embedding, segment,
                w_text, w_pose, w_traj, w_diversity, w_energy, energy_target, energy_band, w_contact,
                w_contact_diversity, w_end, w_entry, w_exit, w_contact_phase, 0, False,
                energy_rerank_top_k, energy_rerank_weight,
            )
        raise RuntimeError("No RAG candidate available")

    scored.sort(key=lambda x: x[0])
    top_k = int(energy_rerank_top_k)
    if top_k > 0 and float(energy_rerank_weight) > 0:
        pool = scored[: min(top_k, len(scored))]

        def rerank_key(item):
            base_score, expr_value, energy_value, _kf = item
            # In expressiveness mode, rerank by high expression rather than medium energy.
            expr_bonus = max(0.0, _env_float("EDGE_UNIT_EXPRESSIVENESS_BONUS", 0.0))
            if _env_bool("EDGE_TENSION_AWARE_PLANNER", False):
                expr_bonus += 0.15
            e_target = float(np.clip(weights["energy_target"], 0.0, 1.0))
            e_band = max(float(energy_band), 1e-6)
            target_cost = float(min(abs(float(energy_value) - e_target) / e_band, 2.0))
            return float(base_score) + float(energy_rerank_weight) * target_cost - 0.25 * expr_bonus * float(expr_value)

        best_base, best_expr, best_energy, best_kf = min(pool, key=rerank_key)
        final_score = float(rerank_key((best_base, best_expr, best_energy, best_kf)))
        best_kf.score = final_score
        best_kf.score_parts["energy_rerank_top_k"] = int(top_k)
        best_kf.score_parts["energy_rerank_weight"] = float(energy_rerank_weight)
        best_kf.score_parts["score_after_energy_rerank"] = final_score
        best_kf.score_parts["compatible_pool_size"] = int(len(pool))
        return best_kf

    best_kf = scored[0][3]
    best_kf.score_parts["energy_rerank_top_k"] = 0
    best_kf.score_parts["energy_rerank_weight"] = 0.0
    best_kf.score_parts["score_after_energy_rerank"] = float(best_kf.score)
    return best_kf


def plan_auto_keyframes(start_pose: np.ndarray, end_pose: np.ndarray, user_mid_poses: Sequence[np.ndarray], user_mid_frames: Sequence[int],
                        audio_feature: np.ndarray, traj_physical: np.ndarray, rag_db: str, normalizer=None, num_frames: int = 150,
                        max_auto_keyframes: int = 3, min_gap: int = 18, rag_pose_space: str = "normalized", max_candidates: int = 5000,
                        sample_stride: int = 3, music_weight: float = 0.6, trajectory_weight: float = 0.4, mmr_checkpoint: str = "",
                        mmr_weight: float = 0.0, pose_weight: float = 1.0, diversity_weight: float = 0.25, energy_weight: float = 0.45,
                        energy_target: float = 0.55, energy_band: float = 0.25, contact_weight: float = 0.85,
                        contact_diversity_weight: float = 0.60, end_weight: float = 0.30, source_gap: int = 120,
                        disallow_same_source: bool = False, energy_rerank_top_k: int = 80, energy_rerank_weight: float = 0.25,
                        **kwargs) -> AutoKeyframePlan:
    candidates = load_rag_candidates(
        rag_db,
        normalizer=normalizer,
        pose_space=rag_pose_space,
        max_candidates=max_candidates,
        sample_stride=sample_stride,
    )
    if not candidates:
        raise RuntimeError(f"No valid RAG candidates found in {rag_db}")
    annotate_candidate_statistics(candidates)

    anchors = [(0, start_pose), (num_frames - 1, end_pose)]
    for f, p in zip(user_mid_frames, user_mid_poses):
        anchors.append((int(f), np.asarray(p, dtype=np.float32)))
    anchors = sorted(anchors, key=lambda x: x[0])

    frames = choose_auto_frames(
        audio_feature,
        traj_physical,
        num_frames,
        int(max_auto_keyframes),
        [f for f, _ in anchors],
        int(min_gap),
        music_weight=music_weight,
        trajectory_weight=trajectory_weight,
    )

    plan = get_choreo_plan(audio_feature, num_frames)
    selected: List[AutoKeyframe] = []
    selected_poses: List[np.ndarray] = []

    w_text = _env_float("EDGE_TEXT_BRIDGE_WEIGHT", 0.50)
    w_entry = _env_float("EDGE_UNIT_ENTRY_WEIGHT", 0.60)
    w_exit = _env_float("EDGE_UNIT_EXIT_WEIGHT", 0.60)
    w_contact_phase = _env_float("EDGE_UNIT_CONTACT_PHASE_WEIGHT", 0.85)

    for frame in frames:
        seg = segment_for_frame(plan, frame)
        query_text = str(seg.get("query_text", seg.get("motion_prompt", "")))
        text_embedding = encode_text_query(query_text)
        kf = choose_candidate_for_frame(
            frame=frame,
            candidates=candidates,
            anchors=anchors + [(k.frame, k.pose) for k in selected],
            traj_physical=traj_physical,
            selected_poses=selected_poses,
            selected_keyframes=selected,
            text_embedding=text_embedding,
            segment=seg,
            w_text=w_text,
            w_pose=pose_weight,
            w_traj=trajectory_weight,
            w_diversity=diversity_weight,
            w_energy=energy_weight,
            energy_target=float(seg.get("energy_target", energy_target)),
            energy_band=energy_band,
            w_contact=contact_weight,
            w_contact_diversity=contact_diversity_weight,
            w_end=end_weight,
            w_entry=w_entry,
            w_exit=w_exit,
            w_contact_phase=w_contact_phase,
            source_gap=source_gap,
            disallow_same_source=disallow_same_source,
            energy_rerank_top_k=energy_rerank_top_k,
            energy_rerank_weight=energy_rerank_weight,
        )
        selected.append(kf)
        selected_poses.append(kf.pose)

    selected = sorted(selected, key=lambda x: x.frame)
    return AutoKeyframePlan(
        keyframes=selected,
        frame_candidates=frames,
        meta={
            "rag_db": str(rag_db),
            "candidate_count": len(candidates),
            "max_auto_keyframes_requested": int(max_auto_keyframes),
            "max_auto_keyframes_effective": len(frames),
            "min_gap": int(min_gap),
            "music_weight": float(music_weight),
            "trajectory_weight": float(trajectory_weight),
            "rag_pose_space": rag_pose_space,
            "pose_weight": float(pose_weight),
            "diversity_weight": float(diversity_weight),
            "energy_weight": float(energy_weight),
            "contact_weight": float(contact_weight),
            "contact_diversity_weight": float(contact_diversity_weight),
            "end_weight": float(end_weight),
            "entry_weight": float(w_entry),
            "exit_weight": float(w_exit),
            "contact_phase_weight": float(w_contact_phase),
            "source_gap": int(source_gap),
            "disallow_same_source": bool(disallow_same_source),
            "energy_rerank_top_k": int(energy_rerank_top_k),
            "energy_rerank_weight": float(energy_rerank_weight),
            "tension_aware": bool(_env_bool("EDGE_TENSION_AWARE_PLANNER", False)),
            "min_expressiveness": float(_env_float("EDGE_UNIT_MIN_EXPRESSIVENESS", -1.0)),
            "expressiveness_bonus": float(_env_float("EDGE_UNIT_EXPRESSIVENESS_BONUS", 0.0)),
            "homogeneity_weight": float(_env_float("EDGE_UNIT_HOMOGENEITY_WEIGHT", 0.0)),
            "choreo_plan": plan,
        },
    )


def _serializable_keyframe(kf: AutoKeyframe, path: str = "", unit_path: str = "") -> Dict[str, object]:
    return {
        "frame": int(kf.frame),
        "pose_path": str(path),
        "path": str(path),
        "unit_path": str(unit_path),
        "score": float(kf.score),
        "source": kf.source,
        "source_frame": int(kf.source_frame),
        "unit_start": int(kf.unit_start),
        "unit_center": int(kf.unit_center),
        "unit_end": int(kf.unit_end),
        "motion_text": kf.motion_text,
        "segment_id": int(kf.segment_id),
        "segment_prompt": kf.segment_prompt,
        "score_parts": kf.score_parts,
    }


def save_auto_keyframes(plan: Optional[AutoKeyframePlan] = None, keyframes: Optional[Sequence[AutoKeyframe]] = None,
                        out_dir: Optional[str] = None, output_dir: Optional[str] = None, prefix: str = "auto_mid",
                        out_motion_path: Optional[str] = None, **kwargs):
    """Save selected auto mid poses.

    This signature is intentionally flexible because generate_controlled.py has
    used multiple save_auto_keyframes calling conventions across branches.
    Returns a list of dict records, which current generate_controlled.py already
    understands for pose path/frame.  unit_path is included for the Phase-4 patch.
    """
    if plan is not None and keyframes is None:
        keyframes = plan.keyframes
    keyframes = list(keyframes or [])

    if out_dir is None:
        out_dir = output_dir
    if out_dir is None and out_motion_path:
        out_dir = str(Path(out_motion_path).parent)
        if prefix == "auto_mid":
            prefix = Path(out_motion_path).stem + "_auto_mid"
    if out_dir is None:
        out_dir = "."
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for idx, kf in enumerate(keyframes, start=1):
        pose_path = out_path / f"{prefix}_{idx:02d}_f{int(kf.frame):03d}.npy"
        np.save(pose_path, np.asarray(kf.pose, dtype=np.float32))
        unit_path = ""
        if kf.unit_motion is not None:
            unit_path_obj = out_path / f"{prefix}_{idx:02d}_f{int(kf.frame):03d}_unit.npy"
            np.save(unit_path_obj, np.asarray(kf.unit_motion, dtype=np.float32))
            unit_path = str(unit_path_obj)
        saved.append(_serializable_keyframe(kf, str(pose_path), unit_path))

    meta_path = out_path / f"{prefix}_plan.json"
    payload = {
        "keyframes": saved,
        "auto_keyframes": saved,  # backward-compatible alias
        "frame_candidates": [] if plan is None else list(plan.frame_candidates),
        "planner_meta": {} if plan is None else plan.meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for rec in saved:
        rec["plan_json"] = str(meta_path)
    return saved


def append_csv(existing: str, values: Sequence[object]) -> str:
    old = [x.strip() for x in str(existing or "").replace(";", ",").split(",") if x.strip()]
    return ",".join(old + [str(v) for v in values])


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_db", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pose_space", default="normalized", choices=["normalized", "physical"])
    parser.add_argument("--max_candidates", type=int, default=5000)
    args = parser.parse_args()
    cands = load_rag_candidates(args.rag_db, None, args.pose_space, args.max_candidates)
    annotate_candidate_statistics(cands)
    print(f"loaded candidates: {len(cands)}")
    if cands:
        print({k: v for k, v in cands[0].items() if k not in {"pose", "unit_motion", "motion_text_embedding", "motion_embedding"}})
        print("pose shape:", cands[0]["pose"].shape)


if __name__ == "__main__":
    _cli()
