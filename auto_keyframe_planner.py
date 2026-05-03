"""ChoreoRAG auto middle-keyframe planner for EDGE.

Direct replacement for auto_keyframe_planner.py.

Key features:
- Backward compatible with old frame-level RAG DB.
- Supports choreo-unit RAG DB built by build_choreo_unit_rag_db.py.
- Supports DanceChat/TM2D-style time-ranged choreography plan through:
    EDGE_CHOREO_PLAN_JSON=/path/to/plan.json
  or heuristic fallback.
- Does not train EDGE; only changes inference-time auto-mid planning.

Environment variables:
  EDGE_CHOREO_PLAN_JSON          optional JSON plan path
  EDGE_CHOREO_STYLE_HINT         default: 敦煌舞，飞天感，上肢舒展，重心稳定
  EDGE_TEXT_BRIDGE_MODEL         default: BAAI/bge-small-zh-v1.5
  EDGE_TEXT_BRIDGE_DEVICE        default: cpu
  EDGE_TEXT_BRIDGE_WEIGHT        default: 0.50
  EDGE_UNIT_ENTRY_WEIGHT         default: 0.60
  EDGE_UNIT_EXIT_WEIGHT          default: 0.60
  EDGE_UNIT_CONTACT_PHASE_WEIGHT default: 0.85
  EDGE_MOTION_UNIT_MODE          auto|on|off, default auto
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
except Exception:
    ax_to_6v = None
    vectorize_many = None
    Normalizer = None
    SMPLSkeleton = None

try:
    from model.text_bridge_encoder import TextBridgeEncoder
except Exception:
    try:
        from text_bridge_encoder import TextBridgeEncoder  # type: ignore
    except Exception:
        TextBridgeEncoder = None  # type: ignore

try:
    from music_choreo_planner import load_or_build_choreo_plan
except Exception:
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
        energy = _field(data, "motion_energy")
        contact_stability = _field(data, "contact_stability")
        motion_text = _field(data, "motion_text")
        contact_entry = _field(data, "contact_entry")
        contact_exit = _field(data, "contact_exit")
        unit_start = _field(data, "unit_start")
        unit_center = _field(data, "unit_center")
        unit_end = _field(data, "unit_end")

        for i in range(0, len(poses), max(1, int(sample_stride))):
            candidates.append({
                "pose": np.asarray(poses[i], dtype=np.float32),
                "source": str(source[i]),
                "source_frame": int(source_frame[i]),
                "root_vel": np.asarray(root_vel[i], dtype=np.float32),
                "motion_text_embedding": None if text_emb is None else np.asarray(text_emb[i], dtype=np.float32),
                "motion_embedding": None if text_emb is None else np.asarray(text_emb[i], dtype=np.float32),
                "motion_mmr_embedding": None if mmr_emb is None else np.asarray(mmr_emb[i], dtype=np.float32),
                "motion_energy": None if energy is None else float(energy[i]),
                "contact_stability": None if contact_stability is None else float(contact_stability[i]),
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
            })
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
                    "contact_stability": None,
                })
                if len(candidates) >= max_candidates:
                    break
        return candidates

    raise ValueError(f"Invalid rag_db: {rag_db}")


def annotate_candidate_statistics(candidates):
    if not candidates:
        return
    energies = np.asarray([float(c.get("motion_energy") or 0.0) for c in candidates], dtype=np.float32)
    if float(energies.max() - energies.min()) > 1e-8:
        lo, hi = np.percentile(energies, [10, 90])
        denom = max(float(hi - lo), 1e-8)
        e_norm = np.clip((energies - float(lo)) / denom, 0.0, 1.0)
    else:
        e_norm = np.zeros_like(energies)
    for c, e in zip(candidates, e_norm):
        c["motion_energy_norm"] = float(e)


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
        }],
    }


def segment_for_frame(plan, frame):
    segments = list(plan.get("segments", []))
    if not segments:
        return {"id": -1, "start": 0, "end": 10**9, "query_text": "", "energy_target": 0.55}
    for seg in segments:
        if int(seg.get("start", 0)) <= frame <= int(seg.get("end", 10**9)):
            return seg
    return min(segments, key=lambda s: abs(int(s.get("center", (int(s.get("start", 0)) + int(s.get("end", 0))) // 2)) - frame))


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
    rejected = 0

    for cand in candidates:
        blocked, _ = is_same_source_region(cand, selected_keyframes, source_gap, disallow_same_source)
        if blocked:
            rejected += 1
            continue

        pose = np.asarray(cand["pose"], dtype=np.float32)
        entry = pose if cand.get("entry_pose") is None else np.asarray(cand["entry_pose"], dtype=np.float32)
        exitp = pose if cand.get("exit_pose") is None else np.asarray(cand["exit_pose"], dtype=np.float32)

        text_cost = cosine_distance(text_embedding, cand.get("motion_text_embedding", cand.get("motion_embedding")))
        pose_cost = pose_distance(pose, ref_pose)
        traj_cost = direction_cost(cand.get("root_vel"), tangent)
        diversity_cost = 0.0 if not selected_poses else 1.0 / (1.0 + min(pose_distance(pose, p) for p in selected_poses))

        en = float(cand.get("motion_energy_norm", 0.0) or 0.0)
        e_target = float(np.clip(energy_target, 0.0, 1.0))
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

        score = (float(w_text) * text_cost + float(w_pose) * pose_cost + float(w_traj) * traj_cost +
                 float(w_diversity) * diversity_cost + float(w_energy) * energy_cost +
                 float(w_contact) * contact_cost + float(w_contact_diversity) * contact_div +
                 float(w_end) * end_cost + float(w_entry) * entry_cost +
                 float(w_exit) * exit_cost + float(w_contact_phase) * contact_phase)

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
                "contact_stability_cost": float(contact_cost),
                "contact_diversity_cost": float(contact_div),
                "entry_compat_cost": float(entry_cost),
                "exit_compat_cost": float(exit_cost),
                "contact_phase_cost": float(contact_phase),
                "end_compat_cost": float(end_cost),
                "is_motion_unit": float(bool(cand.get("is_motion_unit", False))),
                "rejected_by_source_before_pool": int(rejected),
            },
        )
        scored.append((score, en, kf))

    if not scored:
        if selected_keyframes and (source_gap > 0 or disallow_same_source):
            return choose_candidate_for_frame(frame, candidates, anchors, traj_physical, selected_poses, [], text_embedding, segment,
                                             w_text, w_pose, w_traj, w_diversity, w_energy, energy_target, energy_band,
                                             w_contact, w_contact_diversity, w_end, w_entry, w_exit, w_contact_phase, 0, False,
                                             energy_rerank_top_k, energy_rerank_weight)
        raise RuntimeError("No compatible RAG candidate")

    scored.sort(key=lambda x: x[0])
    if int(energy_rerank_top_k) > 0 and float(energy_rerank_weight) > 0:
        pool = scored[: min(int(energy_rerank_top_k), len(scored))]
        def key(item):
            base, en, _ = item
            return float(base) + float(energy_rerank_weight) * min(abs(float(en) - float(energy_target)) / max(float(energy_band), 1e-6), 2.0)
        best = min(pool, key=key)
        best[2].score = float(key(best))
        best[2].score_parts["score_after_energy_rerank"] = float(best[2].score)
        best[2].score_parts["compatible_pool_size"] = int(len(pool))
        return best[2]

    scored[0][2].score_parts["score_after_energy_rerank"] = float(scored[0][0])
    return scored[0][2]


def plan_auto_keyframes(start_pose, end_pose, user_mid_poses, user_mid_frames, audio_feature, traj_physical, rag_db,
                        normalizer=None, num_frames=150, max_auto_keyframes=3, min_gap=18, rag_pose_space="normalized",
                        max_candidates=5000, sample_stride=3, music_weight=0.6, trajectory_weight=0.4,
                        mmr_checkpoint="", mmr_device="cpu", mmr_weight=0.0, pose_weight=1.0, diversity_weight=0.25,
                        energy_weight=0.45, energy_target=0.55, energy_band=0.25, contact_weight=0.85,
                        contact_diversity_weight=0.60, end_weight=0.30, source_gap=120, disallow_same_source=False,
                        energy_rerank_top_k=80, energy_rerank_weight=0.25, fps=30, **kwargs):
    effective_count = adaptive_auto_mid_count(num_frames, max_auto_keyframes, fps=fps)
    if effective_count <= 0:
        return AutoKeyframePlan([], [], {"planner_version": "choreo_unit_rag_v5", "effective_auto_mid_count": 0})

    candidates = load_rag_candidates(rag_db, normalizer=normalizer, pose_space=rag_pose_space, max_candidates=max_candidates, sample_stride=sample_stride)
    annotate_candidate_statistics(candidates)
    if not candidates:
        raise RuntimeError(f"No RAG candidates loaded from {rag_db}")

    plan = get_choreo_plan(audio_feature, num_frames)
    frames = choose_auto_frames(audio_feature, traj_physical, num_frames, effective_count, list(user_mid_frames or []),
                                min_gap=min_gap, music_weight=music_weight, trajectory_weight=trajectory_weight, fps=fps)

    anchors = [(0, np.asarray(start_pose, dtype=np.float32))]
    for f, p in zip(user_mid_frames or [], user_mid_poses or []):
        anchors.append((int(f), np.asarray(p, dtype=np.float32)))
    anchors.append((int(num_frames - 1), np.asarray(end_pose, dtype=np.float32)))
    anchors = sorted(anchors, key=lambda x: x[0])

    motion_unit_mode = os.environ.get("EDGE_MOTION_UNIT_MODE", "auto").strip().lower()
    has_units = any(bool(c.get("is_motion_unit", False)) for c in candidates)
    use_units = (motion_unit_mode == "on") or (motion_unit_mode == "auto" and has_units)
    if motion_unit_mode == "off":
        use_units = False

    text_weight = _env_float("EDGE_TEXT_BRIDGE_WEIGHT", 0.50)
    w_entry = _env_float("EDGE_UNIT_ENTRY_WEIGHT", 0.60) if use_units else 0.0
    w_exit = _env_float("EDGE_UNIT_EXIT_WEIGHT", 0.60) if use_units else 0.0
    w_phase = _env_float("EDGE_UNIT_CONTACT_PHASE_WEIGHT", 0.85) if use_units else 0.0

    selected, selected_poses = [], []
    for frame in frames:
        seg = segment_for_frame(plan, frame)
        query_text = str(seg.get("query_text", seg.get("motion_prompt", "")))
        text_embedding = encode_text_query(
            query_text,
            model_name=os.environ.get("EDGE_TEXT_BRIDGE_MODEL", "BAAI/bge-small-zh-v1.5"),
            device=os.environ.get("EDGE_TEXT_BRIDGE_DEVICE", "cpu"),
        ) if text_weight > 0 else None
        if text_embedding is None:
            text_weight_eff = 0.0
        else:
            text_weight_eff = text_weight

        dyn_anchors = sorted(anchors + [(kf.frame, kf.pose) for kf in selected], key=lambda x: x[0])
        kf = choose_candidate_for_frame(
            frame=frame,
            candidates=candidates,
            anchors=dyn_anchors,
            traj_physical=traj_physical,
            selected_poses=selected_poses,
            selected_keyframes=selected,
            text_embedding=text_embedding,
            segment=seg,
            w_text=text_weight_eff,
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
            w_contact_phase=w_phase,
            source_gap=source_gap,
            disallow_same_source=disallow_same_source,
            energy_rerank_top_k=energy_rerank_top_k,
            energy_rerank_weight=energy_rerank_weight,
        )
        selected.append(kf)
        selected_poses.append(kf.pose)

    meta = {
        "planner_version": "choreo_unit_rag_v5",
        "rag_db": str(rag_db),
        "candidate_count": int(len(candidates)),
        "has_motion_units": bool(has_units),
        "use_motion_units": bool(use_units),
        "effective_auto_mid_count": int(effective_count),
        "frame_candidates": [int(x) for x in frames],
        "choreo_plan": plan,
        "weights": {
            "text": float(text_weight),
            "entry": float(w_entry),
            "exit": float(w_exit),
            "contact_phase": float(w_phase),
            "pose": float(pose_weight),
            "energy": float(energy_weight),
            "contact": float(contact_weight),
        },
    }
    return AutoKeyframePlan(keyframes=selected, frame_candidates=frames, meta=meta)


def save_auto_keyframes(plan=None, keyframes=None, out_dir=None, output_dir=None, prefix="auto"):
    keyframes = list(keyframes if keyframes is not None else getattr(plan, "keyframes", []))
    out_dir = Path(out_dir or output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    segments = []
    for i, kf in enumerate(keyframes, start=1):
        pose_path = out_dir / f"{prefix}_auto_mid_{i:02d}.npy"
        np.save(pose_path, np.asarray(kf.pose, dtype=np.float32))
        rec = {
            "path": str(pose_path),
            "pose_path": str(pose_path),
            "frame": int(kf.frame),
            "source": str(kf.source),
            "source_frame": int(kf.source_frame),
            "unit_start": int(kf.unit_start),
            "unit_center": int(kf.unit_center),
            "unit_end": int(kf.unit_end),
            "segment_id": int(kf.segment_id),
            "segment_prompt": str(kf.segment_prompt),
            "motion_text": str(kf.motion_text),
            "score": float(kf.score),
            "score_parts": dict(kf.score_parts),
        }
        if kf.unit_motion is not None:
            prior_path = out_dir / f"{prefix}_unit_prior_{i:02d}.npy"
            np.save(prior_path, np.asarray(kf.unit_motion, dtype=np.float32))
            rec["unit_prior_path"] = str(prior_path)
        saved.append(rec)
        segments.append(rec)

    meta = {
        "planner_meta": {} if plan is None else getattr(plan, "meta", {}),
        "keyframes": segments,
    }
    meta_path = out_dir / f"{prefix}_choreorag_plan.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ ChoreoRAG auto mids saved: {meta_path}")
    return saved


def append_csv(path, row, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(row.keys())
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_pose(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        arr = d.get("pose", d.get("motion"))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    return arr.reshape(151).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_db", required=True)
    parser.add_argument("--start_pose", required=True)
    parser.add_argument("--end_pose", required=True)
    parser.add_argument("--audio_feature", required=True)
    parser.add_argument("--traj", required=True, help="[T,2] .npy")
    parser.add_argument("--out_dir", default="output/choreorag_test")
    parser.add_argument("--prefix", default="test")
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--auto_mid_count", type=int, default=1)
    parser.add_argument("--rag_pose_space", default="normalized")
    args = parser.parse_args()

    audio = np.load(args.audio_feature).astype(np.float32)
    traj = np.load(args.traj).astype(np.float32)
    plan = plan_auto_keyframes(
        start_pose=_load_pose(args.start_pose),
        end_pose=_load_pose(args.end_pose),
        user_mid_poses=[],
        user_mid_frames=[],
        audio_feature=audio,
        traj_physical=traj,
        rag_db=args.rag_db,
        num_frames=args.num_frames,
        max_auto_keyframes=args.auto_mid_count,
        rag_pose_space=args.rag_pose_space,
    )
    save_auto_keyframes(plan=plan, out_dir=args.out_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
