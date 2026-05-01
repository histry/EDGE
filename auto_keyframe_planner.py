"""Transition-aware auto middle-keyframe planner with optional MMR-RAG retrieval.

Drop-in replacement for the original auto_keyframe_planner.py.

What changed vs. the old pose-only planner:
- still exports plan_auto_keyframes(), save_auto_keyframes(), append_csv();
- keeps compatibility with generate_controlled.py and retrieved_clip_prior.py;
- adds transition/contact-phase aware scoring to reduce repeated one-leg hops;
- writes richer auto_mid_plan.json fields for later diagnosis.

The planner still outputs .npy middle keyframe poses, so downstream generation
and evaluation scripts do not need structural changes.
"""
from __future__ import annotations

import argparse
import json
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
except Exception:  # keeps CLI/import safe in partial environments
    ax_to_6v = None
    vectorize_many = None
    Normalizer = None
    SMPLSkeleton = None

try:
    from model.mmr_encoder import load_mmr_model
except Exception:
    load_mmr_model = None

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


@dataclass
class AutoKeyframePlan:
    keyframes: List[AutoKeyframe]
    frame_candidates: List[int]
    meta: Dict[str, object]


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_normalizer_from_checkpoint(checkpoint_path: str):
    if not checkpoint_path:
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    norm_data = checkpoint.get("normalizer") if isinstance(checkpoint, dict) else None
    if norm_data is None:
        return None
    if hasattr(norm_data, "mean") and hasattr(norm_data, "std"):
        return norm_data
    if isinstance(norm_data, dict) and "mean" in norm_data and "std" in norm_data:
        if Normalizer is None:
            class DummyNormalizer:
                pass
            normalizer = DummyNormalizer()
        else:
            normalizer = Normalizer(torch.zeros((1, 1, 151)))
        normalizer.mean = np.asarray(norm_data["mean"], dtype=np.float32)
        normalizer.std = np.asarray(norm_data["std"], dtype=np.float32)
        return normalizer
    return None


def normalize_motion_if_needed(motion: np.ndarray, normalizer, pose_space: str) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if pose_space == "normalized":
        return motion.astype(np.float32)
    if pose_space == "physical":
        if normalizer is None:
            raise ValueError("pose_space='physical' requires a checkpoint normalizer")
        mt = torch.from_numpy(motion).float()
        if mt.ndim == 1:
            out = normalizer.normalize(mt[None, None, :])
            return _to_numpy(out)[0, 0].astype(np.float32)
        if mt.ndim == 2:
            out = normalizer.normalize(mt[None])
            return _to_numpy(out)[0].astype(np.float32)
        if mt.ndim == 3:
            out = normalizer.normalize(mt)
            return _to_numpy(out).astype(np.float32)
    raise ValueError(f"Unsupported pose_space: {pose_space}")


def _pkl_to_motion_151(path: Path) -> Optional[np.ndarray]:
    if SMPLSkeleton is None or ax_to_6v is None or vectorize_many is None:
        return None
    data = pickle.load(open(path, "rb"))
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


def _safe_float(v, default=0.0) -> float:
    try:
        out = float(v)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    return out


def load_rag_candidates(
    rag_db: str,
    normalizer=None,
    pose_space: str = "normalized",
    max_candidates: int = 5000,
    sample_stride: int = 1,
) -> List[Dict[str, object]]:
    """Load pose-level or MMR clip-level RAG candidates.

    The resulting candidate dict intentionally keeps the old fields used by the
    previous planner and adds segment fields when available. For flat .npz RAG
    indices, segment_start/end are approximate but useful for debugging and
    downstream prior construction.
    """
    path = Path(rag_db)
    candidates: List[Dict[str, object]] = []
    stride = max(1, int(sample_stride))

    if path.is_file() and path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "poses" not in data.files:
            raise ValueError(f"{path} has no 'poses' field")
        poses = normalize_motion_if_needed(np.asarray(data["poses"], dtype=np.float32), normalizer, pose_space)
        source = data["source"] if "source" in data.files else np.array(["rag_index"] * len(poses))
        source_frame = data["source_frame"] if "source_frame" in data.files else np.arange(len(poses))
        root_vel = data["root_vel"] if "root_vel" in data.files else np.zeros((len(poses), 2), dtype=np.float32)
        motion_embedding = None
        if "motion_embedding" in data.files:
            motion_embedding = data["motion_embedding"]
        elif "motion_embeddings" in data.files:
            motion_embedding = data["motion_embeddings"]
        motion_energy = data["motion_energy"] if "motion_energy" in data.files else None
        contact_stability = data["contact_stability"] if "contact_stability" in data.files else None

        for i in range(0, len(poses), stride):
            sf = int(source_frame[i])
            candidates.append({
                "pose": poses[i].astype(np.float32),
                "source": str(source[i]),
                "source_frame": sf,
                "source_segment_start": max(0, sf - stride),
                "source_segment_end": sf + stride,
                "root_vel": np.asarray(root_vel[i], dtype=np.float32),
                "motion_embedding": None if motion_embedding is None else np.asarray(motion_embedding[i], dtype=np.float32),
                "motion_energy": None if motion_energy is None else float(motion_energy[i]),
                "contact_stability": None if contact_stability is None else float(contact_stability[i]),
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
                candidates.extend(load_rag_candidates(str(file), normalizer, pose_space, max_candidates - len(candidates), stride))
                continue

            motion = _pkl_to_motion_151(file) if file.suffix == ".pkl" else _load_npy_motion(file)
            if motion is None:
                continue
            motion = normalize_motion_if_needed(motion, normalizer, pose_space)
            root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
            root_vel = np.zeros_like(root)
            if len(root) > 1:
                root_vel[1:] = root[1:] - root[:-1]
            pose_vel = np.zeros((len(motion),), dtype=np.float32)
            if len(motion) > 1:
                pose_vel[1:] = np.sqrt(np.mean((pose_feature_many(motion[1:]) - pose_feature_many(motion[:-1])) ** 2, axis=1))
            contacts = (motion[:, CONTACT_SLICE] > 0.5).astype(np.float32)
            contact_stability = 1.0 - np.clip(np.mean(np.abs(np.diff(contacts, axis=0)), axis=1, keepdims=False), 0.0, 1.0) if len(motion) > 1 else np.ones((1,), dtype=np.float32)
            contact_stability = np.r_[contact_stability[:1], contact_stability] if len(contact_stability) < len(motion) else contact_stability
            for i in range(0, len(motion), stride):
                candidates.append({
                    "pose": motion[i].astype(np.float32),
                    "source": str(file),
                    "source_frame": int(i),
                    "source_segment_start": max(0, int(i) - stride),
                    "source_segment_end": min(len(motion) - 1, int(i) + stride),
                    "root_vel": root_vel[i].astype(np.float32),
                    "motion_embedding": None,
                    "motion_energy": float(pose_vel[i]),
                    "contact_stability": float(contact_stability[i]),
                })
                if len(candidates) >= max_candidates:
                    break
        return candidates[:max_candidates]

    raise ValueError(f"Invalid rag_db: {rag_db}")


def annotate_candidate_statistics(candidates: List[Dict[str, object]]):
    """Attach robust normalized motion-energy stats to candidates."""
    if not candidates:
        return
    energies = np.asarray([max(0.0, _safe_float(c.get("motion_energy", 0.0))) for c in candidates], dtype=np.float32)
    if float(energies.max() - energies.min()) > 1e-8:
        lo, hi = np.percentile(energies, [10, 90])
        if float(hi - lo) <= 1e-8:
            lo, hi = float(energies.min()), float(energies.max())
        e_norm = np.clip((energies - float(lo)) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    else:
        e_norm = np.zeros_like(energies, dtype=np.float32)
    for c, en in zip(candidates, e_norm):
        c["motion_energy_norm"] = float(en)


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - float(x.min())
    denom = float(x.max())
    if denom <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x / denom).astype(np.float32)


def smooth_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    window = max(1, int(window))
    if window <= 1 or len(x) < 3:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)


def audio_onset_score(audio_feature: np.ndarray, onset_index: int = 768) -> np.ndarray:
    audio_feature = np.asarray(audio_feature, dtype=np.float32)
    if audio_feature.ndim != 2 or len(audio_feature) == 0:
        return np.zeros((0,), dtype=np.float32)
    if audio_feature.shape[1] > onset_index:
        onset = np.maximum(audio_feature[:, onset_index], 0.0)
    else:
        onset = np.zeros((len(audio_feature),), dtype=np.float32)
        if len(audio_feature) > 1:
            onset[1:] = np.linalg.norm(audio_feature[1:] - audio_feature[:-1], axis=1)
            onset[0] = onset[1]
    return normalize_01(smooth_1d(onset, 5))


def trajectory_curvature_score(traj_physical: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj_physical, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    traj = traj[:, :2]
    if len(traj) < 3:
        return np.zeros((len(traj),), dtype=np.float32)
    v1 = traj[1:-1] - traj[:-2]
    v2 = traj[2:] - traj[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-8, None)
    out = np.zeros((len(traj),), dtype=np.float32)
    out[1:-1] = 1.0 - np.clip(cos, -1.0, 1.0)
    return normalize_01(smooth_1d(out, 5))


def choose_auto_frames(audio_feature, traj_physical, num_frames, count, existing_frames, min_gap=18, edge_margin=8, music_weight=0.6, trajectory_weight=0.4) -> List[int]:
    if count <= 0:
        return []
    onset = audio_onset_score(audio_feature)
    if len(onset) != num_frames:
        onset = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, max(len(onset), 1)), onset if len(onset) else [0.0]).astype(np.float32)
    curvature = trajectory_curvature_score(traj_physical)
    if len(curvature) != num_frames:
        curvature = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, max(len(curvature), 1)), curvature if len(curvature) else [0.0]).astype(np.float32)
    score = float(music_weight) * normalize_01(onset) + float(trajectory_weight) * normalize_01(curvature)
    blocked = np.zeros((num_frames,), dtype=bool)
    for frame in list(existing_frames) + [0, num_frames - 1]:
        blocked[max(0, int(frame) - min_gap): min(num_frames, int(frame) + min_gap + 1)] = True
    blocked[:edge_margin] = True
    blocked[num_frames - edge_margin:] = True
    frames = []
    for _ in range(int(count)):
        s = score.copy()
        s[blocked] = -1e9
        frame = int(s.argmax())
        if s[frame] < -1e8:
            break
        frames.append(frame)
        blocked[max(0, frame - min_gap): min(num_frames, frame + min_gap + 1)] = True
    if len(frames) < count:
        grid = [int(round((i + 1) * (num_frames - 1) / (count + 1))) for i in range(count)]
        for frame in grid:
            if len(frames) >= count:
                break
            if all(abs(frame - f) >= min_gap for f in frames + list(existing_frames) + [0, num_frames - 1]):
                frames.append(max(edge_margin, min(num_frames - edge_margin - 1, frame)))
    return sorted(set(frames))[:count]


def pose_feature(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float32).reshape(-1)[POSE_FEATURE_INDEX].astype(np.float32)


def pose_feature_many(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    return motion[:, POSE_FEATURE_INDEX].astype(np.float32)


def pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pose_feature(a) - pose_feature(b)) ** 2)))


def interpolate_pose_feature_at_frame(frame: int, anchors: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    anchors = sorted([(int(f), np.asarray(p, dtype=np.float32)) for f, p in anchors], key=lambda x: x[0])
    if frame <= anchors[0][0]:
        return pose_feature(anchors[0][1])
    if frame >= anchors[-1][0]:
        return pose_feature(anchors[-1][1])
    for (f0, p0), (f1, p1) in zip(anchors[:-1], anchors[1:]):
        if f0 <= frame <= f1:
            alpha = (frame - f0) / max(float(f1 - f0), 1.0)
            return (1.0 - alpha) * pose_feature(p0) + alpha * pose_feature(p1)
    return pose_feature(anchors[-1][1])


def bracketing_anchors(frame: int, anchors: List[Tuple[int, np.ndarray]]) -> Tuple[Tuple[int, np.ndarray], Tuple[int, np.ndarray]]:
    anchors = sorted([(int(f), np.asarray(p, dtype=np.float32)) for f, p in anchors], key=lambda x: x[0])
    prev_a = anchors[0]
    next_a = anchors[-1]
    for a in anchors:
        if a[0] <= frame:
            prev_a = a
        if a[0] >= frame:
            next_a = a
            break
    if prev_a[0] == frame and len(anchors) > 1:
        idx = anchors.index(prev_a)
        prev_a = anchors[max(0, idx - 1)]
    if next_a[0] == frame and len(anchors) > 1:
        idx = anchors.index(next_a)
        next_a = anchors[min(len(anchors) - 1, idx + 1)]
    return prev_a, next_a


def _traj_at(traj_physical: np.ndarray, frame: int) -> np.ndarray:
    traj = np.asarray(traj_physical, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    traj = traj[:, :2]
    frame = max(0, min(len(traj) - 1, int(frame)))
    return traj[frame]


def target_traj_tangent(traj_physical: np.ndarray, frame: int) -> np.ndarray:
    traj = np.asarray(traj_physical, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    traj = traj[:, :2]
    if len(traj) < 2:
        return np.zeros((2,), dtype=np.float32)
    f0 = max(0, frame - 1)
    f1 = min(len(traj) - 1, frame + 1)
    v = traj[f1] - traj[f0]
    n = np.linalg.norm(v)
    return np.zeros((2,), dtype=np.float32) if n <= 1e-8 else (v / n).astype(np.float32)


def direction_cost(candidate_vel, target_tangent) -> float:
    if candidate_vel is None:
        return 0.5
    v = np.asarray(candidate_vel, dtype=np.float32).reshape(-1)[:2]
    t = np.asarray(target_tangent, dtype=np.float32).reshape(-1)[:2]
    nv = np.linalg.norm(v)
    nt = np.linalg.norm(t)
    if nv <= 1e-8 or nt <= 1e-8:
        return 0.5
    cos = float(np.dot(v, t) / (nv * nt))
    return 0.5 * (1.0 - np.clip(cos, -1.0, 1.0))


def velocity_compatibility_cost(candidate_vel, traj_physical, prev_frame: int, frame: int, next_frame: int) -> float:
    if candidate_vel is None:
        return 0.5
    v = np.asarray(candidate_vel, dtype=np.float32).reshape(-1)[:2]
    speed = float(np.linalg.norm(v))
    prev_gap = max(1, int(frame) - int(prev_frame))
    next_gap = max(1, int(next_frame) - int(frame))
    v_in = (_traj_at(traj_physical, frame) - _traj_at(traj_physical, prev_frame)) / float(prev_gap)
    v_out = (_traj_at(traj_physical, next_frame) - _traj_at(traj_physical, frame)) / float(next_gap)
    expected = 0.5 * (v_in + v_out)
    exp_speed = float(np.linalg.norm(expected))
    if exp_speed <= 1e-8 and speed <= 1e-8:
        return 0.0
    dir_c = direction_cost(v, expected)
    speed_c = abs(speed - exp_speed) / max(exp_speed + speed, 1e-6)
    return float(np.clip(0.65 * dir_c + 0.35 * speed_c, 0.0, 2.0))


def encode_audio_with_mmr(audio_feature: np.ndarray, mmr_checkpoint: str, device: str = "cpu") -> Optional[np.ndarray]:
    if not mmr_checkpoint:
        return None
    if load_mmr_model is None:
        raise ImportError("MMR retrieval requires model/mmr_encoder.py")
    model = load_mmr_model(mmr_checkpoint, device=device)
    audio = torch.from_numpy(np.asarray(audio_feature, dtype=np.float32)[None]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        z = model.encode_audio(audio)[0].detach().cpu().numpy().astype(np.float32)
    n = np.linalg.norm(z)
    return z / n if n > 1e-8 else z


def cosine_distance(a, b) -> float:
    if a is None or b is None:
        return 0.5
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if len(a) != len(b):
        return 0.5
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-8 or nb <= 1e-8:
        return 0.5
    return 0.5 * (1.0 - np.clip(float(np.dot(a, b) / (na * nb)), -1.0, 1.0))


def is_same_source_region(candidate, selected_keyframes, source_gap: int = 120, disallow_same_source: bool = False) -> Tuple[bool, str]:
    src = str(candidate.get("source", ""))
    sf = int(candidate.get("source_frame", -1))
    gap = max(0, int(source_gap))
    for prev in selected_keyframes:
        prev_src = str(prev.source)
        prev_sf = int(prev.source_frame)
        if not src or not prev_src or src != prev_src:
            continue
        if bool(disallow_same_source):
            return True, f"same_source:{src}"
        if sf >= 0 and prev_sf >= 0 and abs(sf - prev_sf) < gap:
            return True, f"same_source_region:{src}:{sf}~{prev_sf}:gap<{gap}"
    return False, ""


def contact_pattern(pose: np.ndarray) -> np.ndarray:
    c = np.asarray(pose, dtype=np.float32).reshape(-1)[:4]
    return (c > 0.5).astype(np.float32)


def contact_phase(pose: np.ndarray) -> str:
    c = contact_pattern(pose)
    left = bool(c[0] or c[2])
    right = bool(c[1] or c[3])
    if left and right:
        return "both"
    if left:
        return "left_only"
    if right:
        return "right_only"
    return "none"


def contact_diversity_cost(pose: np.ndarray, selected_poses: Sequence[np.ndarray]) -> float:
    if not selected_poses:
        return 0.0
    cp = contact_pattern(pose)
    sims = [float((cp == contact_pattern(prev)).mean()) for prev in selected_poses]
    return max(sims) if sims else 0.0


def contact_phase_continuity_cost(pose: np.ndarray, selected_poses: Sequence[np.ndarray]) -> float:
    """Extra penalty for repeated one-foot support, the main visual hop failure."""
    if not selected_poses:
        return 0.0
    phase = contact_phase(pose)
    prev_phase = contact_phase(selected_poses[-1])
    if phase == prev_phase and phase in {"left_only", "right_only", "none"}:
        return 1.0
    if phase == prev_phase:
        return 0.35
    return 0.0


def pose_direction_cost(prev_pose: np.ndarray, candidate_pose: np.ndarray, next_pose: np.ndarray) -> float:
    prev_f = pose_feature(prev_pose)
    cand_f = pose_feature(candidate_pose)
    next_f = pose_feature(next_pose)
    candidate_dir = cand_f - prev_f
    global_dir = next_f - prev_f
    return cosine_distance(candidate_dir, global_dir)


def transition_cost(prev_pose: np.ndarray, pose: np.ndarray, next_pose: np.ndarray, prev_gap: int, next_gap: int) -> Tuple[float, float, float]:
    """Return in, out, and balanced transition cost.

    Costs are divided by sqrt(frame gap), because a larger temporal interval can
    naturally support a larger pose change. This makes the planner less likely
    to reject expressive but connectable poses while still penalizing snapping.
    """
    in_c = pose_distance(prev_pose, pose) / max(np.sqrt(max(1, prev_gap)), 1.0)
    out_c = pose_distance(pose, next_pose) / max(np.sqrt(max(1, next_gap)), 1.0)
    balance = abs(in_c - out_c) / max(in_c + out_c, 1e-6)
    return float(in_c), float(out_c), float(balance)


def choose_candidate_for_frame(
    frame: int,
    candidates: List[Dict[str, object]],
    anchors: List[Tuple[int, np.ndarray]],
    traj_physical: np.ndarray,
    selected_poses: List[np.ndarray],
    selected_keyframes: Optional[List[AutoKeyframe]] = None,
    audio_embedding: Optional[np.ndarray] = None,
    w_mmr: float = 0.0,
    w_pose: float = 1.0,
    w_traj: float = 0.25,
    w_diversity: float = 0.15,
    w_energy: float = 0.25,
    energy_target: float = 0.55,
    energy_band: float = 0.25,
    w_contact: float = 0.50,
    w_contact_diversity: float = 0.30,
    w_end: float = 0.25,
    source_gap: int = 120,
    disallow_same_source: bool = False,
    energy_rerank_top_k: int = 80,
    energy_rerank_weight: float = 0.25,
    w_transition: float = 0.75,
    w_velocity: float = 0.35,
    w_pose_direction: float = 0.25,
    w_contact_phase: float = 0.45,
) -> AutoKeyframe:
    ref_feat = interpolate_pose_feature_at_frame(frame, anchors)
    tangent = target_traj_tangent(traj_physical, frame)
    prev_anchor, next_anchor = bracketing_anchors(frame, anchors)
    prev_frame, prev_pose = prev_anchor
    next_frame, next_pose = next_anchor
    prev_gap = max(1, int(frame) - int(prev_frame))
    next_gap = max(1, int(next_frame) - int(frame))

    selected_keyframes = selected_keyframes or []
    rejected_by_source = 0
    scored = []

    for cand in candidates:
        blocked, _reason = is_same_source_region(cand, selected_keyframes, source_gap=source_gap, disallow_same_source=disallow_same_source)
        if blocked:
            rejected_by_source += 1
            continue

        pose = np.asarray(cand["pose"], dtype=np.float32)
        pfeat = pose_feature(pose)
        pose_cost = float(np.sqrt(np.mean((pfeat - ref_feat) ** 2)))
        traj_cost = direction_cost(cand.get("root_vel", None), tangent)
        diversity_cost = 0.0 if not selected_poses else 1.0 / (1.0 + min(pose_distance(pose, p) for p in selected_poses))
        mmr_cost = cosine_distance(audio_embedding, cand.get("motion_embedding", None))

        energy_norm = float(np.clip(_safe_float(cand.get("motion_energy_norm", 0.0)), 0.0, 1.0))
        e_target = float(np.clip(energy_target, 0.0, 1.0))
        e_band = max(float(energy_band), 1e-6)
        energy_cost = float(min(abs(energy_norm - e_target) / e_band, 2.0))

        contact_stability = cand.get("contact_stability", None)
        contact_cost = 0.25 if contact_stability is None else 1.0 - float(np.clip(_safe_float(contact_stability, 0.5), 0.0, 1.0))
        contact_div_cost = contact_diversity_cost(pose, selected_poses)
        phase_cost = contact_phase_continuity_cost(pose, selected_poses)
        in_cost, out_cost, balance_cost = transition_cost(prev_pose, pose, next_pose, prev_gap, next_gap)
        transition_total = in_cost + out_cost + 0.5 * balance_cost
        vel_cost = velocity_compatibility_cost(cand.get("root_vel", None), traj_physical, prev_frame, frame, next_frame)
        dir_cost = pose_direction_cost(prev_pose, pose, next_pose)
        end_alpha = float(frame) / max(float(anchors[-1][0]), 1.0)
        end_compat_cost = end_alpha * pose_distance(pose, anchors[-1][1])

        base_score = (
            float(w_mmr) * mmr_cost
            + float(w_pose) * pose_cost
            + float(w_traj) * traj_cost
            + float(w_diversity) * diversity_cost
            + float(w_energy) * energy_cost
            + float(w_contact) * contact_cost
            + float(w_contact_diversity) * contact_div_cost
            + float(w_end) * end_compat_cost
            + float(w_transition) * transition_total
            + float(w_velocity) * vel_cost
            + float(w_pose_direction) * dir_cost
            + float(w_contact_phase) * phase_cost
        )

        score_parts = {
            "base_score_before_energy_rerank": float(base_score),
            "mmr_cost": float(mmr_cost),
            "pose_cost": float(pose_cost),
            "trajectory_direction_cost": float(traj_cost),
            "diversity_cost": float(diversity_cost),
            "motion_energy_cost": float(energy_cost),
            "motion_energy_norm": float(energy_norm),
            "motion_energy_target": float(e_target),
            "motion_energy_band": float(e_band),
            "contact_stability_cost": float(contact_cost),
            "contact_diversity_cost": float(contact_div_cost),
            "contact_phase_continuity_cost": float(phase_cost),
            "transition_in_cost": float(in_cost),
            "transition_out_cost": float(out_cost),
            "transition_balance_cost": float(balance_cost),
            "transition_total_cost": float(transition_total),
            "transition_velocity_cost": float(vel_cost),
            "pose_direction_cost": float(dir_cost),
            "end_compat_cost": float(end_compat_cost),
            "contact_phase": contact_phase(pose),
            "segment_energy": float(energy_norm),
            "segment_velocity": float(np.linalg.norm(np.asarray(cand.get("root_vel", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2])),
            "source_segment_start": int(cand.get("source_segment_start", cand.get("source_frame", -1))),
            "source_segment_end": int(cand.get("source_segment_end", cand.get("source_frame", -1))),
            "prev_anchor_frame": int(prev_frame),
            "next_anchor_frame": int(next_frame),
            "rejected_by_source_before_pool": int(rejected_by_source),
        }

        kf = AutoKeyframe(
            frame=int(frame),
            pose=pose.astype(np.float32),
            score=float(base_score),
            source=str(cand.get("source", "")),
            source_frame=int(cand.get("source_frame", -1)),
            score_parts=score_parts,
        )
        scored.append((float(base_score), float(energy_norm), kf))

    if not scored:
        if selected_keyframes and (source_gap > 0 or disallow_same_source):
            return choose_candidate_for_frame(
                frame=frame,
                candidates=candidates,
                anchors=anchors,
                traj_physical=traj_physical,
                selected_poses=selected_poses,
                selected_keyframes=[],
                audio_embedding=audio_embedding,
                w_mmr=w_mmr,
                w_pose=w_pose,
                w_traj=w_traj,
                w_diversity=w_diversity,
                w_energy=w_energy,
                energy_target=energy_target,
                energy_band=energy_band,
                w_contact=w_contact,
                w_contact_diversity=w_contact_diversity,
                w_end=w_end,
                source_gap=0,
                disallow_same_source=False,
                energy_rerank_top_k=energy_rerank_top_k,
                energy_rerank_weight=energy_rerank_weight,
                w_transition=w_transition,
                w_velocity=w_velocity,
                w_pose_direction=w_pose_direction,
                w_contact_phase=w_contact_phase,
            )
        raise RuntimeError("No RAG candidate available")

    scored.sort(key=lambda x: x[0])
    top_k = int(energy_rerank_top_k)
    if top_k > 0 and float(energy_rerank_weight) > 0:
        pool = scored[: min(top_k, len(scored))]

        def rerank_key(item):
            base_score, energy_norm_value, _kf = item
            target_cost = float(min(abs(float(energy_norm_value) - e_target) / e_band, 2.0))
            return float(base_score) + float(energy_rerank_weight) * target_cost

        best_base, best_energy_norm, best_kf = min(pool, key=rerank_key)
        energy_target_cost = float(min(abs(float(best_energy_norm) - e_target) / e_band, 2.0))
        final_score = float(best_base + float(energy_rerank_weight) * energy_target_cost)
        best_kf.score = final_score
        best_kf.score_parts["energy_rerank_top_k"] = int(top_k)
        best_kf.score_parts["energy_rerank_weight"] = float(energy_rerank_weight)
        best_kf.score_parts["energy_rerank_target_cost"] = float(energy_target_cost)
        best_kf.score_parts["energy_rerank_penalty"] = float(float(energy_rerank_weight) * energy_target_cost)
        best_kf.score_parts["score_after_energy_rerank"] = final_score
        best_kf.score_parts["compatible_pool_size"] = int(len(pool))
        return best_kf

    best_kf = scored[0][2]
    best_kf.score_parts["energy_rerank_top_k"] = 0
    best_kf.score_parts["energy_rerank_weight"] = 0.0
    best_kf.score_parts["score_after_energy_rerank"] = float(best_kf.score)
    return best_kf


def plan_auto_keyframes(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    user_mid_poses: Sequence[np.ndarray],
    user_mid_frames: Sequence[int],
    audio_feature: np.ndarray,
    traj_physical: np.ndarray,
    rag_db: str,
    normalizer=None,
    num_frames: int = 150,
    max_auto_keyframes: int = 3,
    min_gap: int = 18,
    rag_pose_space: str = "normalized",
    max_candidates: int = 5000,
    sample_stride: int = 3,
    music_weight: float = 0.6,
    trajectory_weight: float = 0.4,
    mmr_checkpoint: str = "",
    mmr_device: str = "cpu",
    mmr_weight: float = 0.0,
    pose_weight: float = 1.0,
    diversity_weight: float = 0.25,
    energy_weight: float = 0.60,
    energy_target: float = 0.55,
    energy_band: float = 0.25,
    contact_weight: float = 0.50,
    contact_diversity_weight: float = 0.30,
    end_weight: float = 0.25,
    source_gap: int = 120,
    disallow_same_source: bool = False,
    energy_rerank_top_k: int = 80,
    energy_rerank_weight: float = 0.25,
    # New Planner-v3 weights. generate_controlled.py can ignore these safely.
    transition_weight: float = 0.75,
    transition_velocity_weight: float = 0.35,
    pose_direction_weight: float = 0.25,
    contact_phase_weight: float = 0.45,
) -> AutoKeyframePlan:
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

    anchors = [(0, np.asarray(start_pose, dtype=np.float32)), (num_frames - 1, np.asarray(end_pose, dtype=np.float32))]
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
    audio_embedding = encode_audio_with_mmr(audio_feature, mmr_checkpoint, mmr_device) if mmr_checkpoint else None
    effective_mmr_weight = float(mmr_weight) if audio_embedding is not None else 0.0

    selected: List[AutoKeyframe] = []
    selected_poses: List[np.ndarray] = []
    for frame in frames:
        # Include already selected auto keyframes as temporary anchors, so later
        # choices are scored against the actual planned transition chain.
        dynamic_anchors = anchors + [(k.frame, k.pose) for k in selected]
        kf = choose_candidate_for_frame(
            frame=frame,
            candidates=candidates,
            anchors=dynamic_anchors,
            traj_physical=traj_physical,
            selected_poses=selected_poses,
            selected_keyframes=selected,
            audio_embedding=audio_embedding,
            w_mmr=effective_mmr_weight,
            w_pose=pose_weight,
            w_traj=trajectory_weight,
            w_diversity=diversity_weight,
            w_energy=energy_weight,
            energy_target=energy_target,
            energy_band=energy_band,
            w_contact=contact_weight,
            w_contact_diversity=contact_diversity_weight,
            w_end=end_weight,
            source_gap=source_gap,
            disallow_same_source=disallow_same_source,
            energy_rerank_top_k=energy_rerank_top_k,
            energy_rerank_weight=energy_rerank_weight,
            w_transition=transition_weight,
            w_velocity=transition_velocity_weight,
            w_pose_direction=pose_direction_weight,
            w_contact_phase=contact_phase_weight,
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
            "max_auto_keyframes": int(max_auto_keyframes),
            "min_gap": int(min_gap),
            "music_weight": float(music_weight),
            "trajectory_weight": float(trajectory_weight),
            "rag_pose_space": rag_pose_space,
            "mmr_checkpoint": str(mmr_checkpoint),
            "mmr_weight": float(effective_mmr_weight),
            "pose_weight": float(pose_weight),
            "diversity_weight": float(diversity_weight),
            "energy_weight": float(energy_weight),
            "contact_weight": float(contact_weight),
            "contact_diversity_weight": float(contact_diversity_weight),
            "end_weight": float(end_weight),
            "transition_weight": float(transition_weight),
            "transition_velocity_weight": float(transition_velocity_weight),
            "pose_direction_weight": float(pose_direction_weight),
            "contact_phase_weight": float(contact_phase_weight),
            "source_gap": int(source_gap),
            "disallow_same_source": bool(disallow_same_source),
            "energy_rerank_top_k": int(energy_rerank_top_k),
            "energy_rerank_weight": float(energy_rerank_weight),
            "retrieval_mode": "mmr" if effective_mmr_weight > 0 else "proxy",
            "planner_version": "v3_transition_contact_segment_aware",
        },
    )


def save_auto_keyframes(plan: AutoKeyframePlan, out_motion_path: str, prefix: str = "auto_mid"):
    out_motion_path = Path(out_motion_path)
    out_dir = out_motion_path.parent
    stem = out_motion_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, frames, rows = [], [], []
    for idx, kf in enumerate(plan.keyframes, start=1):
        path = out_dir / f"{stem}_{prefix}{idx}_f{kf.frame:03d}.npy"
        np.save(path, kf.pose.astype(np.float32))
        paths.append(str(path))
        frames.append(int(kf.frame))
        rows.append({
            "index": idx,
            "frame": int(kf.frame),
            "pose_path": str(path),
            "score": float(kf.score),
            "source": kf.source,
            "source_frame": int(kf.source_frame),
            "source_segment_start": int(kf.score_parts.get("source_segment_start", kf.source_frame)),
            "source_segment_end": int(kf.score_parts.get("source_segment_end", kf.source_frame)),
            "contact_phase": str(kf.score_parts.get("contact_phase", contact_phase(kf.pose))),
            "transition_in_cost": float(kf.score_parts.get("transition_in_cost", 0.0)),
            "transition_out_cost": float(kf.score_parts.get("transition_out_cost", 0.0)),
            "segment_energy": float(kf.score_parts.get("segment_energy", 0.0)),
            "segment_velocity": float(kf.score_parts.get("segment_velocity", 0.0)),
            "score_parts": kf.score_parts,
        })
    meta_path = out_dir / f"{stem}_{prefix}_plan.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"auto_keyframes": rows, "frame_candidates": plan.frame_candidates, "planner_meta": plan.meta}, f, ensure_ascii=False, indent=2)
    return paths, frames, str(meta_path)


def append_csv(existing: str, values: Sequence[object]) -> str:
    old = [x.strip() for x in str(existing or "").replace(";", ",").split(",") if x.strip()]
    return ",".join(old + [str(v) for v in values])


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_db", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pose_space", default="normalized", choices=["normalized", "physical"])
    parser.add_argument("--max_candidates", type=int, default=5000)
    parser.add_argument("--sample_stride", type=int, default=3)
    args = parser.parse_args()
    normalizer = load_normalizer_from_checkpoint(args.checkpoint)
    cands = load_rag_candidates(args.rag_db, normalizer, args.pose_space, args.max_candidates, args.sample_stride)
    annotate_candidate_statistics(cands)
    print(f"loaded candidates: {len(cands)}")
    if cands:
        print({k: v for k, v in cands[0].items() if k != "pose"})
        print("pose shape:", cands[0]["pose"].shape)


if __name__ == "__main__":
    _cli()
