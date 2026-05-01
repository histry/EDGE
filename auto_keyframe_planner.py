import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from dataset.quaternion import ax_to_6v
    from dataset.preprocess import vectorize_many, Normalizer
    from vis import SMPLSkeleton
except Exception:
    ax_to_6v = None
    vectorize_many = None
    Normalizer = None
    SMPLSkeleton = None


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
            raise ValueError("pose_space='physical' requires a checkpoint normalizer.")
        motion_t = torch.from_numpy(motion).float()
        if motion_t.ndim == 2:
            motion_t = motion_t.unsqueeze(0)
            out = normalizer.normalize(motion_t)
            return _to_numpy(out)[0].astype(np.float32)
        if motion_t.ndim == 3:
            out = normalizer.normalize(motion_t)
            return _to_numpy(out).astype(np.float32)
        if motion_t.ndim == 1:
            out = normalizer.normalize(motion_t[None, None])
            return _to_numpy(out)[0, 0].astype(np.float32)

    raise ValueError(f"Unsupported pose_space: {pose_space}")


def _load_npy_records(path: Path, normalizer, pose_space: str):
    arr = np.load(path, allow_pickle=True)

    source = str(path)
    records = []

    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()

        motion = None
        for key in ("motion", "motion_151", "poses", "pose_seq", "pose"):
            if key in data:
                motion = data[key]
                break

        if motion is None:
            return []

        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim == 1:
            motion = motion.reshape(1, -1)
        if motion.ndim != 2 or motion.shape[1] != 151:
            return []

        motion = normalize_motion_if_needed(motion, normalizer, pose_space)

        audio_summary = data.get("audio_summary", None)
        music_embedding = data.get("music_embedding", data.get("audio_embedding", None))
        motion_embedding = data.get("motion_embedding", None)

        for i, pose in enumerate(motion):
            records.append(
                {
                    "pose": pose.astype(np.float32),
                    "source": source,
                    "source_frame": int(data.get("start_frame", 0)) + i,
                    "audio_summary": audio_summary,
                    "music_embedding": music_embedding,
                    "motion_embedding": motion_embedding,
                    "root_vel": None,
                }
            )
        return records

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 151:
        arr = arr.reshape(1, 151)
    if arr.ndim == 2 and arr.shape[1] == 151:
        arr = normalize_motion_if_needed(arr, normalizer, pose_space)
        root_xz = arr[:, [ROOT_X_IDX, ROOT_Z_IDX]]
        root_vel = np.zeros_like(root_xz)
        if len(root_xz) > 1:
            root_vel[1:] = root_xz[1:] - root_xz[:-1]
        for i, pose in enumerate(arr):
            records.append(
                {
                    "pose": pose.astype(np.float32),
                    "source": source,
                    "source_frame": i,
                    "audio_summary": None,
                    "music_embedding": None,
                    "motion_embedding": None,
                    "root_vel": root_vel[i].astype(np.float32),
                }
            )
    return records


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


def load_rag_candidates(
    rag_db: str,
    normalizer=None,
    pose_space: str = "normalized",
    max_candidates: int = 5000,
    sample_stride: int = 1,
) -> List[Dict[str, object]]:
    """
    Load a lightweight RAG pose database.

    Supported formats:
    1. rag_index.npz with fields:
       - poses: [N,151]
       - source: optional [N]
       - source_frame: optional [N]
       - root_vel: optional [N,2]
    2. Folder of .npy records. Each can be [T,151], [151], or dict with motion/poses.
    3. Folder of .pkl raw Dunhuang clips with keys pos/q.
    """
    path = Path(rag_db)
    candidates = []

    if path.is_file() and path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        poses = np.asarray(data["poses"], dtype=np.float32)
        poses = normalize_motion_if_needed(poses, normalizer, pose_space)
        source = data["source"] if "source" in data.files else np.array(["rag_index"] * len(poses))
        source_frame = data["source_frame"] if "source_frame" in data.files else np.arange(len(poses))
        root_vel = data["root_vel"] if "root_vel" in data.files else np.zeros((len(poses), 2), dtype=np.float32)
        music_embedding = data["music_embedding"] if "music_embedding" in data.files else None
        motion_embedding = data["motion_embedding"] if "motion_embedding" in data.files else None

        for i in range(0, len(poses), max(1, int(sample_stride))):
            candidates.append(
                {
                    "pose": poses[i].astype(np.float32),
                    "source": str(source[i]),
                    "source_frame": int(source_frame[i]),
                    "root_vel": np.asarray(root_vel[i], dtype=np.float32),
                    "music_embedding": None if music_embedding is None else np.asarray(music_embedding[i], dtype=np.float32),
                    "motion_embedding": None if motion_embedding is None else np.asarray(motion_embedding[i], dtype=np.float32),
                    "audio_summary": None,
                }
            )
        return candidates[:max_candidates]

    if path.is_dir():
        files = sorted(list(path.glob("*.npy")) + list(path.glob("*.npz")) + list(path.glob("*.pkl")))
        for file in files:
            if len(candidates) >= max_candidates:
                break
            if file.suffix == ".npz":
                candidates.extend(
                    load_rag_candidates(
                        str(file),
                        normalizer=normalizer,
                        pose_space=pose_space,
                        max_candidates=max_candidates - len(candidates),
                        sample_stride=sample_stride,
                    )
                )
            elif file.suffix == ".npy":
                records = _load_npy_records(file, normalizer=normalizer, pose_space=pose_space)
                candidates.extend(records[:: max(1, int(sample_stride))])
            elif file.suffix == ".pkl":
                motion = _pkl_to_motion_151(file)
                if motion is None:
                    continue
                motion = normalize_motion_if_needed(motion, normalizer, pose_space)
                root_xz = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
                root_vel = np.zeros_like(root_xz)
                if len(root_xz) > 1:
                    root_vel[1:] = root_xz[1:] - root_xz[:-1]
                for i in range(0, len(motion), max(1, int(sample_stride))):
                    candidates.append(
                        {
                            "pose": motion[i].astype(np.float32),
                            "source": str(file),
                            "source_frame": i,
                            "root_vel": root_vel[i].astype(np.float32),
                            "music_embedding": None,
                            "motion_embedding": None,
                            "audio_summary": None,
                        }
                    )
                    if len(candidates) >= max_candidates:
                        break

    return candidates[:max_candidates]


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


def trajectory_curvature_score(traj_physical: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj_physical, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    traj = traj[:, :2]
    if len(traj) < 3:
        return np.zeros((len(traj),), dtype=np.float32)

    v1 = traj[1:-1] - traj[:-2]
    v2 = traj[2:] - traj[1:-1]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    cos = np.sum(v1 * v2, axis=1, keepdims=False) / np.clip(n1[:, 0] * n2[:, 0], 1e-8, None)
    turn = 1.0 - np.clip(cos, -1.0, 1.0)
    out = np.zeros((len(traj),), dtype=np.float32)
    out[1:-1] = turn.astype(np.float32)
    return normalize_01(smooth_1d(out, 5))


def audio_onset_score(audio_feature: np.ndarray, onset_index: int = 768) -> np.ndarray:
    audio_feature = np.asarray(audio_feature, dtype=np.float32)
    if audio_feature.ndim != 2 or len(audio_feature) == 0:
        return np.zeros((0,), dtype=np.float32)

    if audio_feature.shape[1] > onset_index:
        onset = np.maximum(audio_feature[:, onset_index], 0.0)
    else:
        diff = np.zeros((len(audio_feature),), dtype=np.float32)
        if len(audio_feature) > 1:
            diff[1:] = np.linalg.norm(audio_feature[1:] - audio_feature[:-1], axis=1)
        onset = diff
    return normalize_01(smooth_1d(onset, 5))


def choose_auto_frames(
    audio_feature: np.ndarray,
    traj_physical: np.ndarray,
    num_frames: int,
    count: int,
    existing_frames: Sequence[int],
    min_gap: int = 18,
    edge_margin: int = 8,
    music_weight: float = 0.6,
    trajectory_weight: float = 0.4,
) -> List[int]:
    if count <= 0:
        return []

    onset = audio_onset_score(audio_feature)
    if len(onset) != num_frames:
        x_old = np.linspace(0.0, 1.0, len(onset))
        x_new = np.linspace(0.0, 1.0, num_frames)
        onset = np.interp(x_new, x_old, onset).astype(np.float32)

    curvature = trajectory_curvature_score(traj_physical)
    if len(curvature) != num_frames:
        x_old = np.linspace(0.0, 1.0, len(curvature))
        x_new = np.linspace(0.0, 1.0, num_frames)
        curvature = np.interp(x_new, x_old, curvature).astype(np.float32)

    score = float(music_weight) * normalize_01(onset) + float(trajectory_weight) * normalize_01(curvature)

    blocked = np.zeros((num_frames,), dtype=bool)
    for frame in list(existing_frames) + [0, num_frames - 1]:
        start = max(0, int(frame) - min_gap)
        end = min(num_frames, int(frame) + min_gap + 1)
        blocked[start:end] = True
    blocked[:edge_margin] = True
    blocked[num_frames - edge_margin:] = True

    frames = []
    for _ in range(count):
        candidate_score = score.copy()
        candidate_score[blocked] = -1e9
        frame = int(candidate_score.argmax())
        if candidate_score[frame] < -1e8:
            break
        frames.append(frame)
        start = max(0, frame - min_gap)
        end = min(num_frames, frame + min_gap + 1)
        blocked[start:end] = True

    if len(frames) < count:
        # Deterministic fallback to evenly spaced slots that do not collide with existing frames.
        grid = [
            int(round((i + 1) * (num_frames - 1) / (count + 1)))
            for i in range(count)
        ]
        for frame in grid:
            if len(frames) >= count:
                break
            if all(abs(frame - f) >= min_gap for f in frames + list(existing_frames) + [0, num_frames - 1]):
                frames.append(max(edge_margin, min(num_frames - edge_margin - 1, frame)))

    return sorted(set(int(f) for f in frames))[:count]


def pose_feature(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    return pose[POSE_FEATURE_INDEX].astype(np.float32)


def pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    fa = pose_feature(a)
    fb = pose_feature(b)
    return float(np.sqrt(np.mean((fa - fb) ** 2)))


def interpolate_pose_feature_at_frame(
    frame: int,
    anchors: List[Tuple[int, np.ndarray]],
) -> np.ndarray:
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
    if n <= 1e-8:
        return np.zeros((2,), dtype=np.float32)
    return (v / n).astype(np.float32)


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


def choose_candidate_for_frame(
    frame: int,
    candidates: List[Dict[str, object]],
    anchors: List[Tuple[int, np.ndarray]],
    traj_physical: np.ndarray,
    selected_poses: List[np.ndarray],
    w_pose: float = 1.0,
    w_traj: float = 0.25,
    w_diversity: float = 0.15,
) -> AutoKeyframe:
    ref_feat = interpolate_pose_feature_at_frame(frame, anchors)
    tangent = target_traj_tangent(traj_physical, frame)

    best = None
    for cand in candidates:
        pose = np.asarray(cand["pose"], dtype=np.float32)
        pfeat = pose_feature(pose)
        pose_cost = float(np.sqrt(np.mean((pfeat - ref_feat) ** 2)))
        traj_cost = direction_cost(cand.get("root_vel", None), tangent)

        if selected_poses:
            div = min(pose_distance(pose, p) for p in selected_poses)
            diversity_cost = 1.0 / (1.0 + div)
        else:
            diversity_cost = 0.0

        score = float(w_pose) * pose_cost + float(w_traj) * traj_cost + float(w_diversity) * diversity_cost

        if best is None or score < best.score:
            best = AutoKeyframe(
                frame=int(frame),
                pose=pose.astype(np.float32),
                score=float(score),
                source=str(cand.get("source", "")),
                source_frame=int(cand.get("source_frame", -1)),
                score_parts={
                    "pose_cost": pose_cost,
                    "trajectory_direction_cost": float(traj_cost),
                    "diversity_cost": float(diversity_cost),
                },
            )

    if best is None:
        raise RuntimeError("No RAG candidate available for auto keyframe planning.")
    return best


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
) -> AutoKeyframePlan:
    candidates = load_rag_candidates(
        rag_db,
        normalizer=normalizer,
        pose_space=rag_pose_space,
        max_candidates=max_candidates,
        sample_stride=sample_stride,
    )
    if not candidates:
        raise RuntimeError(
            f"No valid RAG candidates found in {rag_db}. "
            "Build one with build_dunhuang_rag_db.py or provide .npy motions."
        )

    anchors = [(0, start_pose), (num_frames - 1, end_pose)]
    for f, p in zip(user_mid_frames, user_mid_poses):
        anchors.append((int(f), np.asarray(p, dtype=np.float32)))
    anchors = sorted(anchors, key=lambda x: x[0])

    existing_frames = [f for f, _ in anchors]
    frames = choose_auto_frames(
        audio_feature=audio_feature,
        traj_physical=traj_physical,
        num_frames=num_frames,
        count=int(max_auto_keyframes),
        existing_frames=existing_frames,
        min_gap=int(min_gap),
        music_weight=float(music_weight),
        trajectory_weight=float(trajectory_weight),
    )

    selected = []
    selected_poses = []
    for frame in frames:
        kf = choose_candidate_for_frame(
            frame=frame,
            candidates=candidates,
            anchors=anchors + [(k.frame, k.pose) for k in selected],
            traj_physical=traj_physical,
            selected_poses=selected_poses,
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
        },
    )


def save_auto_keyframes(plan: AutoKeyframePlan, out_motion_path: str, prefix: str = "auto_mid"):
    out_motion_path = Path(out_motion_path)
    out_dir = out_motion_path.parent
    stem = out_motion_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    frames = []
    rows = []
    for idx, kf in enumerate(plan.keyframes, start=1):
        path = out_dir / f"{stem}_{prefix}{idx}_f{kf.frame:03d}.npy"
        np.save(path, kf.pose.astype(np.float32))
        paths.append(str(path))
        frames.append(int(kf.frame))
        rows.append(
            {
                "index": idx,
                "frame": int(kf.frame),
                "pose_path": str(path),
                "score": float(kf.score),
                "source": kf.source,
                "source_frame": int(kf.source_frame),
                "score_parts": kf.score_parts,
            }
        )

    meta_path = out_dir / f"{stem}_{prefix}_plan.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "auto_keyframes": rows,
                "frame_candidates": plan.frame_candidates,
                "planner_meta": plan.meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return paths, frames, str(meta_path)


def append_csv(existing: str, values: Sequence[object]) -> str:
    old = [x.strip() for x in str(existing or "").replace(";", ",").split(",") if x.strip()]
    new = [str(v) for v in values]
    return ",".join(old + new)


def _cli():
    parser = argparse.ArgumentParser(description="Inspect or test auto keyframe planner.")
    parser.add_argument("--rag_db", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pose_space", default="normalized", choices=["normalized", "physical"])
    parser.add_argument("--max_candidates", type=int, default=5000)
    args = parser.parse_args()
    normalizer = load_normalizer_from_checkpoint(args.checkpoint)
    cands = load_rag_candidates(args.rag_db, normalizer=normalizer, pose_space=args.pose_space, max_candidates=args.max_candidates)
    print(f"loaded candidates: {len(cands)}")
    if cands:
        print("first:", {k: v for k, v in cands[0].items() if k != "pose"})
        print("pose shape:", cands[0]["pose"].shape)


if __name__ == "__main__":
    _cli()
