"""Utilities for RAG segment training in EDGE.

This module is intentionally standalone.  It reuses EDGE's existing 151-D
motion representation:
    [0:4] foot contacts, [4:7] root xyz, [7:151] 24 joints * 6D rotation.

The training stage uses retrieved continuous clips as *conditioning prior* and
uses the target motion itself as the reconstruction / segment imitation target.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CONTACT_SLICE = slice(0, 4)
ROOT_SLICE = slice(4, 7)
ROT_SLICE = slice(7, 151)
ROOT_X_IDX = 4
ROOT_Z_IDX = 6

# Conservative SMPL-style joint groups for training/inference priors.
JOINT_GROUPS: Dict[str, List[int]] = {
    "arms": [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
    # Safe upper body: no root/pelvis orientation and no legs.
    "upper": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "upper_safe_plus": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    # For ablation only.  These may destabilize foot contacts when used with
    # trajectory anchoring.
    "torso_arms": [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "body_no_root": list(range(24)),
    "all_rot": list(range(24)),
}


def load_jsonl(path: str | Path) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SimpleNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(np.abs(self.std) < 1e-8, 1.0, self.std).astype(np.float32)

    def normalize(self, x):
        if torch.is_tensor(x):
            mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
            return (x - mean) / std
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def unnormalize(self, x):
        if torch.is_tensor(x):
            mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
            return x * std + mean
        return np.asarray(x, dtype=np.float32) * self.std + self.mean


def load_normalizer_from_checkpoint(path: str | Path | None) -> Optional[SimpleNormalizer]:
    if not path:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    norm = ckpt.get("normalizer") if isinstance(ckpt, dict) else None
    if norm is None:
        return None
    if isinstance(norm, dict) and "mean" in norm and "std" in norm:
        return SimpleNormalizer(np.asarray(norm["mean"], dtype=np.float32), np.asarray(norm["std"], dtype=np.float32))
    if hasattr(norm, "mean") and hasattr(norm, "std"):
        return SimpleNormalizer(np.asarray(norm.mean, dtype=np.float32), np.asarray(norm.std, dtype=np.float32))
    return None


def _edge_vectorize_pos_q(pos: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Convert processed Dunhuang/AIST pkl {'pos','q'} to EDGE 151-D physical motion."""
    try:
        from dataset.quaternion import ax_to_6v
        from dataset.preprocess import vectorize_many
        from vis import SMPLSkeleton
    except Exception as exc:  # pragma: no cover
        raise ImportError("EDGE dataset/vis modules are required to load pos/q pkl files") from exc

    pos_t = torch.as_tensor(pos, dtype=torch.float32).unsqueeze(0)
    q_t = torch.as_tensor(q, dtype=torch.float32).unsqueeze(0)
    if q_t.ndim == 3:
        q_t = q_t.reshape(q_t.shape[0], q_t.shape[1], -1, 3)

    smpl = SMPLSkeleton()
    joints = smpl.forward(q_t, pos_t)
    feet = joints[:, :, (7, 8, 10, 11)]
    feetv = torch.zeros(feet.shape[:3], dtype=torch.float32)
    feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
    contacts = (feetv < 0.01).to(q_t)
    q_6v = ax_to_6v(q_t)
    motion = vectorize_many([contacts, pos_t, q_6v]).float()[0].detach().cpu().numpy()
    return motion.astype(np.float32)


def load_motion_151(path: str | Path) -> np.ndarray:
    """Load a motion file as physical-space [T,151].

    Supports npy [T,151], pkl [T,151], and processed pkl dicts with keys
    {'pos','q'}.
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[-1] == 151:
            return arr.astype(np.float32)
        if arr.shape == ():
            obj = arr.item()
            if isinstance(obj, dict):
                if "motion" in obj:
                    arr = np.asarray(obj["motion"], dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[-1] == 151:
                        return arr
        raise ValueError(f"Unsupported npy motion format: {path}, shape={getattr(arr, 'shape', None)}")

    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[-1] == 151:
                return arr
        if isinstance(obj, dict):
            for key in ("motion", "motion_151", "x", "pose"):
                if key in obj:
                    arr = np.asarray(obj[key], dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[-1] == 151:
                        return arr
            if "pos" in obj and "q" in obj:
                return _edge_vectorize_pos_q(np.asarray(obj["pos"]), np.asarray(obj["q"]))
        raise ValueError(f"Unsupported pkl motion format: {path}")

    raise ValueError(f"Unsupported motion suffix: {path}")


def crop_or_pad_center(motion: np.ndarray, center: int, seq_len: int) -> np.ndarray:
    """Return [seq_len, C] clip centered at source frame with edge padding."""
    motion = np.asarray(motion, dtype=np.float32)
    T = motion.shape[0]
    if T <= 0:
        raise ValueError("empty motion")
    center = int(np.clip(center, 0, T - 1))
    start = center - seq_len // 2
    end = start + seq_len
    src_start = max(0, start)
    src_end = min(T, end)
    clip = motion[src_start:src_end]
    if len(clip) == 0:
        clip = motion[[center]]
    if len(clip) < seq_len:
        before = max(0, -start)
        after = seq_len - len(clip) - before
        clip = np.pad(clip, ((before, after), (0, 0)), mode="edge")
    return clip[:seq_len].astype(np.float32)


def normalize_motion(motion: np.ndarray, normalizer: Optional[Any]) -> np.ndarray:
    if normalizer is None:
        return np.asarray(motion, dtype=np.float32)
    out = normalizer.normalize(torch.from_numpy(np.asarray(motion, dtype=np.float32))).detach().cpu().numpy()
    return out.astype(np.float32)


def build_feature_mask(seq_len: int, mode: str = "upper", include_contacts: bool = False) -> np.ndarray:
    """Build [T,151] feature mask for retrieved prior conditioning/loss."""
    mask = np.zeros((seq_len, 151), dtype=np.float32)
    if include_contacts:
        mask[:, 0:4] = 1.0

    mode = str(mode)
    if mode in ("full", "all"):
        mask[:, :] = 1.0
        # Keep root X/Z free to avoid fighting trajectory anchor.
        mask[:, ROOT_X_IDX] = 0.0
        mask[:, ROOT_Z_IDX] = 0.0
        return mask

    joints = JOINT_GROUPS.get(mode, JOINT_GROUPS["upper"])
    for j in joints:
        start = 7 + int(j) * 6
        end = start + 6
        if 7 <= start and end <= 151:
            mask[:, start:end] = 1.0
    return mask


def build_segment_mask(
    seq_len: int,
    segment_start: int,
    segment_end: int,
    feature_mode: str = "upper",
    protect_width: int = 2,
    include_contacts: bool = False,
) -> np.ndarray:
    """Build [T,151] mask active only inside middle segment and away from endpoints."""
    segment_start = int(np.clip(segment_start, 0, seq_len - 1))
    segment_end = int(np.clip(segment_end, segment_start + 1, seq_len))
    feature_mask = build_feature_mask(seq_len, feature_mode, include_contacts=include_contacts)
    time_mask = np.zeros((seq_len, 1), dtype=np.float32)
    time_mask[segment_start:segment_end] = 1.0
    if protect_width > 0:
        time_mask[:protect_width] = 0.0
        time_mask[seq_len - protect_width:] = 0.0
    return feature_mask * time_mask


def random_segment(seq_len: int, min_len: int, max_len: int, rng: np.random.Generator) -> Tuple[int, int]:
    min_len = max(2, int(min_len))
    max_len = max(min_len, int(max_len))
    max_len = min(max_len, seq_len - 4)
    seg_len = int(rng.integers(min_len, max_len + 1))
    start_min = max(2, seq_len // 5)
    start_max = max(start_min + 1, seq_len - seg_len - 2)
    start = int(rng.integers(start_min, start_max + 1))
    return start, start + seg_len


def normalize_root_to_start(motion: np.ndarray) -> np.ndarray:
    """Make physical clip start at local root X/Z = 0, matching DunhuangDataset."""
    motion = np.asarray(motion, dtype=np.float32).copy()
    motion[:, ROOT_X_IDX] -= motion[0, ROOT_X_IDX]
    motion[:, ROOT_Z_IDX] -= motion[0, ROOT_Z_IDX]
    return motion
