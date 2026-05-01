"""Shared IO utilities for MMR/RAG scripts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resample_sequence(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,C], got {arr.shape}")
    if arr.shape[0] == target_len:
        return arr.astype(np.float32)
    tensor = torch.from_numpy(arr).float().unsqueeze(0).transpose(1, 2)
    tensor = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
    return tensor.transpose(1, 2).squeeze(0).numpy().astype(np.float32)


def load_normalizer_from_checkpoint(checkpoint_path: str):
    if not checkpoint_path:
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    norm_data = ckpt.get("normalizer") if isinstance(ckpt, dict) else None
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
            raise ValueError("pose_space=physical requires checkpoint normalizer")
        t = torch.from_numpy(motion).float()
        if t.ndim == 1:
            out = normalizer.normalize(t[None, None, :])
            return to_numpy(out)[0, 0].astype(np.float32)
        if t.ndim == 2:
            out = normalizer.normalize(t[None])
            return to_numpy(out)[0].astype(np.float32)
        if t.ndim == 3:
            out = normalizer.normalize(t)
            return to_numpy(out).astype(np.float32)
    raise ValueError(f"Unsupported pose_space: {pose_space}")


def load_motion_151(path: str | Path, key: Optional[str] = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".pkl":
        return pkl_to_motion_151(path)
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        if key is not None:
            arr = data[key]
        else:
            for k in ("motion", "motion_151", "poses", "pose_seq", "pose", "pos"):
                if k in data:
                    arr = data[k]
                    break
            else:
                raise ValueError(f"Cannot infer motion key from {path}; keys={list(data.keys())}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 151:
        arr = arr[None]
    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"Expected [T,151], got {arr.shape} from {path}")
    return arr.astype(np.float32)


def load_audio_feature(path: str | Path, key: Optional[str] = None) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        if key is not None:
            arr = data[key]
        else:
            for k in ("audio", "audio_feature", "feature", "features"):
                if k in data:
                    arr = data[k]
                    break
            else:
                raise ValueError(f"Cannot infer audio key from {path}; keys={list(data.keys())}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected audio feature [T,C], got {arr.shape} from {path}")
    return arr.astype(np.float32)


def pkl_to_motion_151(path: Path) -> np.ndarray:
    if SMPLSkeleton is None or ax_to_6v is None or vectorize_many is None:
        raise ImportError("Cannot convert pkl to 151-D motion; dataset/vis helpers are unavailable")
    data = pickle.load(open(path, "rb"))
    if "pos" not in data or "q" not in data:
        raise ValueError(f"{path} must contain pos and q")
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


def iter_motion_files(path: str | Path):
    path = Path(path)
    if path.is_file():
        yield path
    else:
        for ext in ("*.npy", "*.npz", "*.pkl"):
            yield from sorted(path.rglob(ext))


def rhythm_summary_from_audio(audio: np.ndarray, onset_index: int = 768) -> Dict[str, float]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or len(audio) < 2:
        return {"onset_mean": 0.0, "onset_std": 0.0, "energy_mean": 0.0, "energy_std": 0.0}
    if audio.shape[1] > onset_index:
        onset = np.maximum(audio[:, onset_index], 0.0)
    else:
        onset = np.linalg.norm(np.diff(audio, axis=0, prepend=audio[:1]), axis=1)
    energy = np.linalg.norm(audio, axis=1)
    return {
        "onset_mean": float(onset.mean()),
        "onset_std": float(onset.std()),
        "energy_mean": float(energy.mean()),
        "energy_std": float(energy.std()),
    }


def rhythm_summary_from_motion(motion: np.ndarray) -> Dict[str, float]:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or len(motion) < 2:
        return {"motion_energy_mean": 0.0, "motion_energy_std": 0.0, "root_speed_mean": 0.0}
    delta = motion[1:] - motion[:-1]
    motion_energy = np.linalg.norm(delta[:, 7:151], axis=1)
    root = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    root_speed = np.linalg.norm(root[1:] - root[:-1], axis=1)
    return {
        "motion_energy_mean": float(motion_energy.mean()),
        "motion_energy_std": float(motion_energy.std()),
        "root_speed_mean": float(root_speed.mean()),
    }
