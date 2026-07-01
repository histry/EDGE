#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.1 MotionRAG-Diff for EDGE 151D Dunhuang whole-song generation
==================================================================

This file is designed as a drop-in research patch for an EDGE-style repository.
It does not depend on README assumptions. It directly operates on EDGE 151D
motion arrays and can rebuild a source-aware motion database from a new
"change" dataset.

Core versions included:
- V43: true lower-body IK that writes lower-body rotation channels back into
       EDGE 151D, rather than saving fake foot XYZ columns.
- V44: music-motion contrastive learning for retrieval alignment.
- V45: residual temporal Motion Refiner to escape pure stitching.
- V46: retrieval-augmented conditional residual diffusion with IK finalization.
- V46.1 safety fixes: dynamic landing damping, C1 Hanning flight gate, intentional-slide release guard.

Expected EDGE 151D convention
-----------------------------
root translation: motion[:, 4:7] = [x, y, z]
rot6d local joints: motion[:, 7:151].reshape(T, 24, 6)
foot FK ids: [7, 8, 10, 11]

Typical commands
----------------
python tools/v46_motionrag_diff.py build-db --motion_dirs change data/motions --out_db output/v46_db
python tools/v46_motionrag_diff.py train-contrastive --db output/v46_db/events.npz --out output/v46_db/v44_contrastive.pt
python tools/v46_motionrag_diff.py train-refiner --db output/v46_db/events.npz --out output/v46_db/v45_refiner.pt
python tools/v46_motionrag_diff.py train-diffusion --db output/v46_db/events.npz --out output/v46_db/v46_diffusion.pt
python tools/v46_motionrag_diff.py generate --audio test_music_bank/dunhuangwu2.wav --db output/v46_db/events.npz --out output/v46_dunhuangwu2.npy
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

try:
    import scipy.io.wavfile as wavfile
except Exception:  # pragma: no cover
    wavfile = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT6D_START = 7
ROT6D_END = 151
EDGE_DIM = 151
NUM_JOINTS = 24
DEFAULT_FOOT_JOINTS = (7, 8, 10, 11)
LOWER_BODY_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)

FALLBACK_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
FALLBACK_OFFSETS = np.array(
    [
        [0.0000, 0.0000, 0.0000],
        [0.0586, -0.0823, 0.0177],
        [-0.0603, -0.0905, 0.0135],
        [0.0044, 0.1244, -0.0384],
        [0.0435, -0.3865, 0.0080],
        [-0.0433, -0.3837, -0.0048],
        [0.0045, 0.1379, 0.0268],
        [-0.0148, -0.4269, -0.0374],
        [0.0195, -0.4200, -0.0346],
        [-0.0023, 0.0560, 0.0029],
        [0.0411, -0.0603, 0.1220],
        [-0.0348, -0.0621, 0.1309],
        [0.0264, 0.2146, -0.0375],
        [0.0714, 0.1138, -0.0189],
        [-0.0824, 0.1125, -0.0237],
        [0.0103, 0.0889, 0.0504],
        [0.1229, 0.0452, -0.0190],
        [-0.1132, 0.0469, -0.0085],
        [0.2553, -0.0156, -0.0229],
        [-0.2601, -0.0144, -0.0319],
        [0.2657, 0.0127, -0.0074],
        [-0.2691, 0.0067, -0.0060],
        [0.0867, -0.0106, -0.0156],
        [-0.0888, -0.0087, -0.0101],
    ],
    dtype=np.float32,
)


def _load_repo_fk_tree() -> Tuple[np.ndarray, np.ndarray, str]:
    """Try to reuse the local EDGE repository FK constants."""
    candidates = [
        "tools.v41b_inject_min_foot_y_to_db",
        "tools.v42_root_footplant_physics_optimizer",
    ]
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["PARENTS", "OFFSETS"])
            parents = np.asarray(getattr(mod, "PARENTS"), dtype=np.int64)
            offsets = np.asarray(getattr(mod, "OFFSETS"), dtype=np.float32)
            if parents.shape[0] == NUM_JOINTS and offsets.shape == (NUM_JOINTS, 3):
                return parents, offsets, name
        except Exception:
            pass
    return FALLBACK_PARENTS, FALLBACK_OFFSETS, "fallback_smpl_like_tree"


PARENTS, OFFSETS, FK_TREE_SOURCE = _load_repo_fk_tree()


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)


def load_json(path: Optional[str | Path], default: Optional[dict] = None) -> dict:
    if not path:
        return dict(default or {})
    p = Path(path)
    if not p.exists():
        return dict(default or {})
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    base = dict(default or {})
    base.update(data)
    return base


def save_json(obj: object, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def smooth_np(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or ndi is None:
        return x
    return ndi.gaussian_filter1d(x, sigma=float(sigma), axis=0, mode="nearest")


def median_bool_filter(x: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or ndi is None:
        return x.astype(bool)
    return ndi.median_filter(x.astype(np.uint8), size=size).astype(bool)


def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def resample_motion_np(motion: np.ndarray, new_len: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if new_len <= 1 or motion.shape[0] <= 1:
        return np.repeat(motion[:1], max(1, new_len), axis=0)
    old_x = np.linspace(0.0, 1.0, motion.shape[0])
    new_x = np.linspace(0.0, 1.0, new_len)
    out = np.empty((new_len, motion.shape[1]), dtype=np.float32)
    for d in range(motion.shape[1]):
        out[:, d] = np.interp(new_x, old_x, motion[:, d])
    return out


def normalize_motion_shape(arr: np.ndarray) -> List[np.ndarray]:
    arr = np.asarray(arr)
    outs: List[np.ndarray] = []
    if arr.ndim == 2 and arr.shape[1] >= EDGE_DIM:
        outs.append(arr[:, :EDGE_DIM].astype(np.float32))
    elif arr.ndim == 3 and arr.shape[-1] >= EDGE_DIM:
        for i in range(arr.shape[0]):
            outs.append(arr[i, :, :EDGE_DIM].astype(np.float32))
    return outs


def load_motion_file(path: str | Path) -> List[np.ndarray]:
    p = Path(path)
    outs: List[np.ndarray] = []
    try:
        if p.suffix.lower() == ".npy":
            outs.extend(normalize_motion_shape(np.load(p, allow_pickle=True)))
        elif p.suffix.lower() == ".npz":
            data = np.load(p, allow_pickle=True)
            for k in data.files:
                if k.lower() in {"motion", "motions", "x", "arr_0", "data"} or data[k].ndim in (2, 3):
                    outs.extend(normalize_motion_shape(data[k]))
        elif p.suffix.lower() in {".pkl", ".pickle"}:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                for k in ["motion", "motions", "x", "poses", "data"]:
                    if k in obj:
                        outs.extend(normalize_motion_shape(np.asarray(obj[k])))
            else:
                outs.extend(normalize_motion_shape(np.asarray(obj)))
    except Exception as exc:
        print(f"[V46 WARN] failed loading {p}: {exc}", file=sys.stderr)
    return [x for x in outs if x.ndim == 2 and x.shape[0] >= 8]


def rot6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def matrix_to_rot6d_np(mat: np.ndarray) -> np.ndarray:
    return mat[..., :, 0:2].reshape(*mat.shape[:-2], 6).astype(np.float32)


def fk_24_np(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < ROT6D_END:
        raise ValueError(f"Expected EDGE 151D motion [T,151], got {motion.shape}")
    T = motion.shape[0]
    root = motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    rot6d = motion[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)
    local_r = rot6d_to_matrix_np(rot6d)
    global_r = np.zeros((T, NUM_JOINTS, 3, 3), dtype=np.float32)
    joints = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    global_r[:, 0] = local_r[:, 0]
    joints[:, 0] = root
    for j in range(1, NUM_JOINTS):
        p = int(PARENTS[j])
        if p < 0:
            global_r[:, j] = local_r[:, j]
            joints[:, j] = root
        else:
            global_r[:, j] = np.matmul(global_r[:, p], local_r[:, j])
            offset = OFFSETS[j].astype(np.float32)[None, :, None]
            joints[:, j] = joints[:, p] + np.matmul(global_r[:, p], offset)[..., 0]
    return joints


def rot6d_to_matrix_torch(x):
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d_torch(mat):
    return mat[..., :, 0:2].reshape(*mat.shape[:-2], 6)


def project_rot6d_torch(x):
    return matrix_to_rot6d_torch(rot6d_to_matrix_torch(x))


def fk_24_torch(motion, parents=None, offsets=None):
    parents = torch.as_tensor(PARENTS if parents is None else parents, device=motion.device, dtype=torch.long)
    offsets = torch.as_tensor(OFFSETS if offsets is None else offsets, device=motion.device, dtype=motion.dtype)
    T = motion.shape[0]
    root = motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    rot6d = motion[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)
    local_r = rot6d_to_matrix_torch(rot6d)
    global_r = []
    joints = []
    for j in range(NUM_JOINTS):
        p = int(parents[j].item())
        if j == 0 or p < 0:
            gr = local_r[:, j]
            pos = root
        else:
            gr = torch.matmul(global_r[p], local_r[:, j])
            off = offsets[j].view(1, 3, 1)
            pos = joints[p] + torch.matmul(global_r[p], off).squeeze(-1)
        global_r.append(gr)
        joints.append(pos)
    return torch.stack(joints, dim=1)


def root_yaw_np(motion: np.ndarray) -> np.ndarray:
    root_r = rot6d_to_matrix_np(motion[:, ROT6D_START:ROT6D_START + 6].reshape(-1, 1, 6))[:, 0]
    forward = root_r[:, :, 2]
    return np.arctan2(forward[:, 0], forward[:, 2]).astype(np.float32)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b)).astype(np.float32)


@dataclasses.dataclass
class V46Config:
    fps: float = 30.0
    window_len: int = 120
    hop_len: int = 60
    min_event_frames: int = 36
    max_event_frames: int = 180
    db_feature_dim: int = 32
    embed_dim: int = 128
    top_k: int = 32
    beam_size: int = 8
    overlap: int = 12
    retrieval_source_penalty: float = 0.08
    retrieval_transition_penalty: float = 0.65
    retrieval_warp_penalty: float = 0.18
    retrieval_repeat_penalty: float = 0.15
    ik_enable: bool = True
    ik_iters: int = 80
    ik_lr: float = 0.035
    ik_chunk: int = 240
    ik_pose_w: float = 0.035
    ik_temporal_w: float = 0.055
    ik_root_w: float = 0.015
    ik_contact_w: float = 1.0
    ik_penetration_w: float = 0.60
    ik_contact_high: float = 0.58
    ik_contact_low: float = 0.38
    ik_height_margin: float = 0.050
    ik_speed_gate_mpf: float = 0.035
    ik_max_delta_rot: float = 0.65
    # V46.1: do not lock intentional Dunhuang cloud-step / sliding-step contacts.
    # If a contacted foot moves smoothly over this XZ span, it is treated as
    # designed sliding support rather than skating error. 0.12 m is the default
    # threshold requested for Dunhuang cloud-step safety.
    ik_slide_release_m: float = 0.12
    ik_slide_release_min_frames: int = 4
    # V46.1: root-Y ballistic/damping pass. It is deliberately C1-safe and
    # never breaks a damping cycle mid-contact.
    root_y_physics_enable: bool = True
    root_y_flight_strength: float = 0.18
    root_y_min_flight_frames: int = 5
    root_y_damping_max_dip: float = 0.018
    lower_body_only: bool = True
    refiner_enable: bool = True
    diffusion_enable: bool = True
    diffusion_steps: int = 50
    diffusion_train_steps: int = 15000
    refiner_train_steps: int = 8000
    contrastive_epochs: int = 120
    batch_size: int = 64
    lr: float = 2e-4
    seed: int = 42
    device: str = "cuda"

    @staticmethod
    def from_json(path: Optional[str | Path]) -> "V46Config":
        cfg = V46Config()
        if path and Path(path).exists():
            data = load_json(path)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def apply_env(self) -> "V46Config":
        env_map = {
            "V46_ENABLE_TRUE_IK": ("ik_enable", lambda x: bool(int(x))),
            "V46_ENABLE_REFINER": ("refiner_enable", lambda x: bool(int(x))),
            "V46_ENABLE_DIFFUSION": ("diffusion_enable", lambda x: bool(int(x))),
            "V46_TOP_K": ("top_k", int),
            "V46_BEAM_SIZE": ("beam_size", int),
            "V46_IK_ITERS": ("ik_iters", int),
            "V46_IK_SLIDE_RELEASE_M": ("ik_slide_release_m", float),
            "V46_ENABLE_ROOT_Y_PHYSICS": ("root_y_physics_enable", lambda x: bool(int(x))),
            "V46_DIFFUSION_STEPS": ("diffusion_steps", int),
            "V46_DEVICE": ("device", str),
        }
        for e, (attr, caster) in env_map.items():
            if e in os.environ:
                setattr(self, attr, caster(os.environ[e]))
        if self.device == "cuda" and (torch is None or not torch.cuda.is_available()):
            self.device = "cpu"
        return self


def event_descriptor(motion: np.ndarray, fps: float = 30.0) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    T = motion.shape[0]
    joints = fk_24_np(motion)
    root = motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    root_v = np.zeros_like(root)
    root_v[1:] = root[1:] - root[:-1]
    joint_v = np.zeros_like(joints)
    joint_v[1:] = joints[1:] - joints[:-1]
    foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
    foot_vxz = np.zeros(foot.shape[:2], dtype=np.float32)
    foot_vxz[1:] = np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
    foot_y = foot[..., 1]
    floor = np.percentile(foot_y.reshape(-1), 5)
    contact = (foot_y < floor + 0.05) & (foot_vxz < 0.035)
    yaw = root_yaw_np(motion)
    yaw_v = np.zeros_like(yaw)
    yaw_v[1:] = angle_diff(yaw[1:], yaw[:-1])
    lower_ids = [1, 2, 4, 5, 7, 8, 10, 11]
    upper_ids = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_energy = float(np.mean(np.linalg.norm(joint_v[:, lower_ids], axis=-1)))
    upper_energy = float(np.mean(np.linalg.norm(joint_v[:, upper_ids], axis=-1)))
    desc = np.array(
        [
            T / fps,
            np.linalg.norm(root[-1, [0, 2]] - root[0, [0, 2]]),
            np.mean(np.linalg.norm(root_v[:, [0, 2]], axis=-1)),
            np.percentile(np.linalg.norm(root_v[:, [0, 2]], axis=-1), 95),
            np.mean(np.abs(root_v[:, 1])),
            np.mean(np.linalg.norm(joint_v.reshape(T, -1, 3), axis=-1)),
            np.percentile(np.linalg.norm(joint_v.reshape(T, -1, 3), axis=-1), 95),
            lower_energy,
            upper_energy,
            lower_energy / max(upper_energy, 1e-6),
            np.mean(contact),
            np.mean(contact[:, :2]),
            np.mean(contact[:, 2:]),
            np.mean(foot_vxz),
            np.percentile(foot_vxz, 95),
            float(np.sum(yaw_v)),
            float(np.mean(np.abs(yaw_v))),
            float(np.percentile(np.abs(yaw_v), 95)),
            float(np.max(root[:, 1]) - np.min(root[:, 1])),
            float(np.mean(np.abs(motion[:, ROT6D_START:ROT6D_END]))),
        ],
        dtype=np.float32,
    )
    stats = []
    for q in [5, 25, 50, 75, 95]:
        stats.append(np.percentile(np.linalg.norm(joint_v.reshape(T, -1, 3), axis=-1), q))
    desc = np.concatenate([desc, np.asarray(stats, dtype=np.float32)], axis=0)
    if desc.shape[0] < 32:
        desc = np.pad(desc, (0, 32 - desc.shape[0]))
    return desc[:32].astype(np.float32)


def motion_boundary_state(motion: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joints = fk_24_np(motion)
    v = np.zeros_like(joints)
    v[1:] = joints[1:] - joints[:-1]
    entry = np.concatenate([joints[0].reshape(-1), v[min(1, len(v) - 1)].reshape(-1)], axis=0)
    exit_ = np.concatenate([joints[-1].reshape(-1), v[-1].reshape(-1)], axis=0)
    foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
    foot_vxz = np.zeros(foot.shape[:2], dtype=np.float32)
    foot_vxz[1:] = np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
    floor = np.percentile(foot[..., 1].reshape(-1), 5)
    contact = ((foot[..., 1] < floor + 0.05) & (foot_vxz < 0.035)).astype(np.float32)
    return entry.astype(np.float32), exit_.astype(np.float32), contact[0], contact[-1]


def source_group_from_path(path: str | Path) -> str:
    p = Path(path)
    parts = p.parts[-4:]
    stem = p.stem
    tokens = stem.replace("-", "_").split("_")
    if len(tokens) >= 3:
        key = "_".join(tokens[:3])
    else:
        key = "/".join(parts[:-1]) if len(parts) > 1 else stem
    return key


def scan_motion_files(motion_dirs: Sequence[str], exts: Sequence[str] = (".npy", ".npz", ".pkl", ".pickle")) -> List[str]:
    files: List[str] = []
    for d in motion_dirs:
        if not d:
            continue
        p = Path(d)
        if p.is_file():
            if p.suffix.lower() in exts:
                files.append(str(p))
            continue
        if not p.exists():
            continue
        for ext in exts:
            files.extend(glob.glob(str(p / "**" / f"*{ext}"), recursive=True))
    return sorted(set(files))


def build_db(args: argparse.Namespace) -> int:
    cfg = V46Config.from_json(args.config).apply_env()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = ensure_dir(args.out_db)
    files = scan_motion_files(args.motion_dirs)
    if not files:
        raise FileNotFoundError(f"No motion files found in: {args.motion_dirs}")
    motions: List[np.ndarray] = []
    meta: List[dict] = []
    descs: List[np.ndarray] = []
    entries: List[np.ndarray] = []
    exits: List[np.ndarray] = []
    c0s: List[np.ndarray] = []
    c1s: List[np.ndarray] = []

    npy_dir = ensure_dir(out_dir / "events")
    event_idx = 0
    for f in files:
        seqs = load_motion_file(f)
        if not seqs:
            continue
        source = source_group_from_path(f)
        for seq_id, seq in enumerate(seqs):
            T = seq.shape[0]
            if T < cfg.min_event_frames:
                continue
            # Full natural segment when short enough; otherwise sliding source-aware events.
            starts = [0] if T <= cfg.max_event_frames else list(range(0, max(1, T - cfg.min_event_frames + 1), cfg.hop_len))
            for st in starts:
                end = min(T, st + cfg.window_len)
                if end - st < cfg.min_event_frames:
                    continue
                clip = seq[st:end].astype(np.float32)
                if clip.shape[0] > cfg.max_event_frames:
                    clip = clip[: cfg.max_event_frames]
                path = npy_dir / f"event_{event_idx:07d}.npy"
                np.save(path, clip)
                desc = event_descriptor(clip, cfg.fps)
                entry, exit_, c0, c1 = motion_boundary_state(clip)
                motions.append(clip)
                descs.append(desc)
                entries.append(entry)
                exits.append(exit_)
                c0s.append(c0)
                c1s.append(c1)
                meta.append(
                    {
                        "event_id": event_idx,
                        "path": str(path),
                        "source_file": f,
                        "source_group": source,
                        "seq_id": seq_id,
                        "start": int(st),
                        "end": int(st + clip.shape[0]),
                        "frames": int(clip.shape[0]),
                        "duration": float(clip.shape[0] / cfg.fps),
                    }
                )
                event_idx += 1

    if not meta:
        raise RuntimeError("No valid EDGE 151D motion events were built.")
    desc = np.stack(descs).astype(np.float32)
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1e-6
    desc_z = (desc - mean) / std
    db_path = out_dir / "events.npz"
    np.savez_compressed(
        db_path,
        desc=desc.astype(np.float32),
        desc_z=desc_z.astype(np.float32),
        desc_mean=mean.astype(np.float32),
        desc_std=std.astype(np.float32),
        entry=np.stack(entries).astype(np.float32),
        exit=np.stack(exits).astype(np.float32),
        contact_entry=np.stack(c0s).astype(np.float32),
        contact_exit=np.stack(c1s).astype(np.float32),
        paths=np.array([m["path"] for m in meta], dtype=object),
        source_groups=np.array([m["source_group"] for m in meta], dtype=object),
        durations=np.array([m["duration"] for m in meta], dtype=np.float32),
        frames=np.array([m["frames"] for m in meta], dtype=np.int32),
    )
    save_json({"version": "v46_db", "config": dataclasses.asdict(cfg), "events": meta, "num_events": len(meta), "fk_tree_source": FK_TREE_SOURCE}, out_dir / "events_meta.json")
    print(json.dumps({"db": str(db_path), "num_events": len(meta), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    return 0


def read_wav_mono(path: str | Path) -> Tuple[int, np.ndarray]:
    if wavfile is None:
        raise RuntimeError("scipy.io.wavfile is unavailable; install scipy or provide --slots_json")
    sr, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype.kind in "iu":
        maxv = max(float(np.iinfo(data.dtype).max), 1.0)
        data = data.astype(np.float32) / maxv
    else:
        data = data.astype(np.float32)
    if data.size == 0:
        data = np.zeros(1, dtype=np.float32)
    return int(sr), data


def audio_global_features(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size < 16:
        wav = np.pad(wav, (0, 16 - wav.size))
    frame = max(256, int(0.046 * sr))
    hop = max(64, int(0.023 * sr))
    vals = []
    cents = []
    zcrs = []
    for st in range(0, max(1, wav.size - frame + 1), hop):
        x = wav[st : st + frame]
        if x.size < frame:
            x = np.pad(x, (0, frame - x.size))
        vals.append(float(np.sqrt(np.mean(x * x) + 1e-8)))
        zcrs.append(float(np.mean(x[1:] * x[:-1] < 0)))
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
        cents.append(float((spec * freqs).sum() / max(spec.sum(), 1e-8) / max(sr, 1)))
    env = np.asarray(vals, dtype=np.float32)
    zcr = np.asarray(zcrs, dtype=np.float32)
    cen = np.asarray(cents, dtype=np.float32)
    onset = np.maximum(0.0, np.diff(env, prepend=env[:1]))
    if env.size > 4:
        ac = np.correlate(env - env.mean(), env - env.mean(), mode="full")[env.size - 1 :]
        lag = int(np.argmax(ac[1 : min(len(ac), 128)]) + 1) if len(ac) > 2 else 1
    else:
        lag = 1
    features = np.array(
        [
            wav.size / sr,
            env.mean(), env.std(), np.percentile(env, 90), np.percentile(env, 10),
            onset.mean(), onset.std(), np.percentile(onset, 90),
            zcr.mean(), zcr.std(), cen.mean(), cen.std(),
            lag / max(1.0, env.size),
            float(np.max(env)), float(np.min(env)), float(np.median(env)),
        ],
        dtype=np.float32,
    )
    # Add histogram-like dynamics.
    qs = np.percentile(env, [5, 25, 50, 75, 95]).astype(np.float32)
    oqs = np.percentile(onset, [5, 25, 50, 75, 95]).astype(np.float32)
    out = np.concatenate([features, qs, oqs], axis=0)
    if out.size < 32:
        out = np.pad(out, (0, 32 - out.size))
    return out[:32].astype(np.float32)


def audio_slots(path: str | Path, cfg: V46Config, slot_seconds: float = 4.0, slots_json: Optional[str] = None) -> Tuple[List[dict], np.ndarray]:
    if slots_json and Path(slots_json).exists():
        data = load_json(slots_json)
        slots = data.get("slots", data if isinstance(data, list) else [])
        feats = []
        for s in slots:
            base = np.asarray(s.get("feature", []), dtype=np.float32)
            if base.size < 32:
                base = np.pad(base, (0, 32 - base.size))
            feats.append(base[:32])
        return slots, np.stack(feats).astype(np.float32)
    sr, wav = read_wav_mono(path)
    total = wav.size / sr
    n_slots = max(1, int(math.ceil(total / slot_seconds)))
    slots: List[dict] = []
    feats: List[np.ndarray] = []
    for i in range(n_slots):
        st = int(i * slot_seconds * sr)
        ed = min(wav.size, int((i + 1) * slot_seconds * sr))
        seg = wav[st:ed]
        f = audio_global_features(seg, sr)
        # Convert audio descriptor into same coarse semantic order as motion descriptor.
        pseudo = np.zeros(32, dtype=np.float32)
        dur = max((ed - st) / sr, 1e-6)
        energy = float(f[1])
        onset = float(f[5])
        dyn = float(f[2] + f[6])
        pseudo[0] = dur
        pseudo[1] = energy * 2.0
        pseudo[2] = energy
        pseudo[3] = f[3]
        pseudo[4] = dyn
        pseudo[5] = energy + onset
        pseudo[6] = f[7]
        pseudo[7] = energy + 0.5 * onset
        pseudo[8] = energy
        pseudo[9] = 1.0 + onset
        pseudo[10] = np.clip(0.75 - energy, 0.0, 1.0)
        pseudo[13] = max(0.02, onset)
        pseudo[15] = 0.0
        pseudo[16] = onset
        pseudo[17] = f[7]
        pseudo[18] = dyn
        pseudo[19:] = f[: 32 - 19]
        role = "climax" if energy > np.percentile([energy, f[3], f[15]], 75) else ("calm" if energy < 0.03 else "normal")
        slots.append({"slot_id": i, "start": st / sr, "end": ed / sr, "duration": dur, "energy": energy, "onset": onset, "role": role})
        feats.append(pseudo.astype(np.float32))
    return slots, np.stack(feats).astype(np.float32)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int = 32, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ContrastiveModel(nn.Module):
    def __init__(self, feat_dim: int = 32, emb_dim: int = 128):
        super().__init__()
        self.motion = MLPEncoder(feat_dim, emb_dim)
        self.music = MLPEncoder(feat_dim, emb_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def forward(self, music_feat, motion_feat):
        me = self.music(music_feat)
        de = self.motion(motion_feat)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * me @ de.t(), me, de


def make_weak_music_features_from_motion(desc: np.ndarray, noise: float = 0.08) -> np.ndarray:
    # Weak-pair fallback: retain duration/energy/turn/contact semantics with jitter.
    m = desc.copy().astype(np.float32)
    rng = np.random.default_rng(1234)
    m[:, 1:9] *= rng.normal(1.0, noise, size=m[:, 1:9].shape).astype(np.float32)
    m[:, 15:18] *= rng.normal(1.0, noise * 1.5, size=m[:, 15:18].shape).astype(np.float32)
    m += rng.normal(0.0, noise * 0.15, size=m.shape).astype(np.float32)
    return m.astype(np.float32)


def load_db(db_path: str | Path) -> dict:
    data = np.load(db_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def train_contrastive(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for V44 training.")
    cfg = V46Config.from_json(args.config).apply_env()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    db = load_db(args.db)
    motion = np.asarray(db["desc_z"], dtype=np.float32)
    if args.music_feature_npz and Path(args.music_feature_npz).exists():
        mf = np.load(args.music_feature_npz)["music"].astype(np.float32)
        if mf.shape != motion.shape:
            raise ValueError(f"music feature shape {mf.shape} != motion feature shape {motion.shape}")
        music = mf
    else:
        music_raw = make_weak_music_features_from_motion(np.asarray(db["desc"], dtype=np.float32))
        mean = np.asarray(db["desc_mean"], dtype=np.float32)
        std = np.asarray(db["desc_std"], dtype=np.float32)
        music = (music_raw - mean) / std
    N = motion.shape[0]
    device = torch.device(cfg.device)
    model = ContrastiveModel(motion.shape[1], cfg.embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    bs = min(cfg.batch_size, N)
    for ep in range(int(args.epochs or cfg.contrastive_epochs)):
        perm = np.random.permutation(N)
        losses = []
        for st in range(0, N, bs):
            idx = perm[st:st + bs]
            if len(idx) < 2:
                continue
            mf = torch.from_numpy(music[idx]).float().to(device)
            df = torch.from_numpy(motion[idx]).float().to(device)
            logits, _, _ = model(mf, df)
            labels = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep % 10 == 0 or ep == int(args.epochs or cfg.contrastive_epochs) - 1:
            print(f"[V44 contrastive] epoch={ep} loss={np.mean(losses):.5f} scale={model.logit_scale.exp().item():.2f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"version": "v44_contrastive", "state_dict": model.state_dict(), "config": dataclasses.asdict(cfg), "feat_dim": motion.shape[1], "embed_dim": cfg.embed_dim}
    torch.save(ckpt, out)
    print(json.dumps({"contrastive_ckpt": str(out), "num_events": int(N)}, ensure_ascii=False, indent=2))
    return 0


class TemporalRefiner(nn.Module):
    def __init__(self, motion_dim: int = EDGE_DIM, cond_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.in_proj = nn.Conv1d(motion_dim + cond_dim + 1, hidden, 1)
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU(),
        )
        self.out = nn.Conv1d(hidden, motion_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, cond, seam_mask):
        # x: B,T,D cond: B,C seam_mask: B,T,1
        B, T, D = x.shape
        c = cond[:, None, :].expand(B, T, cond.shape[-1])
        y = torch.cat([x, c, seam_mask], dim=-1).transpose(1, 2)
        h = self.in_proj(y)
        h = h + self.net(h)
        delta = self.out(h).transpose(1, 2)
        return delta


def sample_motion_window(paths: np.ndarray, target_len: int) -> np.ndarray:
    p = str(random.choice(paths.tolist()))
    m = np.load(p).astype(np.float32)
    if m.shape[0] == target_len:
        return m
    if m.shape[0] > target_len:
        st = random.randint(0, m.shape[0] - target_len)
        return m[st:st + target_len]
    return resample_motion_np(m, target_len)


def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    x = clean.copy()
    T, D = x.shape
    seam = np.zeros((T, 1), dtype=np.float32)
    if T > 30:
        c = random.randint(T // 4, 3 * T // 4)
        w = random.randint(6, min(20, T // 5))
        seam[max(0, c - w): min(T, c + w)] = 1.0
        offset = np.zeros(D, dtype=np.float32)
        offset[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity, size=3)
        offset[ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.35, size=ROT6D_END - ROT6D_START)
        x[c:] += offset
        if c + 1 < T:
            x[c:] += np.linspace(0, 1, T - c)[:, None] * np.random.normal(0, severity * 0.5, size=(1, D))
    x += np.random.normal(0, severity * 0.15, size=x.shape).astype(np.float32)
    return x.astype(np.float32), seam


def train_refiner(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for V45 training.")
    cfg = V46Config.from_json(args.config).apply_env()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    db = load_db(args.db)
    paths = db["paths"]
    desc_mean = np.asarray(db["desc_z"], dtype=np.float32).mean(axis=0)
    device = torch.device(cfg.device)
    model = TemporalRefiner(EDGE_DIM, 32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = int(args.steps or cfg.refiner_train_steps)
    bs = min(cfg.batch_size, max(2, len(paths)))
    for step in range(steps):
        clean_batch = []
        bad_batch = []
        seam_batch = []
        cond_batch = []
        for _ in range(bs):
            clean = sample_motion_window(paths, cfg.window_len)
            bad, seam = degrade_for_refiner(clean)
            clean_batch.append(clean)
            bad_batch.append(bad)
            seam_batch.append(seam)
            cond_batch.append(desc_mean)
        clean_t = torch.from_numpy(np.stack(clean_batch)).float().to(device)
        bad_t = torch.from_numpy(np.stack(bad_batch)).float().to(device)
        seam_t = torch.from_numpy(np.stack(seam_batch)).float().to(device)
        cond_t = torch.from_numpy(np.stack(cond_batch)).float().to(device)
        delta = model(bad_t, cond_t, seam_t)
        pred = bad_t + delta * (0.35 + 0.65 * seam_t)
        rec = F.smooth_l1_loss(pred, clean_t)
        smooth = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], clean_t[:, 1:] - clean_t[:, :-1])
        loss = rec + 0.25 * smooth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"[V45 refiner] step={step} loss={loss.item():.6f} rec={rec.item():.6f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": "v45_refiner", "state_dict": model.state_dict(), "config": dataclasses.asdict(cfg)}, out)
    print(json.dumps({"refiner_ckpt": str(out), "steps": steps}, ensure_ascii=False, indent=2))
    return 0


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device).float() * (-math.log(10000.0) / max(half - 1, 1)))
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class DiffusionDenoiser(nn.Module):
    def __init__(self, motion_dim: int = EDGE_DIM, cond_dim: int = 32, hidden: int = 256, time_dim: int = 128):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim + time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.in_proj = nn.Conv1d(motion_dim * 2 + 1, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU()),
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GroupNorm(8, hidden), nn.SiLU()),
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.GroupNorm(8, hidden), nn.SiLU()),
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU()),
        ])
        self.out = nn.Conv1d(hidden, motion_dim, 1)

    def forward(self, x_t, retrieval, cond, seam_mask, t):
        B, T, D = x_t.shape
        inp = torch.cat([x_t, retrieval, seam_mask], dim=-1).transpose(1, 2)
        h = self.in_proj(inp)
        te = self.time(t)
        ce = self.cond_proj(torch.cat([cond, te], dim=-1))[:, :, None]
        h = h + ce
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h).transpose(1, 2)


def make_beta_schedule(n: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, n, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


def train_diffusion(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for V46 diffusion training.")
    cfg = V46Config.from_json(args.config).apply_env()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    db = load_db(args.db)
    paths = db["paths"]
    desc_z = np.asarray(db["desc_z"], dtype=np.float32)
    device = torch.device(cfg.device)
    model = DiffusionDenoiser(EDGE_DIM, 32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = int(args.steps or cfg.diffusion_train_steps)
    Tdiff = int(args.diffusion_steps or cfg.diffusion_steps)
    _, _, abar = make_beta_schedule(Tdiff, device)
    bs = min(cfg.batch_size, max(2, len(paths)))
    for step in range(steps):
        clean_batch = []
        retr_batch = []
        seam_batch = []
        cond_batch = []
        for _ in range(bs):
            idx = random.randrange(len(paths))
            clean = np.load(str(paths[idx])).astype(np.float32)
            clean = resample_motion_np(clean, cfg.window_len)
            retr, seam = degrade_for_refiner(clean, severity=0.045)
            clean_batch.append(clean)
            retr_batch.append(retr)
            seam_batch.append(seam)
            cond_batch.append(desc_z[idx])
        x0 = torch.from_numpy(np.stack(clean_batch)).float().to(device)
        retr = torch.from_numpy(np.stack(retr_batch)).float().to(device)
        seam = torch.from_numpy(np.stack(seam_batch)).float().to(device)
        cond = torch.from_numpy(np.stack(cond_batch)).float().to(device)
        t = torch.randint(0, Tdiff, (bs,), device=device)
        noise = torch.randn_like(x0)
        a = abar[t].view(bs, 1, 1)
        x_t = torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise
        pred_noise = model(x_t, retr, cond, seam, t)
        loss_noise = F.mse_loss(pred_noise, noise)
        # Encourage denoised sample to stay close to motion manifold.
        x0_hat = (x_t - torch.sqrt(1.0 - a) * pred_noise) / torch.sqrt(a).clamp_min(1e-6)
        loss_vel = F.smooth_l1_loss(x0_hat[:, 1:] - x0_hat[:, :-1], x0[:, 1:] - x0[:, :-1])
        loss = loss_noise + 0.10 * loss_vel
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            print(f"[V46 diffusion] step={step} loss={loss.item():.6f} noise={loss_noise.item():.6f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": "v46_conditional_residual_diffusion", "state_dict": model.state_dict(), "config": dataclasses.asdict(cfg), "diffusion_steps": Tdiff}, out)
    print(json.dumps({"diffusion_ckpt": str(out), "steps": steps}, ensure_ascii=False, indent=2))
    return 0


def load_contrastive(path: Optional[str], cfg: V46Config):
    if torch is None or not path or not Path(path).exists():
        return None
    ckpt = torch.load(path, map_location=cfg.device)
    model = ContrastiveModel(ckpt.get("feat_dim", 32), ckpt.get("embed_dim", cfg.embed_dim)).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def embed_with_contrastive(model, music_feat_z: np.ndarray, motion_feat_z: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, np.ndarray]:
    if model is None or torch is None:
        mz = music_feat_z / np.maximum(np.linalg.norm(music_feat_z, axis=-1, keepdims=True), 1e-8)
        dz = motion_feat_z / np.maximum(np.linalg.norm(motion_feat_z, axis=-1, keepdims=True), 1e-8)
        return mz.astype(np.float32), dz.astype(np.float32)
    with torch.no_grad():
        mf = torch.from_numpy(music_feat_z.astype(np.float32)).to(cfg.device)
        df = torch.from_numpy(motion_feat_z.astype(np.float32)).to(cfg.device)
        me = model.music(mf).detach().cpu().numpy()
        de = model.motion(df).detach().cpu().numpy()
    return me.astype(np.float32), de.astype(np.float32)


def transition_cost(exit_state: np.ndarray, entry_state: np.ndarray, cexit: np.ndarray, centry: np.ndarray) -> float:
    pose_exit = exit_state[: NUM_JOINTS * 3]
    vel_exit = exit_state[NUM_JOINTS * 3 :]
    pose_entry = entry_state[: NUM_JOINTS * 3]
    vel_entry = entry_state[NUM_JOINTS * 3 :]
    pose = float(np.mean((pose_exit - pose_entry) ** 2))
    vel = float(np.mean((vel_exit - vel_entry) ** 2))
    contact = float(np.mean(np.abs(cexit - centry)))
    return pose * 0.8 + vel * 1.6 + contact * 0.12


def retrieve_schedule(slots: List[dict], slot_feat: np.ndarray, db: dict, cfg: V46Config, contrastive=None) -> Tuple[List[int], List[dict]]:
    desc = np.asarray(db["desc"], dtype=np.float32)
    desc_z = np.asarray(db["desc_z"], dtype=np.float32)
    mean = np.asarray(db["desc_mean"], dtype=np.float32)
    std = np.asarray(db["desc_std"], dtype=np.float32)
    music_z = (slot_feat - mean) / std
    music_emb, motion_emb = embed_with_contrastive(contrastive, music_z, desc_z, cfg)
    sources = np.asarray(db["source_groups"], dtype=object)
    durations = np.asarray(db["durations"], dtype=np.float32)
    entries = np.asarray(db["entry"], dtype=np.float32)
    exits = np.asarray(db["exit"], dtype=np.float32)
    centry = np.asarray(db["contact_entry"], dtype=np.float32)
    cexit = np.asarray(db["contact_exit"], dtype=np.float32)

    beams: List[Tuple[float, List[int], Dict[str, int]]] = [(0.0, [], {})]
    reports: List[dict] = []
    for i, slot in enumerate(slots):
        sim = music_emb[i] @ motion_emb.T
        dur_cost = np.abs(np.log(np.maximum(durations, 1e-4) / max(float(slot["duration"]), 1e-4)))
        base_score = sim - cfg.retrieval_warp_penalty * dur_cost
        cand = np.argsort(-base_score)[: max(cfg.top_k, cfg.beam_size)].tolist()
        new_beams: List[Tuple[float, List[int], Dict[str, int]]] = []
        for score, path, src_counts in beams:
            prev = path[-1] if path else None
            for idx in cand:
                sc = float(base_score[idx])
                src = str(sources[idx])
                sc -= cfg.retrieval_source_penalty * src_counts.get(src, 0)
                if prev is not None:
                    sc -= cfg.retrieval_transition_penalty * transition_cost(exits[prev], entries[idx], cexit[prev], centry[idx])
                    if src == str(sources[prev]):
                        sc -= cfg.retrieval_repeat_penalty
                ns = dict(src_counts)
                ns[src] = ns.get(src, 0) + 1
                new_beams.append((score + sc, path + [int(idx)], ns))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[: cfg.beam_size]
        reports.append({"slot": i, "start": slot.get("start"), "end": slot.get("end"), "top_candidate": int(cand[0]), "beam_best_score": float(beams[0][0])})
    return beams[0][1], reports


def align_next_to_prev(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    out = nxt.copy()
    delta = prev[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += delta[0]
    out[:, ROOT_Z_IDX] += delta[1]
    # Soft root-y adjustment only; do not force same height completely.
    dy = prev[-1, ROOT_Y_IDX] - out[0, ROOT_Y_IDX]
    ramp = np.linspace(1.0, 0.0, min(18, len(out)), dtype=np.float32)
    out[: len(ramp), ROOT_Y_IDX] += dy * ramp
    return out


def concat_events(event_paths: Sequence[str], target_durations: Sequence[float], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:
    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        m = np.load(str(p)).astype(np.float32)
        target_len = max(cfg.min_event_frames, int(round(float(dur) * cfg.fps)))
        warp = target_len / max(1, m.shape[0])
        m = resample_motion_np(m, target_len)
        if pieces:
            m = align_next_to_prev(pieces[-1], m)
            ov = min(cfg.overlap, len(pieces[-1]) // 3, len(m) // 3)
            if ov > 0:
                a = pieces[-1][-ov:].copy()
                b = m[:ov].copy()
                w = np.linspace(0, 1, ov, dtype=np.float32)[:, None]
                blend = (1 - w) * a + w * b
                pieces[-1] = np.concatenate([pieces[-1][:-ov], blend], axis=0)
                m = m[ov:]
        pieces.append(m)
        rep.append({"path": str(p), "target_frames": int(target_len), "warp": float(warp)})
    return np.concatenate(pieces, axis=0).astype(np.float32), rep


def make_boundary_mask(T: int, seams: Sequence[int], width: int = 18) -> np.ndarray:
    mask = np.zeros((T, 1), dtype=np.float32)
    for s in seams:
        a = max(0, int(s) - width)
        b = min(T, int(s) + width)
        mask[a:b, 0] = 1.0
    return mask


def analytic_residual_refine(motion: np.ndarray, seam_positions: Sequence[int], width: int = 24) -> np.ndarray:
    out = motion.copy().astype(np.float32)
    for s in seam_positions:
        a = max(0, s - width)
        b = min(len(out), s + width)
        if b - a < 4:
            continue
        left = out[a].copy()
        right = out[b - 1].copy()
        x = np.linspace(0, 1, b - a, dtype=np.float32)[:, None]
        cubic = x * x * (3 - 2 * x)
        bridge = (1 - cubic) * left + cubic * right
        # Only blend root and rotations near boundary; keep original high-frequency content.
        w = np.sin(np.linspace(0, math.pi, b - a, dtype=np.float32))[:, None] ** 2
        idx = list(range(ROOT_X_IDX, ROOT_Z_IDX + 1)) + list(range(ROT6D_START, ROT6D_END))
        out[a:b, idx] = (1 - 0.35 * w) * out[a:b, idx] + 0.35 * w * bridge[:, idx]
    out[:, ROOT_Y_IDX] = smooth_np(out[:, ROOT_Y_IDX:ROOT_Y_IDX + 1], 1.0)[:, 0]
    return out.astype(np.float32)


def apply_refiner_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        seams = np.where(seam_mask[:, 0] > 0.5)[0]
        seam_centers = []
        for a, b in contiguous_regions(seam_mask[:, 0] > 0.5):
            seam_centers.append((a + b) // 2)
        return analytic_residual_refine(motion, seam_centers)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model = TemporalRefiner(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    chunks = []
    with torch.no_grad():
        for st in range(0, motion.shape[0], cfg.window_len):
            ed = min(motion.shape[0], st + cfg.window_len)
            chunk = motion[st:ed]
            mask = seam_mask[st:ed]
            orig_len = len(chunk)
            if orig_len < cfg.window_len:
                chunk = resample_motion_np(chunk, cfg.window_len)
                mask = resample_motion_np(mask, cfg.window_len)
            x = torch.from_numpy(chunk[None]).float().to(cfg.device)
            c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
            sm = torch.from_numpy(mask[None].astype(np.float32)).float().to(cfg.device)
            delta = model(x, c, sm)
            y = x + delta * (0.2 + 0.8 * sm)
            y_np = y[0].detach().cpu().numpy()
            if orig_len < cfg.window_len:
                y_np = resample_motion_np(y_np, orig_len)
            chunks.append(y_np)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def apply_diffusion_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        return motion
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    Tdiff = int(ckpt.get("diffusion_steps", cfg.diffusion_steps))
    model = DiffusionDenoiser(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    betas, alphas, abar = make_beta_schedule(Tdiff, torch.device(cfg.device))
    outs = []
    with torch.no_grad():
        for st in range(0, motion.shape[0], cfg.window_len):
            ed = min(motion.shape[0], st + cfg.window_len)
            retr_np = motion[st:ed]
            mask_np = seam_mask[st:ed]
            orig_len = len(retr_np)
            if orig_len < cfg.window_len:
                retr_np = resample_motion_np(retr_np, cfg.window_len)
                mask_np = resample_motion_np(mask_np, cfg.window_len)
            retr = torch.from_numpy(retr_np[None]).float().to(cfg.device)
            mask = torch.from_numpy(mask_np[None].astype(np.float32)).float().to(cfg.device)
            c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
            # residual diffusion starts near retrieval instead of pure noise.
            x = retr + 0.03 * torch.randn_like(retr) * (0.25 + 0.75 * mask)
            for ti in reversed(range(Tdiff)):
                t = torch.full((1,), ti, device=cfg.device, dtype=torch.long)
                eps = model(x, retr, c, mask, t)
                beta = betas[ti]
                alpha = alphas[ti]
                ab = abar[ti]
                mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - ab).clamp_min(1e-6) * eps)
                if ti > 0:
                    x = mean + torch.sqrt(beta) * torch.randn_like(x) * 0.35
                else:
                    x = mean
                # keep non-boundary close to retrieved path, let seams regenerate.
                x = retr * (1.0 - 0.65 * mask) + x * (0.65 * mask)
            y = x[0].detach().cpu().numpy()
            if orig_len < cfg.window_len:
                y = resample_motion_np(y, orig_len)
            outs.append(y)
    return np.concatenate(outs, axis=0).astype(np.float32)


def derive_contacts_np(motion: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    joints = fk_24_np(motion)
    foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
    foot_y = foot[..., 1]
    floor_y = float(np.percentile(foot_y.reshape(-1), 5))
    vel = np.zeros(foot.shape[:2], dtype=np.float32)
    vel[1:] = np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
    height_score = np.clip(1.0 - (foot_y - floor_y) / max(cfg.ik_height_margin, 1e-6), 0.0, 1.0)
    speed_score = np.clip(1.0 - vel / max(cfg.ik_speed_gate_mpf, 1e-6), 0.0, 1.0)
    conf = 0.62 * height_score + 0.38 * speed_score
    clean = np.zeros_like(conf, dtype=bool)
    for f in range(conf.shape[1]):
        state = False
        for t, p in enumerate(conf[:, f]):
            if p >= cfg.ik_contact_high:
                state = True
            elif p <= cfg.ik_contact_low:
                state = False
            clean[t, f] = state
        clean[:, f] = median_bool_filter(clean[:, f], 5)
    return clean, conf.astype(np.float32), floor_y, foot.astype(np.float32)


def c1_hanning_window01(phase: np.ndarray | float) -> np.ndarray | float:
    """C1-safe 0->1->0 window. Value and first derivative are zero at both ends."""
    ph = np.clip(np.asarray(phase, dtype=np.float32), 0.0, 1.0)
    out = np.sin(np.pi * ph) ** 2
    if np.isscalar(phase):
        return float(out)
    return out.astype(np.float32)


def smoothstep01(x: np.ndarray | float) -> np.ndarray | float:
    y = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    out = y * y * (3.0 - 2.0 * y)
    if np.isscalar(x):
        return float(out)
    return out.astype(np.float32)


def apply_root_y_c1_physics_np(motion: np.ndarray, contacts: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:
    """
    V46.1 root-Y safety pass.

    Fixes two common post-process artifacts:
    1) Damping Snap Bug: landing damping length is the real contact duration,
       so the damping dip is exactly zero before the next flight frame.
    2) C1 Discontinuity: flight/parabola blending uses a Hanning/sin^2 gate,
       whose first derivative is zero at takeoff and landing boundaries.

    Only the legal EDGE root-Y channel is edited here; lower-body IK later writes
    legal lower-body rot6d channels.
    """
    out = motion.copy().astype(np.float32)
    if not bool(cfg.root_y_physics_enable) or out.shape[0] < 4:
        return out, {"enabled": False, "reason": "disabled_or_too_short"}

    root_y0 = out[:, ROOT_Y_IDX].copy()
    any_contact = contacts.any(axis=1)
    is_flight = ~any_contact

    flight_applied = 0
    for start, end in contiguous_regions(is_flight):
        n = end - start
        if n < int(cfg.root_y_min_flight_frames):
            continue
        left = max(0, start - 1)
        right = min(len(root_y0) - 1, end)
        if right <= left:
            continue
        y0 = float(root_y0[left])
        y1 = float(root_y0[right])
        duration = max((right - left) / max(float(cfg.fps), 1e-6), 1.0 / max(float(cfg.fps), 1e-6))
        v0 = (y1 - y0 + 0.5 * 9.81 * duration * duration) / duration
        for k, ti in enumerate(range(start, end)):
            # Use exact endpoint phases for zero-value/zero-slope blend.
            phase = 0.0 if n <= 1 else k / float(n - 1)
            gate = float(cfg.root_y_flight_strength) * float(c1_hanning_window01(phase))
            tau = (ti - left) / max(float(cfg.fps), 1e-6)
            parabola = y0 + v0 * tau - 0.5 * 9.81 * tau * tau
            out[ti, ROOT_Y_IDX] = (1.0 - gate) * out[ti, ROOT_Y_IDX] + gate * parabola
        flight_applied += 1

    damping_applied = 0
    damping_preview: List[Dict[str, object]] = []
    for start, end in contiguous_regions(any_contact):
        # Only damp actual landings: a contact island that follows flight.
        if start <= 0 or any_contact[start - 1]:
            continue
        n = end - start
        if n <= 2:
            continue
        # Critical fix: dynamic damping duration is exactly this contact island.
        # No break-on-flight is needed, and the last contact frame has zero dip.
        max_abs_dip = 0.0
        for k, ti in enumerate(range(start, end)):
            phase = k / float(max(n - 1, 1))
            gate = float(c1_hanning_window01(phase))
            # Decay shapes the dip toward early landing but remains exactly zero at both ends.
            dip = float(cfg.root_y_damping_max_dip) * math.exp(-4.0 * phase) * gate
            out[ti, ROOT_Y_IDX] -= dip
            max_abs_dip = max(max_abs_dip, abs(dip))
        damping_applied += 1
        if len(damping_preview) < 24:
            damping_preview.append({"start": int(start), "end": int(end), "frames": int(n), "max_dip_m": float(max_abs_dip), "dynamic_duration": True})

    delta = out[:, ROOT_Y_IDX] - root_y0
    return out.astype(np.float32), {
        "enabled": True,
        "version": "v46_1_c1_dynamic_root_y_physics",
        "fixes_damping_snap": True,
        "fixes_c1_discontinuity": True,
        "flight_gate": "hanning_sin_squared",
        "damping_duration": "exact_contact_island_length",
        "flight_segments_applied": int(flight_applied),
        "landing_damping_applied": int(damping_applied),
        "damping_preview": damping_preview,
        "delta_mean": float(delta.mean()),
        "delta_p95_abs": float(np.percentile(np.abs(delta), 95)),
        "delta_max_abs": float(np.max(np.abs(delta))),
    }


def generate_ik_targets_np(native_foot: np.ndarray, contacts: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:
    """
    Generate V43 lower-body IK targets with an intentional-slide release guard.

    If a foot remains in contact but travels more than cfg.ik_slide_release_m in
    XZ during the contact island, this is treated as cloud-step / designed slide,
    not as foot skating. The target stays native, preventing IK from stretching
    the leg and crushing the knee.
    """
    targets = native_foot.copy().astype(np.float32)
    locked_segments = 0
    released_slide_segments = 0
    skipped_short = 0
    preview: List[Dict[str, object]] = []

    for f in range(native_foot.shape[1]):
        for start, end in contiguous_regions(contacts[:, f]):
            length = end - start
            if length < 3:
                skipped_short += 1
                continue

            seg_xz = native_foot[start:end, f, [0, 2]].astype(np.float32)
            span = float(np.linalg.norm(seg_xz.max(axis=0) - seg_xz.min(axis=0))) if length > 1 else 0.0
            step = np.linalg.norm(seg_xz[1:] - seg_xz[:-1], axis=-1) if length > 1 else np.zeros((0,), dtype=np.float32)
            arc = float(step.sum())
            smoothness = float(span / max(arc, 1e-6)) if arc > 1e-6 else 1.0

            if length >= int(cfg.ik_slide_release_min_frames) and span > float(cfg.ik_slide_release_m):
                # Intentional slide/cloud-step: do not anchor. Native foot trajectory is the target.
                released_slide_segments += 1
                if len(preview) < 32:
                    preview.append({
                        "foot": int(f), "start": int(start), "end": int(end),
                        "frames": int(length), "mode": "released_intentional_slide",
                        "xz_span_m": span, "arc_m": arc, "smoothness": smoothness,
                        "threshold_m": float(cfg.ik_slide_release_m),
                    })
                continue

            # Contact-internal anchor only; never sample pre-contact flight.
            anchor_end = min(start + 3, end)
            anchor = native_foot[start:anchor_end, f].mean(axis=0)
            locked_segments += 1
            for k, t in enumerate(range(start, end)):
                phase_in = min(1.0, k / 6.0)
                phase_out = min(1.0, (end - 1 - t) / 6.0)
                w = min(float(smoothstep01(phase_in)), float(smoothstep01(phase_out)))
                targets[t, f] = (1 - w) * native_foot[t, f] + w * anchor
            if len(preview) < 32:
                preview.append({
                    "foot": int(f), "start": int(start), "end": int(end),
                    "frames": int(length), "mode": "locked_footplant",
                    "xz_span_m": span, "arc_m": arc, "smoothness": smoothness,
                    "anchor_source": "contact_internal_first_frames",
                })

    diff = np.linalg.norm(targets - native_foot, axis=-1)
    non_contact = ~contacts
    meta = {
        "version": "v46_1_slide_safe_ik_target_generator",
        "intentional_slide_guard": True,
        "slide_release_threshold_m": float(cfg.ik_slide_release_m),
        "locked_segments": int(locked_segments),
        "released_slide_segments": int(released_slide_segments),
        "skipped_short_segments": int(skipped_short),
        "non_contact_diff_max": float(diff[non_contact].max()) if non_contact.any() else 0.0,
        "contact_diff_p95": float(np.percentile(diff[contacts], 95)) if contacts.any() else 0.0,
        "preview": preview,
    }
    return targets.astype(np.float32), meta


def true_lower_body_ik(motion: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:
    if torch is None:
        return motion, {"enabled": False, "reason": "torch_unavailable"}
    contacts0, _, _, _ = derive_contacts_np(motion, cfg)
    motion_base, root_y_report = apply_root_y_c1_physics_np(motion, contacts0, cfg)
    contacts, conf, floor_y, native_foot = derive_contacts_np(motion_base, cfg)
    targets, target_meta = generate_ik_targets_np(native_foot, contacts, cfg)
    device = torch.device(cfg.device)
    out_all = motion_base.copy().astype(np.float32)
    reports = []
    T = motion.shape[0]
    chunk = int(cfg.ik_chunk)
    # Use overlap to avoid chunk seams.
    starts = list(range(0, T, max(1, chunk - 24)))
    for st in starts:
        ed = min(T, st + chunk)
        if ed - st < 4:
            continue
        base_np = out_all[st:ed].copy()
        L = base_np.shape[0]
        base = torch.from_numpy(base_np).float().to(device)
        rot_full = base[:, ROT6D_START:ROT6D_END].reshape(L, NUM_JOINTS, 6).detach().clone()
        root = base[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].detach().clone().requires_grad_(True)
        lower_idx = torch.as_tensor(LOWER_BODY_JOINTS, device=device, dtype=torch.long)
        lower_rot = rot_full[:, lower_idx].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([lower_rot, root], lr=cfg.ik_lr)
        target = torch.from_numpy(targets[st:ed]).float().to(device)
        contact = torch.from_numpy(contacts[st:ed].astype(np.float32)).float().to(device)
        confidence = torch.from_numpy(conf[st:ed]).float().to(device)
        floor = torch.tensor(floor_y, device=device, dtype=torch.float32)
        base_rot = rot_full[:, lower_idx].detach().clone()
        base_root = root.detach().clone()
        best_loss = float("inf")
        best_motion = None
        for it in range(int(cfg.ik_iters)):
            rr = project_rot6d_torch(lower_rot)
            rr = base_rot + torch.clamp(rr - base_rot, -cfg.ik_max_delta_rot, cfg.ik_max_delta_rot)
            rot = rot_full.clone()
            rot[:, lower_idx] = rr
            mm = base.clone()
            mm[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root
            mm[:, ROT6D_START:ROT6D_END] = rot.reshape(L, -1)
            joints = fk_24_torch(mm)
            foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
            w = (contact * confidence).unsqueeze(-1)
            foot_loss = ((foot - target) ** 2 * w).sum() / w.sum().clamp_min(1.0)
            pose_loss = F.smooth_l1_loss(rr, base_rot)
            if L > 1:
                vel_loss = F.smooth_l1_loss(rr[1:] - rr[:-1], base_rot[1:] - base_rot[:-1])
                root_vel = F.smooth_l1_loss(root[1:] - root[:-1], base_root[1:] - base_root[:-1])
            else:
                vel_loss = torch.tensor(0.0, device=device)
                root_vel = torch.tensor(0.0, device=device)
            pen = F.relu(floor + 0.003 - foot[..., 1]).pow(2).mean()
            root_loss = F.smooth_l1_loss(root, base_root) + root_vel
            loss = cfg.ik_contact_w * foot_loss + cfg.ik_pose_w * pose_loss + cfg.ik_temporal_w * vel_loss + cfg.ik_root_w * root_loss + cfg.ik_penetration_w * pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([lower_rot, root], 1.0)
            opt.step()
            if float(loss.detach().cpu()) < best_loss:
                best_loss = float(loss.detach().cpu())
                best_motion = mm.detach().cpu().numpy()
        if best_motion is not None:
            if st > 0:
                # feather first overlap if previous chunk already wrote it.
                ov = min(12, L, st)
                w = np.linspace(0, 1, ov, dtype=np.float32)[:, None]
                old = out_all[st:st + ov]
                out_all[st:st + ov] = (1 - w) * old + w * best_motion[:ov]
                out_all[st + ov:ed] = best_motion[ov:]
            else:
                out_all[st:ed] = best_motion
        reports.append({"start": int(st), "end": int(ed), "best_loss": float(best_loss), "contact_ratio": float(contacts[st:ed].mean())})
    # Re-orthogonalize all rotation channels after optimization.
    if torch is not None:
        with torch.no_grad():
            x = torch.from_numpy(out_all[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)).float()
            out_all[:, ROT6D_START:ROT6D_END] = project_rot6d_torch(x).numpy().reshape(T, -1)
    audit_before = audit_motion_np(motion, cfg)
    audit_after = audit_motion_np(out_all, cfg)
    # Rollback only if IK makes both skate and jerk worse; otherwise preserve actual IK.
    rollback = False
    if audit_after["foot_skate_p95_mpf"] > audit_before["foot_skate_p95_mpf"] * 1.25 and audit_after["mean_joint_jerk_p95"] > audit_before["mean_joint_jerk_p95"] * 1.25:
        rollback = True
        final = motion.copy()
    else:
        final = out_all
    report = {
        "version": "v46_1_true_lower_body_ik_slide_safe_c1_root_y",
        "enabled": True,
        "writes_lower_body_rot6d": True,
        "root_y_physics": root_y_report,
        "ik_target_generator": target_meta,
        "lower_body_joints": list(map(int, LOWER_BODY_JOINTS)),
        "foot_joint_ids": list(map(int, DEFAULT_FOOT_JOINTS)),
        "floor_y": float(floor_y),
        "contact_ratio": float(contacts.mean()),
        "chunks": reports,
        "audit_before": audit_before,
        "audit_after_candidate": audit_after,
        "rollback_triggered": rollback,
        "audit_final": audit_motion_np(final, cfg),
    }
    return final.astype(np.float32), report


def audit_motion_np(motion: np.ndarray, cfg: Optional[V46Config] = None) -> dict:
    cfg = cfg or V46Config()
    joints = fk_24_np(motion)
    foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
    foot_y = foot[..., 1]
    floor_y = float(np.percentile(foot_y.reshape(-1), 5))
    vel = np.zeros(foot.shape[:2], dtype=np.float32)
    vel[1:] = np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
    contacts, _, _, _ = derive_contacts_np(motion, cfg)
    skate = vel[contacts] if contacts.any() else vel.reshape(-1)
    if joints.shape[0] >= 4:
        jerk = np.diff(joints, n=3, axis=0)
        jerk_frame = np.linalg.norm(jerk, axis=-1).mean(axis=-1)
        jerk_p95 = float(np.percentile(jerk_frame, 95))
        jerk_max = float(np.max(jerk_frame))
    else:
        jerk_p95 = 0.0
        jerk_max = 0.0
    return {
        "frames": int(motion.shape[0]),
        "floor_y": floor_y,
        "contact_ratio": float(contacts.mean()),
        "foot_skate_mean_mpf": float(np.mean(skate)) if skate.size else 0.0,
        "foot_skate_p95_mpf": float(np.percentile(skate, 95)) if skate.size else 0.0,
        "foot_skate_max_mpf": float(np.max(skate)) if skate.size else 0.0,
        "foot_penetration_min_m": float(np.min(foot_y - floor_y)),
        "mean_joint_jerk_p95": jerk_p95,
        "mean_joint_jerk_max": jerk_max,
        "root_y_range_m": float(np.max(motion[:, ROOT_Y_IDX]) - np.min(motion[:, ROOT_Y_IDX])),
    }


def render_if_possible(motion_path: str, audio_path: Optional[str], output_mp4: Optional[str], render_script: str = "render_from_npy.py") -> None:
    if not output_mp4 or not audio_path:
        return
    if not Path(render_script).exists() or not Path(audio_path).exists():
        print("[V46 WARN] render skipped: render script or audio missing", file=sys.stderr)
        return
    cmd = [sys.executable, render_script, "--motion", motion_path, "--audio", audio_path, "--output", output_mp4, "--camera_mode", "follow", "--render_smooth_window", "5"]
    print("[V46 RENDER]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def generate(args: argparse.Namespace) -> int:
    cfg = V46Config.from_json(args.config).apply_env()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch is not None:
        torch.manual_seed(cfg.seed)
    db = load_db(args.db)
    contrastive = load_contrastive(args.contrastive, cfg)
    slots, slot_feat = audio_slots(args.audio, cfg, args.slot_seconds, args.slots_json)
    path_idx, retrieval_report = retrieve_schedule(slots, slot_feat, db, cfg, contrastive)
    paths = np.asarray(db["paths"], dtype=object)
    selected_paths = [str(paths[i]) for i in path_idx]
    motion, concat_report = concat_events(selected_paths, [s["duration"] for s in slots], cfg)
    seam_positions = []
    acc = 0
    for r in concat_report[:-1]:
        acc += int(r["target_frames"] - min(cfg.overlap, r["target_frames"] // 3))
        seam_positions.append(acc)
    seam_mask = make_boundary_mask(motion.shape[0], seam_positions, width=24)
    cond = np.mean(slot_feat, axis=0).astype(np.float32)
    # Normalize cond using database stats if available.
    cond = (cond - np.asarray(db["desc_mean"], dtype=np.float32)[0]) / np.asarray(db["desc_std"], dtype=np.float32)[0]
    stage_reports = {"retrieval": retrieval_report, "concat": concat_report, "seams": seam_positions}
    pre_audit = audit_motion_np(motion, cfg)
    if cfg.refiner_enable:
        motion = apply_refiner_model(motion, cond, seam_mask, args.refiner, cfg)
        stage_reports["v45_refiner_audit"] = audit_motion_np(motion, cfg)
    if cfg.diffusion_enable:
        motion = apply_diffusion_model(motion, cond, seam_mask, args.diffusion, cfg)
        stage_reports["v46_diffusion_audit"] = audit_motion_np(motion, cfg)
    ik_report = {"enabled": False}
    if cfg.ik_enable:
        motion, ik_report = true_lower_body_ik(motion, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, motion.astype(np.float32))
    report = {
        "version": "v46_motionrag_diff",
        "audio": args.audio,
        "db": args.db,
        "config": dataclasses.asdict(cfg),
        "fk_tree_source": FK_TREE_SOURCE,
        "selected_event_indices": path_idx,
        "selected_event_paths": selected_paths,
        "slots": slots,
        "pre_refine_audit": pre_audit,
        "stage_reports": stage_reports,
        "v43_true_ik": ik_report,
        "final_audit": audit_motion_np(motion, cfg),
    }
    json_path = args.json or str(out).replace(".npy", ".v46_report.json")
    save_json(report, json_path)
    if args.render_output:
        render_if_possible(str(out), args.audio, args.render_output, args.render_script)
    print(json.dumps({"motion": str(out), "json": json_path, "frames": int(motion.shape[0]), "final_audit": report["final_audit"]}, ensure_ascii=False, indent=2))
    return 0


def run_ik(args: argparse.Namespace) -> int:
    cfg = V46Config.from_json(args.config).apply_env()
    motion = np.load(args.input).astype(np.float32)
    out_motion, report = true_lower_body_ik(motion, cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out_motion)
    save_json(report, args.json or str(args.output).replace(".npy", ".v43_true_ik.json"))
    print(json.dumps({"output": args.output, "audit_final": report.get("audit_final")}, ensure_ascii=False, indent=2))
    return 0


def run_audit(args: argparse.Namespace) -> int:
    cfg = V46Config.from_json(args.config).apply_env()
    motion = np.load(args.input).astype(np.float32)
    report = audit_motion_np(motion, cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        save_json(report, args.json)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V46 MotionRAG-Diff for EDGE 151D")
    p.add_argument("--config", default="configs/v46_motionrag_diff_config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-db", help="Build source-aware event database from EDGE 151D motion files")
    b.add_argument("--motion_dirs", nargs="+", required=True)
    b.add_argument("--out_db", required=True, help="Output directory, containing events.npz and events/*.npy")
    b.set_defaults(func=build_db)

    c = sub.add_parser("train-contrastive", help="V44 music-motion contrastive training")
    c.add_argument("--db", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--music_feature_npz", default=None)
    c.add_argument("--epochs", type=int, default=None)
    c.set_defaults(func=train_contrastive)

    r = sub.add_parser("train-refiner", help="V45 residual Motion Refiner training")
    r.add_argument("--db", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=None)
    r.set_defaults(func=train_refiner)

    d = sub.add_parser("train-diffusion", help="V46 conditional residual diffusion training")
    d.add_argument("--db", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--steps", type=int, default=None)
    d.add_argument("--diffusion_steps", type=int, default=None)
    d.set_defaults(func=train_diffusion)

    g = sub.add_parser("generate", help="Generate whole-song motion from music via V46 pipeline")
    g.add_argument("--audio", required=True)
    g.add_argument("--slots_json", default=None)
    g.add_argument("--slot_seconds", type=float, default=4.0)
    g.add_argument("--db", required=True)
    g.add_argument("--contrastive", default=None)
    g.add_argument("--refiner", default=None)
    g.add_argument("--diffusion", default=None)
    g.add_argument("--out", required=True)
    g.add_argument("--json", default=None)
    g.add_argument("--render_output", default=None)
    g.add_argument("--render_script", default="render_from_npy.py")
    g.set_defaults(func=generate)

    ik = sub.add_parser("ik", help="Run V43 true lower-body IK on an existing EDGE 151D npy")
    ik.add_argument("--input", required=True)
    ik.add_argument("--output", required=True)
    ik.add_argument("--json", default=None)
    ik.set_defaults(func=run_ik)

    a = sub.add_parser("audit", help="Audit EDGE 151D foot skate, floor penetration and jerk")
    a.add_argument("--input", required=True)
    a.add_argument("--json", default=None)
    a.set_defaults(func=run_audit)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
