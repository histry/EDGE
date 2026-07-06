#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.12 MotionRAG-Diff for EDGE 151D Dunhuang whole-song generation
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
- V46.12 External Classical-Music Semantic Encoder integration plus V46.11 canonical Chang-E semantics: direct Chang-E BVH loading, 210fps-to-30fps resampling, filename/source-aware RAG semantics, external slot-level music semantic labels, unpaired semantic OT, true lower-body IK, residual refiner, conditional diffusion, capped root-Y physics, root-aware sliding anchors, weighted IK chunks, and strict rollback gates.

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
import csv
import dataclasses
import glob
import hashlib
import json
import math
import os
import pickle
import random
import re
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


def _v46_json_safe(x):
    """Make report/meta objects JSON serializable.

    V46.31 hotfix:
    Chang-E semantic ontology may contain Python set values, e.g. aliases.
    events_meta.json must remain writable, so convert sets/numpy/Path safely.
    """
    import dataclasses as _dataclasses
    import numpy as _np
    from pathlib import Path as _Path

    if _dataclasses.is_dataclass(x):
        return _v46_json_safe(_dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _v46_json_safe(v) for k, v in x.items()}
    if isinstance(x, set):
        return sorted([_v46_json_safe(v) for v in x], key=lambda z: str(z))
    if isinstance(x, (list, tuple)):
        return [_v46_json_safe(v) for v in x]
    if isinstance(x, _Path):
        return str(x)
    if isinstance(x, _np.ndarray):
        return _v46_json_safe(x.tolist())
    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_v46_json_safe(obj), f, ensure_ascii=False, indent=2)
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



def _bvh_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """Single-axis right-handed rotation matrix used by BVH Euler channels."""
    a = math.radians(float(angle_deg))
    c, ss = math.cos(a), math.sin(a)
    axis = axis.upper()[0]
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -ss], [0.0, ss, c]], dtype=np.float32)
    if axis == "Y":
        return np.array([[c, 0.0, ss], [0.0, 1.0, 0.0], [-ss, 0.0, c]], dtype=np.float32)
    return np.array([[c, -ss, 0.0], [ss, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _bvh_euler_to_matrix(channels: Sequence[str], values: Sequence[float]) -> np.ndarray:
    """
    Convert BVH local Euler rotation channels to a rotation matrix.

    BVH stores each joint's channels in an explicit order, commonly
    Zrotation/Xrotation/Yrotation. We multiply in that listed order, which is
    the standard practical interpretation for BVH local transforms.
    """
    R = np.eye(3, dtype=np.float32)
    for ch, v in zip(channels, values):
        if ch.lower().endswith("rotation"):
            R = R @ _bvh_rotation_matrix(ch[0], float(v))
    return R.astype(np.float32)


def _norm_joint_name(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _pick_bvh_joint(norm_names: List[str], aliases: Sequence[str], used: set[int], allow_used: bool = False) -> int:
    alias_norm = [_norm_joint_name(a) for a in aliases]
    # First pass: exact or contains match with unused joints.
    for a in alias_norm:
        for i, n in enumerate(norm_names):
            if (allow_used or i not in used) and (n == a or a in n or n in a):
                used.add(i)
                return i
    # Second pass: allow duplicates for non-critical missing end-effectors.
    for a in alias_norm:
        for i, n in enumerate(norm_names):
            if n == a or a in n or n in a:
                return i
    return -1


def _bvh_target_joint_indices(joint_names: Sequence[str]) -> List[int]:
    """
    Map a generic Chang-E/BVH skeleton to the 24-joint EDGE/SMPL-like order.

    This is a pragmatic name-based adapter. It is intended for event-database
    building and retrieval/refinement, not for claiming exact SMPL conversion.
    If a joint is missing, identity rotation is used for that target joint.
    """
    norm = [_norm_joint_name(x) for x in joint_names]
    used: set[int] = set()
    aliases = [
        ["hips", "hip", "pelvis", "root", "mixamorigHips"],
        ["leftupleg", "lefthip", "leftthigh", "lhip", "lthigh"],
        ["rightupleg", "righthip", "rightthigh", "rhip", "rthigh"],
        ["spine", "spine1", "lowerspine", "abdomen"],
        ["leftleg", "leftknee", "leftshin", "lleg", "lknee", "lshin"],
        ["rightleg", "rightknee", "rightshin", "rleg", "rknee", "rshin"],
        ["spine1", "spine2", "chest", "midspine"],
        ["leftfoot", "leftankle", "lfoot", "lankle"],
        ["rightfoot", "rightankle", "rfoot", "rankle"],
        ["spine2", "spine3", "upperchest", "chest", "thorax"],
        ["lefttoe", "lefttoebase", "lefttoeend", "leftball", "ltoe", "leftfoot"],
        ["righttoe", "righttoebase", "righttoeend", "rightball", "rtoe", "rightfoot"],
        ["neck", "neck1"],
        ["leftshoulder", "leftcollar", "leftclavicle", "lshoulder", "lcollar"],
        ["rightshoulder", "rightcollar", "rightclavicle", "rshoulder", "rcollar"],
        ["head", "headtop", "headendeffector"],
        ["leftarm", "leftupperarm", "larm", "lupperarm"],
        ["rightarm", "rightupperarm", "rarm", "rupperarm"],
        ["leftforearm", "leftlowerarm", "leftelbow", "lforearm", "lelbow"],
        ["rightforearm", "rightlowerarm", "rightelbow", "rforearm", "relbow"],
        ["lefthand", "leftwrist", "lhand", "lwrist"],
        ["righthand", "rightwrist", "rhand", "rwrist"],
        ["lefthand", "leftfinger", "leftthumb", "lhand"],
        ["righthand", "rightfinger", "rightthumb", "rhand"],
    ]
    out: List[int] = []
    for target_id, al in enumerate(aliases):
        # Allow duplicate hands/toes if the BVH skeleton has no separate finger/toe joints.
        allow = target_id in {10, 11, 22, 23}
        out.append(_pick_bvh_joint(norm, al, used, allow_used=allow))
    # Last-resort fallback for very small/nonstandard skeletons.
    for i in range(len(out)):
        if out[i] < 0 and i < len(joint_names):
            out[i] = i
    return out


def load_bvh_file(path: str | Path) -> List[np.ndarray]:
    """
    Load a Chang-E-style `.bvh` file and convert it to an EDGE-like 151D array.

    Output convention:
      - motion[:, 4:7] = root XYZ translation
      - motion[:, 7:151] = 24 local joint rotations in 6D form

    BVH files commonly store positions in centimeters. The loader auto-scales
    to meters when skeleton offsets look centimeter-scale. This direct adapter
    is sufficient for source-aware event indexing, retrieval, V45/V46 training,
    and V43 IK; for exact SMPL/EDGE reproduction, a dedicated retargeting stage
    can still be used before build-db.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    motion_line = None
    for i, line in enumerate(text):
        if line.strip().upper() == "MOTION":
            motion_line = i
            break
    if motion_line is None:
        raise ValueError(f"BVH MOTION section not found: {p}")

    joints: List[dict] = []
    stack: List[Optional[int]] = []
    pending_joint: Optional[int] = None
    pending_end = False
    channel_cursor = 0

    for raw in text[:motion_line]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()
        if key in {"ROOT", "JOINT"} and len(parts) >= 2:
            parent = stack[-1] if stack else -1
            if parent is None:
                parent = -1
            joints.append({"name": parts[1], "parent": int(parent), "offset": np.zeros(3, dtype=np.float32), "channels": [], "channel_start": channel_cursor})
            pending_joint = len(joints) - 1
            pending_end = False
        elif key == "END":
            pending_end = True
            pending_joint = None
        elif key == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            elif pending_end:
                stack.append(None)
                pending_end = False
        elif key == "}":
            if stack:
                stack.pop()
        elif key == "OFFSET" and len(parts) >= 4:
            if stack and stack[-1] is not None:
                joints[stack[-1]]["offset"] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
        elif key == "CHANNELS" and len(parts) >= 2:
            if stack and stack[-1] is not None:
                n = int(parts[1])
                ch = parts[2:2 + n]
                joints[stack[-1]]["channels"] = ch
                joints[stack[-1]]["channel_start"] = channel_cursor
                channel_cursor += n

    if not joints or channel_cursor <= 0:
        raise ValueError(f"No BVH joints/channels parsed: {p}")

    frames = None
    frame_time = 1.0 / 30.0
    data_start = None
    for i in range(motion_line + 1, len(text)):
        line = text[i].strip()
        low = line.lower()
        if low.startswith("frames"):
            frames = int(line.replace(":", " ").split()[-1])
        elif low.startswith("frame time"):
            frame_time = float(line.replace(":", " ").split()[-1])
            data_start = i + 1
            break
    if frames is None or data_start is None:
        raise ValueError(f"BVH frame metadata not found: {p}")

    values: List[List[float]] = []
    for raw in text[data_start:]:
        line = raw.strip()
        if not line:
            continue
        row = [float(x) for x in line.split()]
        if len(row) >= channel_cursor:
            values.append(row[:channel_cursor])
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError(f"BVH motion values empty: {p}")
    if frames is not None and data.shape[0] != frames:
        frames = data.shape[0]

    offsets = np.stack([j["offset"] for j in joints]).astype(np.float32)
    bone_lens = np.linalg.norm(offsets, axis=1)
    nonzero = bone_lens[bone_lens > 1e-6]
    # BVH is often centimeters; meters-scale skeletons have bone lengths < 2.
    auto_scale = 0.01 if (nonzero.size and float(np.percentile(nonzero, 90)) > 2.0) else 1.0
    # If root trajectory itself is huge, also treat it as centimeters.
    root_j = 0
    root_ch = joints[root_j]["channels"]
    root_st = int(joints[root_j]["channel_start"])
    pos_cols = {ch.lower(): root_st + k for k, ch in enumerate(root_ch) if ch.lower().endswith("position")}
    root_xyz = np.zeros((data.shape[0], 3), dtype=np.float32)
    for axis, out_i in [("xposition", 0), ("yposition", 1), ("zposition", 2)]:
        if axis in pos_cols:
            root_xyz[:, out_i] = data[:, pos_cols[axis]]
    if np.nanpercentile(np.linalg.norm(root_xyz[:, [0, 2]] - root_xyz[:1, [0, 2]], axis=1), 95) > 20.0:
        auto_scale = 0.01
    root_xyz *= float(auto_scale)

    local_all = np.tile(np.eye(3, dtype=np.float32), (data.shape[0], len(joints), 1, 1))
    for j_idx, j in enumerate(joints):
        ch = list(j["channels"])
        st = int(j["channel_start"])
        rot_idx = [k for k, c in enumerate(ch) if c.lower().endswith("rotation")]
        rot_ch = [ch[k] for k in rot_idx]
        if not rot_idx:
            continue
        for t in range(data.shape[0]):
            vals = [data[t, st + k] for k in rot_idx]
            local_all[t, j_idx] = _bvh_euler_to_matrix(rot_ch, vals)

    target_idx = _bvh_target_joint_indices([str(j["name"]) for j in joints])
    target_local = np.tile(np.eye(3, dtype=np.float32), (data.shape[0], NUM_JOINTS, 1, 1))
    for tgt, src in enumerate(target_idx):
        if 0 <= src < len(joints):
            target_local[:, tgt] = local_all[:, src]

    out = np.zeros((data.shape[0], EDGE_DIM), dtype=np.float32)
    out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root_xyz
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(target_local).reshape(data.shape[0], -1)
    # Lightweight metadata in unused leading channels for traceability without
    # touching the EDGE root/rotation convention used downstream.
    out[:, 0] = float(1.0 / max(frame_time, 1e-8))
    out[:, 1] = float(auto_scale)
    return [out.astype(np.float32)]

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
        elif p.suffix.lower() == ".bvh":
            outs.extend(load_bvh_file(p))
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
    """Convert rotation matrices to EDGE/Zhou 6D in column-concatenated form.

    V46.21/V46.31 critical fix:
    The inverse of rot6d_to_matrix_np() must concatenate the first two matrix
    columns as [R[:,0], R[:,1]].  The previous row-major expression
    ``mat[..., :, 0:2].reshape(..., 6)`` interleaves rows as
    [R00, R01, R10, R11, R20, R21], which turns the identity matrix into
    [1, 0, 0, 1, 0, 0] instead of [1, 0, 0, 0, 1, 0].  That silently corrupts
    saved Event-RAG clips and makes strict raw-rot6d audit fail even after
    projection.
    """
    m = np.asarray(mat, dtype=np.float32)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"matrix_to_rot6d_np expects [...,3,3], got {m.shape}")
    c0 = m[..., :, 0]
    c1 = m[..., :, 1]
    return np.concatenate([c0, c1], axis=-1).astype(np.float32)


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
    # V46.8: Chang-E BVH files are often high-FPS (manifest reports about 210fps).
    # Always resample source motions into the EDGE training FPS before event slicing.
    bvh_resample_to_config_fps: bool = True
    manifest_enable: bool = True
    manifest_secondary_event_split: bool = True

    # V46.9: Chang-E/change BVH filename semantics are scientifically meaningful.
    # Each original BVH file is a source_uid/source_group, while the parsed
    # category/gender/take fields become semantic metadata and weak alignment
    # priors for unpaired slot-to-event routing.  This fixes the V46.8 issue
    # where singleton names such as female_lotus/male_ribbon were collapsed
    # into the same directory-level source group.
    source_group_mode: str = "filename"  # filename | legacy_prefix
    filename_semantic_enable: bool = True
    filename_semantic_weight: float = 0.35
    filename_semantic_retrieval_weight: float = 0.20
    filename_semantic_ot_weight: float = 0.35
    # V46.11: stronger multi-label RAG action semantics and music alignment labels.
    classification_semantic_enable: bool = True
    classification_semantic_ratio: float = 0.70
    classification_retrieval_weight: float = 0.34
    classification_ot_weight: float = 0.45
    classification_retrieval_bonus: float = 0.28
    classification_report_topk: int = 8
    # V46.31: Chang-E event-level semantic routing.
    chang_e_event_semantic_enable: bool = True
    semantic_routing_weight: float = 0.72
    event_family_bonus: float = 0.58
    motion_stage_role_bonus: float = 0.36
    preferred_dance_key_bonus: float = 0.28
    route_natural_duration_weight: float = 0.20
    route_family_balance_penalty: float = 0.18
    route_family_recent_window: int = 8
    route_family_penalty_cap: float = 0.25
    route_dance_key_repeat_penalty: float = 0.16
    route_family_repeat_penalty: float = 0.12
    route_source_repeat_penalty: float = 0.10
    route_motif_recall_bonus: float = 0.12
    route_debug_topk: int = 10
    # V46.31: convert long Chang-E BVH into a curated 72BVH-like semantic event library.
    chang_e_boundary_event_split: bool = True
    chang_e_boundary_max_extra_starts: int = 96
    chang_e_min_event_quality: float = 0.22
    chang_e_keep_pose_anchor_quality: float = 0.16
    event_quality_weight: float = 0.22
    route_support_bonus: float = 0.12
    route_locomotion_bonus: float = 0.14
    route_stage_sequence_weight: float = 0.16
    route_source_run_hard_penalty: float = 0.30
    route_semantic_bonus_scale: float = 1.50

    # V46.12: external classical-music semantic encoder.  The current EDGE
    # repository contains rule/librosa music feature extractors, not a trained
    # classical-music classifier.  When a trained external encoder exists, it
    # can write slot-level JSON/NPZ semantics or be invoked through a command
    # template.  These semantic probabilities then drive V44 unpaired OT and
    # V46 retrieval reports.
    external_music_semantic_enable: bool = True
    external_music_semantic_required: bool = False
    external_music_semantic_dirs: str = ""
    external_music_semantic_cmd: str = ""
    external_music_semantic_cache_dir: str = "output/v46_external_music_semantic_cache"
    external_music_semantic_weight: float = 0.78
    external_music_semantic_temperature: float = 0.65
    external_music_semantic_proxy_enable: bool = True
    external_music_semantic_filename_proxy: bool = True

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
    # V46.4: cloud-step is not a release / continue any more.  Large XZ travel
    # in contact is classified by speed and then mapped to a sliding anchor.
    # This avoids the Footskate Forgiveness Paradox: severe slow AI drifting is
    # still locked, while true Dunhuang cloud-step gets a smooth moving target.
    ik_slide_release_m: float = 0.12
    ik_slide_release_min_frames: int = 4
    ik_cloud_step_speed_mps: float = 0.15
    ik_sliding_anchor_window: int = 10
    ik_cloud_speed_cv_max: float = 1.75
    # V46.1: root-Y ballistic/damping pass. It is deliberately C1-safe and
    # never breaks a damping cycle mid-contact.
    root_y_physics_enable: bool = True
    root_y_flight_strength: float = 0.18
    root_y_min_flight_frames: int = 3
    # V46.3: biological fuse. If the no-contact interval is longer than this,
    # treat it as corrupted contact labels / bad upstream generation, not a
    # real human jump. Do not inject a huge ballistic parabola or landing dip.
    root_y_max_flight_seconds: float = 1.20
    root_y_damping_max_dip: float = 0.018
    # V46.8: cap landing damping to an early post-touchdown window.  The window
    # still starts and ends at zero dip, but it no longer stretches across a
    # multi-second support island and therefore cannot create delayed squats.
    root_y_damping_max_seconds: float = 0.28

    # V46.8: root-aware cloud-step guard. A true cloud-step requires foot travel
    # to be consistent with root/CoM translation; smooth AI dark-drift with no
    # body support is kept on the static-anchor repair path.
    ik_cloud_root_min_travel_m: float = 0.045
    ik_cloud_direction_cos_min: float = 0.35
    ik_cloud_root_foot_rel_max_m: float = 0.18

    # V46.8: long-sequence IK stitching and rollback safety.
    ik_chunk_overlap: int = 24
    rollback_skate_ratio: float = 1.18
    rollback_jerk_ratio: float = 1.18
    rollback_penetration_margin_m: float = 0.012
    rollback_root_delta_max_m: float = 0.18

    # V46.8: require real audio features for V44; false keeps the weak-proxy
    # fallback for smoke tests, while reports mark it as weak supervision.
    contrastive_require_real_music: bool = False
    audio_pair_min_coverage: float = 0.15

    # V46.8: Chang-E BVH is usually motion-only.  When no synchronized
    # BVH/audio pairs exist, train V44 with real unpaired music clips plus
    # semantic optimal-transport pseudo pairs instead of motion-descriptor
    # self-distillation.  This keeps the method honest: the checkpoint records
    # unpaired_audio_semantic_ot, not paired supervision.
    unpaired_audio_enable: bool = True
    unpaired_audio_slot_seconds: float = 4.0
    unpaired_positive_topk: int = 8
    unpaired_pairs_per_audio_slot: int = 4
    unpaired_min_audio_slots: int = 1
    unpaired_disable_motion_proxy: bool = False

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
            "V46_CHANG_E_EVENT_SEMANTIC_ENABLE": ("chang_e_event_semantic_enable", lambda x: bool(int(x))),
            "V46_SEMANTIC_ROUTING_WEIGHT": ("semantic_routing_weight", float),
            "V46_EVENT_FAMILY_BONUS": ("event_family_bonus", float),
            "V46_MOTION_STAGE_ROLE_BONUS": ("motion_stage_role_bonus", float),
            "V46_PREFERRED_DANCE_KEY_BONUS": ("preferred_dance_key_bonus", float),
            "V46_ROUTE_NATURAL_DURATION_WEIGHT": ("route_natural_duration_weight", float),
            "V46_ROUTE_FAMILY_BALANCE_PENALTY": ("route_family_balance_penalty", float),
            "V46_ROUTE_FAMILY_RECENT_WINDOW": ("route_family_recent_window", int),
            "V46_ROUTE_FAMILY_PENALTY_CAP": ("route_family_penalty_cap", float),
            "V46_ROUTE_DANCE_KEY_REPEAT_PENALTY": ("route_dance_key_repeat_penalty", float),
            "V46_ROUTE_FAMILY_REPEAT_PENALTY": ("route_family_repeat_penalty", float),
            "V46_ROUTE_SOURCE_REPEAT_PENALTY": ("route_source_repeat_penalty", float),
            "V46_ROUTE_MOTIF_RECALL_BONUS": ("route_motif_recall_bonus", float),
            "V46_ROUTE_DEBUG_TOPK": ("route_debug_topk", int),
            "V46_CHANG_E_BOUNDARY_EVENT_SPLIT": ("chang_e_boundary_event_split", lambda x: bool(int(x))),
            "V46_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS": ("chang_e_boundary_max_extra_starts", int),
            "V46_CHANG_E_MIN_EVENT_QUALITY": ("chang_e_min_event_quality", float),
            "V46_CHANG_E_KEEP_POSE_ANCHOR_QUALITY": ("chang_e_keep_pose_anchor_quality", float),
            "V46_EVENT_QUALITY_WEIGHT": ("event_quality_weight", float),
            "V46_ROUTE_SUPPORT_BONUS": ("route_support_bonus", float),
            "V46_ROUTE_LOCOMOTION_BONUS": ("route_locomotion_bonus", float),
            "V46_ROUTE_STAGE_SEQUENCE_WEIGHT": ("route_stage_sequence_weight", float),
            "V46_ROUTE_SOURCE_RUN_HARD_PENALTY": ("route_source_run_hard_penalty", float),
            "V46_ROUTE_SEMANTIC_BONUS_SCALE": ("route_semantic_bonus_scale", float),
            "V46_OVERLAP": ("overlap", int),
            "V46_WINDOW_LEN": ("window_len", int),
            "V46_HOP_LEN": ("hop_len", int),
            "V46_MIN_EVENT_FRAMES": ("min_event_frames", int),
            "V46_MAX_EVENT_FRAMES": ("max_event_frames", int),
            "V46_ENABLE_CONTRASTIVE": ("contrastive_enable", lambda x: bool(int(x))),
            "V46_IK_ITERS": ("ik_iters", int),
            "V46_IK_SLIDE_RELEASE_M": ("ik_slide_release_m", float),
            "V46_IK_CLOUD_STEP_SPEED_MPS": ("ik_cloud_step_speed_mps", float),
            "V46_IK_SLIDING_ANCHOR_WINDOW": ("ik_sliding_anchor_window", int),
            "V46_IK_CLOUD_SPEED_CV_MAX": ("ik_cloud_speed_cv_max", float),
            "V46_IK_CLOUD_ROOT_MIN_TRAVEL_M": ("ik_cloud_root_min_travel_m", float),
            "V46_IK_CLOUD_DIRECTION_COS_MIN": ("ik_cloud_direction_cos_min", float),
            "V46_IK_CLOUD_ROOT_FOOT_REL_MAX_M": ("ik_cloud_root_foot_rel_max_m", float),
            "V46_IK_CHUNK_OVERLAP": ("ik_chunk_overlap", int),
            "V46_ROLLBACK_SKATE_RATIO": ("rollback_skate_ratio", float),
            "V46_ROLLBACK_JERK_RATIO": ("rollback_jerk_ratio", float),
            "V46_ROLLBACK_PENETRATION_MARGIN_M": ("rollback_penetration_margin_m", float),
            "V46_ROLLBACK_ROOT_DELTA_MAX_M": ("rollback_root_delta_max_m", float),
            "V46_CONTRASTIVE_REQUIRE_REAL_MUSIC": ("contrastive_require_real_music", lambda x: bool(int(x))),
            "V46_AUDIO_PAIR_MIN_COVERAGE": ("audio_pair_min_coverage", float),
            "V46_UNPAIRED_AUDIO_ENABLE": ("unpaired_audio_enable", lambda x: bool(int(x))),
            "V46_UNPAIRED_AUDIO_SLOT_SECONDS": ("unpaired_audio_slot_seconds", float),
            "V46_UNPAIRED_POSITIVE_TOPK": ("unpaired_positive_topk", int),
            "V46_UNPAIRED_PAIRS_PER_AUDIO_SLOT": ("unpaired_pairs_per_audio_slot", int),
            "V46_UNPAIRED_DISABLE_MOTION_PROXY": ("unpaired_disable_motion_proxy", lambda x: bool(int(x))),
            "V46_ROOT_Y_DAMPING_MAX_SECONDS": ("root_y_damping_max_seconds", float),
            "V46_ENABLE_ROOT_Y_PHYSICS": ("root_y_physics_enable", lambda x: bool(int(x))),
            "V46_ROOT_Y_MIN_FLIGHT_FRAMES": ("root_y_min_flight_frames", int),
            "V46_ROOT_Y_MAX_FLIGHT_SECONDS": ("root_y_max_flight_seconds", float),
            "V46_DIFFUSION_STEPS": ("diffusion_steps", int),
            "V46_DEVICE": ("device", str),
            "V46_BVH_RESAMPLE_TO_CONFIG_FPS": ("bvh_resample_to_config_fps", lambda x: bool(int(x))),
            "V46_SOURCE_GROUP_MODE": ("source_group_mode", str),
            "V46_FILENAME_SEMANTIC_ENABLE": ("filename_semantic_enable", lambda x: bool(int(x))),
            "V46_FILENAME_SEMANTIC_WEIGHT": ("filename_semantic_weight", float),
            "V46_FILENAME_SEMANTIC_RETRIEVAL_WEIGHT": ("filename_semantic_retrieval_weight", float),
            "V46_FILENAME_SEMANTIC_OT_WEIGHT": ("filename_semantic_ot_weight", float),
            "V46_CLASSIFICATION_SEMANTIC_ENABLE": ("classification_semantic_enable", lambda x: bool(int(x))),
            "V46_CLASSIFICATION_SEMANTIC_RATIO": ("classification_semantic_ratio", float),
            "V46_CLASSIFICATION_RETRIEVAL_WEIGHT": ("classification_retrieval_weight", float),
            "V46_CLASSIFICATION_OT_WEIGHT": ("classification_ot_weight", float),
            "V46_CLASSIFICATION_RETRIEVAL_BONUS": ("classification_retrieval_bonus", float),
            "V46_CLASSIFICATION_REPORT_TOPK": ("classification_report_topk", int),
            "V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE": ("external_music_semantic_enable", lambda x: bool(int(x))),
            "V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED": ("external_music_semantic_required", lambda x: bool(int(x))),
            "V46_EXTERNAL_MUSIC_SEMANTIC_DIRS": ("external_music_semantic_dirs", str),
            "V46_EXTERNAL_MUSIC_SEMANTIC_CMD": ("external_music_semantic_cmd", str),
            "V46_EXTERNAL_MUSIC_SEMANTIC_CACHE_DIR": ("external_music_semantic_cache_dir", str),
            "V46_EXTERNAL_MUSIC_SEMANTIC_WEIGHT": ("external_music_semantic_weight", float),
            "V46_EXTERNAL_MUSIC_SEMANTIC_TEMPERATURE": ("external_music_semantic_temperature", float),
            "V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE": ("external_music_semantic_proxy_enable", lambda x: bool(int(x))),
            "V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY": ("external_music_semantic_filename_proxy", lambda x: bool(int(x))),
            "V46_MANIFEST_ENABLE": ("manifest_enable", lambda x: bool(int(x))),
            "V46_MANIFEST_SECONDARY_EVENT_SPLIT": ("manifest_secondary_event_split", lambda x: bool(int(x))),
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
































# -----------------------------------------------------------------------------
# V46.31 research contract guards for Chang-E/change RAG DB
# -----------------------------------------------------------------------------
def identity6d_np(shape_prefix: Tuple[int, ...] = ()) -> np.ndarray:
    """Return identity rotation in the repository's 6D convention."""
    base = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if not shape_prefix:
        return base.copy()
    return np.broadcast_to(base, tuple(shape_prefix) + (6,)).copy().astype(np.float32)


def sanitize_rot6d_np(rot6d: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Replace invalid / degenerate 6D rotations with identity before projection."""
    r = np.asarray(rot6d, dtype=np.float32).copy()
    if r.size == 0:
        return r.astype(np.float32), {"bad_joint_count": 0, "bad_joint_ratio": 0.0}
    r = r.reshape(-1, NUM_JOINTS, 6)
    finite = np.isfinite(r).all(axis=-1)
    a1 = r[..., 0:3]
    a2 = r[..., 3:6]
    a1_clean = np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0)
    a2_clean = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0)
    n1 = np.linalg.norm(a1_clean, axis=-1)
    n2 = np.linalg.norm(a2_clean, axis=-1)
    # V46.31: also reject near-collinear 6D vectors.  Gram-Schmidt
    # can collapse when a1 and a2 are parallel/anti-parallel even if both
    # vector norms are valid, which can happen during early diffusion denoising.
    denom = np.maximum(n1 * n2, 1e-8)
    cross_norm = np.linalg.norm(np.cross(a1_clean, a2_clean), axis=-1) / denom
    bad = (~finite) | (n1 < 1e-5) | (n2 < 1e-5) | (cross_norm < 1e-5)
    bad_count = int(np.sum(bad))
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if bad_count:
        r[bad] = identity6d_np((bad_count,))
    report = {
        "bad_joint_count": bad_count,
        "bad_joint_ratio": float(bad_count / max(1, bad.size)),
        "min_a1_norm_before_identity": float(np.nanmin(n1)) if n1.size else 0.0,
        "min_a2_norm_before_identity": float(np.nanmin(n2)) if n2.size else 0.0,
        "min_cross_norm_before_identity": float(np.nanmin(cross_norm)) if cross_norm.size else 0.0,
        "near_collinear_joint_count": int(np.sum(cross_norm < 1e-5)) if cross_norm.size else 0,
    }
    return r.reshape(np.asarray(rot6d).shape).astype(np.float32), report


def project_edge151_rot6d_np(motion: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Project every EDGE 6D rotation channel back to SO(3)-derived 6D safely."""
    x = np.asarray(motion, dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[1] < EDGE_DIM or x.shape[0] <= 0:
        return x.astype(np.float32), {"projected": False, "reason": "invalid_shape"}
    rot = x[:, ROT6D_START:ROT6D_END].reshape(x.shape[0], NUM_JOINTS, 6)
    rot, sanitize_report = sanitize_rot6d_np(rot)
    x[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(
        rot6d_to_matrix_np(rot.reshape(x.shape[0], NUM_JOINTS, 6))
    ).reshape(x.shape[0], -1)
    sanitize_report["projected"] = True
    return x.astype(np.float32), sanitize_report


def rotate_motion_around_y_np(motion: np.ndarray, yaw_delta: float, pivot_xz: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotate a whole EDGE-151D motion around the vertical Y axis.

    This is a world-space rigid yaw transform for event stitching. It rotates
    the root XZ trajectory around ``pivot_xz`` and left-multiplies the root
    joint rotation by R_y(yaw_delta). Child joint local rotations remain valid
    because the root orientation carries the global heading change.
    """
    out = np.asarray(motion, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < ROT6D_END or out.shape[0] <= 0:
        return out.astype(np.float32)
    yaw = float(yaw_delta)
    if not np.isfinite(yaw) or abs(yaw) < 1e-8:
        return out.astype(np.float32)
    c = float(np.cos(yaw))
    ss = float(np.sin(yaw))
    if pivot_xz is None:
        pivot = out[0, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
    else:
        pivot = np.asarray(pivot_xz, dtype=np.float32).reshape(2)
    rel_x = out[:, ROOT_X_IDX].copy() - float(pivot[0])
    rel_z = out[:, ROOT_Z_IDX].copy() - float(pivot[1])
    out[:, ROOT_X_IDX] = c * rel_x + ss * rel_z + float(pivot[0])
    out[:, ROOT_Z_IDX] = -ss * rel_x + c * rel_z + float(pivot[1])

    ry = np.asarray([[c, 0.0, ss], [0.0, 1.0, 0.0], [-ss, 0.0, c]], dtype=np.float32)
    root6 = out[:, ROT6D_START:ROT6D_START + 6].reshape(out.shape[0], 1, 6)
    root_r = rot6d_to_matrix_np(root6)
    root_r = np.matmul(ry[None, None, :, :], root_r).astype(np.float32)
    out[:, ROT6D_START:ROT6D_START + 6] = matrix_to_rot6d_np(root_r).reshape(out.shape[0], 6)
    return out.astype(np.float32)


def _safe_percentile(arr: np.ndarray, q: float, default: float = 0.0) -> float:
    try:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            return float(default)
        return float(np.nanpercentile(a, q))
    except Exception:
        return float(default)


def heuristic_contacts_fallback_np(motion: np.ndarray, cfg: V46Config, source_hint: str = "") -> Tuple[np.ndarray, dict]:
    """Kinematic fallback for contact channels when the main FK contact builder fails.

    V46.19 fix: never replace a time-varying foot contact signal with a static
    scalar such as 0.50 or 0.60.  First try a simple foot-height + foot-velocity
    heuristic from FK joints.  If even FK is unavailable, fall back to a root
    height/speed heuristic so the signal remains temporally varying rather than
    permanently locking or releasing both feet.
    """
    x = np.asarray(motion, dtype=np.float32)[:, :EDGE_DIM]
    T = int(x.shape[0])
    margin = float(getattr(cfg, "ik_height_margin", 0.05))
    speed_gate = float(getattr(cfg, "ik_speed_gate_mpf", 0.035))
    report = {"source_hint": str(source_hint), "mode": "uninitialized"}
    contacts = np.zeros((T, 4), dtype=np.float32)
    if T <= 0:
        report["mode"] = "empty"
        return contacts, report

    try:
        joints = fk_24_np(x)
        foot_ids = list(DEFAULT_FOOT_JOINTS)
        foot = joints[:, foot_ids]
        foot_vxz = np.zeros(foot.shape[:2], dtype=np.float32)
        if T > 1:
            foot_vxz[1:] = np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
        floor_y = float(np.nanpercentile(foot[..., 1].reshape(-1), 5))
        near = foot[..., 1] <= floor_y + max(0.015, margin)
        slow = foot_vxz <= max(0.01, speed_gate)
        contacts = (near & slow).astype(np.float32)
        # Avoid all-zero output caused by overly strict speed thresholds on noisy
        # data.  Use near-floor alone as a second-stage fallback, still per-frame.
        if float(contacts.mean()) < 0.02:
            contacts = near.astype(np.float32)
            report["secondary_mode"] = "near_floor_without_speed_gate"
        report.update({
            "mode": "fk_height_velocity_heuristic",
            "floor_y": floor_y,
            "contact_ratio": float(contacts.mean()),
            "height_margin": float(margin),
            "speed_gate_mpf": float(speed_gate),
        })
        return contacts.astype(np.float32), report
    except Exception as exc:
        report["fk_heuristic_error"] = str(exc)

    # Last-resort fallback when FK is unavailable.  Without foot joints, there is
    # no physically reliable way to decide left/right support.  Therefore V46.19
    # deliberately avoids copying one root-level state to all four foot contacts:
    # that would weld both feet on near-root frames and release both feet otherwise.
    # Instead, produce a weak, non-anchoring, time-varying uncertainty signal that
    # stays below ik_contact_high, so IK will not impose a false strong foot lock.
    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    root_speed = np.zeros((T,), dtype=np.float32)
    if T > 1:
        root_speed[1:] = np.linalg.norm(root[1:, [0, 2]] - root[:-1, [0, 2]], axis=-1)
    root_floor = float(np.nanpercentile(root[:, 1], 20)) if T else 0.0
    near_root = root[:, 1] <= root_floor + max(0.02, margin)
    slow_root = root_speed <= max(0.02, speed_gate * 2.0)
    support_like = (near_root & slow_root).astype(np.float32)
    uncertain = min(0.50, max(0.42, float(getattr(cfg, "ik_contact_low", 0.38)) + 0.06))
    release = max(0.05, min(0.25, float(getattr(cfg, "ik_contact_low", 0.38)) - 0.10))
    base = release + (uncertain - release) * support_like
    contacts = np.repeat(base[:, None], 4, axis=1).astype(np.float32)
    report.update({
        "mode": "root_uncertain_nonlocking_no_fk",
        "root_floor_y": root_floor,
        "contact_ratio": float(contacts.mean()),
        "uncertain_contact_value": float(uncertain),
        "release_contact_value": float(release),
        "height_margin": float(margin),
        "speed_gate_mpf": float(speed_gate),
        "warning": "FK unavailable; foot-specific contacts cannot be recovered, so fallback intentionally avoids strong IK anchoring.",
    })
    return contacts.astype(np.float32), report


def enforce_edge151_contract_np(
    motion: np.ndarray,
    cfg: Optional[V46Config] = None,
    source_hint: str = "",
    derive_contact: bool = True,
    project_rot: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Return a valid EDGE-151D motion tensor and an audit report.

    Critical reason:
    direct BVH loading may temporarily use channel 0 to carry native fps before
    resampling.  That is legal only inside loading.  Once an array is saved as an
    Event-RAG clip or passed into V45/V46, EDGE [0:4] must again mean contacts.
    """
    cfg = cfg or V46Config()
    x0 = np.asarray(motion, dtype=np.float32)
    report = {
        "version": "v46_24_edge151_contract_guard",
        "source_hint": str(source_hint),
        "input_shape": list(x0.shape),
    }
    if x0.ndim != 2 or x0.shape[1] < EDGE_DIM:
        raise ValueError(
            f"EDGE151 contract violation: expected [T,151+], got {tuple(x0.shape)} from {source_hint}"
        )

    x = x0[:, :EDGE_DIM].astype(np.float32).copy()
    finite_before = bool(np.isfinite(x).all())
    report["finite_before"] = finite_before

    # Handle root/contact/other scalar channels conservatively, but never let
    # invalid rot6d become all-zero rotations.  Rot6D is sanitized separately.
    scalar_idx = list(range(0, ROT6D_START))
    x[:, scalar_idx] = np.nan_to_num(x[:, scalar_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rot_flat, rot_sanitize_report = sanitize_rot6d_np(x[:, ROT6D_START:ROT6D_END])
    x[:, ROT6D_START:ROT6D_END] = rot_flat.reshape(x.shape[0], -1)
    report["rot6d_sanitize"] = rot_sanitize_report

    contact_before = x[:, 0:4].copy()
    report["contact_before_min"] = float(np.min(contact_before)) if contact_before.size else 0.0
    report["contact_before_max"] = float(np.max(contact_before)) if contact_before.size else 0.0
    report["contact_before_abs_p95"] = _safe_percentile(np.abs(contact_before), 95)
    contact_polluted = bool(report["contact_before_abs_p95"] > 1.5 or report["contact_before_min"] < -0.05)
    report["contact_metadata_pollution_detected"] = contact_polluted

    if project_rot:
        report["rot6d_abs_p95_before_project"] = _safe_percentile(np.abs(x[:, ROT6D_START:ROT6D_END]), 95)
        x, project_report = project_edge151_rot6d_np(x)
        report["rot6d_project"] = project_report
        report["rot6d_projected"] = True
    else:
        report["rot6d_projected"] = False

    if derive_contact:
        try:
            contacts, conf, floor_y, _ = derive_contacts_np(x, cfg)
            x[:, 0:4] = contacts.astype(np.float32)
            report["contact_rebuilt_from_fk"] = True
            report["contact_ratio"] = float(contacts.mean())
            report["contact_conf_mean"] = float(np.mean(conf))
            report["floor_y"] = float(floor_y)
        except Exception as exc:
            # V46.19: do not replace a dynamic contact signal with a global
            # constant such as 0.50 or 0.60.  Generate a per-frame kinematic
            # fallback so IK neither releases nor welds both feet for the whole clip.
            if contact_polluted:
                contacts_fb, fb_report = heuristic_contacts_fallback_np(
                    x, cfg, source_hint=f"contact_rebuild_failed:{source_hint}"
                )
                x[:, 0:4] = contacts_fb.astype(np.float32)
                report["contact_fallback_mode"] = "time_varying_kinematic_heuristic_due_to_metadata_pollution"
                report["contact_fallback_report"] = fb_report
            else:
                x[:, 0:4] = np.clip(np.nan_to_num(x[:, 0:4], nan=0.0), 0.0, 1.0)
                report["contact_fallback_mode"] = "clipped_existing_contact"
            report["contact_rebuilt_from_fk"] = False
            report["contact_rebuild_error"] = str(exc)
    else:
        if contact_polluted:
            contacts_fb, fb_report = heuristic_contacts_fallback_np(
                x, cfg, source_hint=f"derive_contact_false:{source_hint}"
            )
            x[:, 0:4] = contacts_fb.astype(np.float32)
            report["contact_fallback_mode"] = "derive_contact_false_time_varying_kinematic_heuristic_due_to_metadata_pollution"
            report["contact_fallback_report"] = fb_report
        else:
            x[:, 0:4] = np.clip(np.nan_to_num(x[:, 0:4], nan=0.0), 0.0, 1.0)
            report["contact_fallback_mode"] = "derive_contact_false_clipped_existing_contact"
        report["contact_rebuilt_from_fk"] = False

    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    report["root_min"] = [float(v) for v in np.min(root, axis=0)]
    report["root_max"] = [float(v) for v in np.max(root, axis=0)]
    report["root_y_range_m"] = float(np.max(x[:, ROOT_Y_IDX]) - np.min(x[:, ROOT_Y_IDX]))
    report["root_xz_travel_m"] = float(
        np.linalg.norm(x[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - x[0, [ROOT_X_IDX, ROOT_Z_IDX]])
    )
    report["contact_after_abs_p95"] = _safe_percentile(np.abs(x[:, 0:4]), 95)
    report["rot6d_abs_p95_after"] = _safe_percentile(np.abs(x[:, ROT6D_START:ROT6D_END]), 95)
    return x.astype(np.float32), report


def sliding_window_ranges(T: int, window: int, hop: int) -> List[Tuple[int, int]]:
    """Return coverage-complete sliding windows for long-sequence inference."""
    T = int(T)
    window = max(1, int(window))
    hop = max(1, int(hop))
    if T <= window:
        return [(0, T)]
    starts = list(range(0, max(1, T - window + 1), hop))
    last = T - window
    if starts[-1] != last:
        starts.append(last)
    return [(int(s), int(min(T, s + window))) for s in starts]


def overlap_add_weight_np(length: int, start: int, total: int, hop: int, window: int) -> np.ndarray:
    """Raised-cosine weight with global-boundary one-sided protection.

    V46.19 fix:
    a full symmetric Hann window attenuates the very first and very last global
    frames even though no outside window can compensate them.  We therefore keep
    the non-overlapped side of the first/last chunk at weight 1.0 and only use
    cosine weights inside actual cross-window transition regions.
    """
    length = int(length)
    start = int(start)
    total = int(total)
    if length <= 0:
        return np.zeros((0, 1), dtype=np.float32)
    if length == 1 or total <= length:
        return np.ones((length, 1), dtype=np.float32)

    n = np.arange(length, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / float(max(length - 1, 1)))
    w = np.maximum(w, 1e-4).astype(np.float32)

    # The first global chunk has no previous chunk on its left side; do not
    # attenuate the leading half.  The last global chunk has no following chunk
    # on its right side; do not attenuate the trailing half.
    half = max(1, length // 2)
    if start <= 0:
        w[:half] = 1.0
    if start + length >= total:
        w[half:] = 1.0
    return w[:, None].astype(np.float32)

def normalize_quat_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    out = q / np.maximum(norm, 1e-8)
    bad = (~np.isfinite(out).all(axis=-1)) | (norm[..., 0] < 1e-8)
    if np.any(bad):
        out = out.copy()
        out[bad] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return out.astype(np.float32)


def matrix_to_quat_np(R: np.ndarray) -> np.ndarray:
    """Vectorized rotation-matrix to unit quaternion conversion [w,x,y,z].

    V46.19 fix: avoid per-matrix Python loops.  Long whole-song inference calls
    this function many times for [T,24,3,3] arrays, so the branch logic is
    implemented with NumPy masks to keep official evaluation practical.
    """
    arr = np.asarray(R, dtype=np.float32)
    prefix = arr.shape[:-2]
    m = arr.reshape(-1, 3, 3)
    q = np.zeros((m.shape[0], 4), dtype=np.float32)
    if m.shape[0] == 0:
        return q.reshape(prefix + (4,)).astype(np.float32)

    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    tr = m00 + m11 + m22

    mask = tr > 0.0
    if np.any(mask):
        s = np.sqrt(np.maximum(tr[mask] + 1.0, 1e-8)) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (m21[mask] - m12[mask]) / s
        q[mask, 2] = (m02[mask] - m20[mask]) / s
        q[mask, 3] = (m10[mask] - m01[mask]) / s

    rem = ~mask
    mask_x = rem & (m00 > m11) & (m00 > m22)
    if np.any(mask_x):
        s = np.sqrt(np.maximum(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x], 1e-8)) * 2.0
        q[mask_x, 0] = (m21[mask_x] - m12[mask_x]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (m01[mask_x] + m10[mask_x]) / s
        q[mask_x, 3] = (m02[mask_x] + m20[mask_x]) / s

    mask_y = rem & (~mask_x) & (m11 > m22)
    if np.any(mask_y):
        s = np.sqrt(np.maximum(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y], 1e-8)) * 2.0
        q[mask_y, 0] = (m02[mask_y] - m20[mask_y]) / s
        q[mask_y, 1] = (m01[mask_y] + m10[mask_y]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (m12[mask_y] + m21[mask_y]) / s

    mask_z = rem & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        s = np.sqrt(np.maximum(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z], 1e-8)) * 2.0
        q[mask_z, 0] = (m10[mask_z] - m01[mask_z]) / s
        q[mask_z, 1] = (m02[mask_z] + m20[mask_z]) / s
        q[mask_z, 2] = (m12[mask_z] + m21[mask_z]) / s
        q[mask_z, 3] = 0.25 * s

    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return normalize_quat_np(q.reshape(prefix + (4,)))

def quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions [w,x,y,z] to rotation matrices."""
    q = normalize_quat_np(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R.astype(np.float32)


def init_motion_window_accumulators(T: int, D: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accum = np.zeros((int(T), int(D)), dtype=np.float32)
    weight_sum = np.zeros((int(T), 1), dtype=np.float32)
    rot_quat_accum = np.zeros((int(T), NUM_JOINTS, 4), dtype=np.float32)
    rot_quat_weight = np.zeros((int(T), 1, 1), dtype=np.float32)
    return accum, weight_sum, rot_quat_accum, rot_quat_weight


def accumulate_motion_window_np(
    accum: np.ndarray,
    weight_sum: np.ndarray,
    rot_quat_accum: np.ndarray,
    rot_quat_weight: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    start: int,
    end: int,
) -> None:
    """Accumulate a generated chunk without linearly averaging Rot6D.

    Root/contact/scalar channels use Euclidean overlap-add.  Rotation channels
    are converted to quaternions and accumulated on S^3 with sign alignment
    before a final normalized quaternion-to-rot6d projection.  This avoids the
    near-zero Rot6D cancellation and snap risk caused by direct Rot6D averaging.
    """
    y = np.asarray(y, dtype=np.float32)[: int(end - start), :EDGE_DIM]
    w = np.asarray(w, dtype=np.float32).reshape(-1, 1)[: y.shape[0]]
    if y.shape[0] == 0:
        return
    y_linear = y.copy()
    y_linear[:, ROT6D_START:ROT6D_END] = 0.0
    accum[start:end] += y_linear * w
    weight_sum[start:end] += w

    R = rot6d_to_matrix_np(y[:, ROT6D_START:ROT6D_END].reshape(y.shape[0], NUM_JOINTS, 6))
    q = matrix_to_quat_np(R)
    for li, gi in enumerate(range(int(start), int(end))):
        wi = float(w[li, 0])
        if wi <= 0.0:
            continue
        qi = q[li]
        if float(rot_quat_weight[gi, 0, 0]) > 1e-8:
            ref = normalize_quat_np(rot_quat_accum[gi])
            dots = np.sum(qi * ref, axis=-1, keepdims=True)
            qi = np.where(dots < 0.0, -qi, qi)
        rot_quat_accum[gi] += qi * wi
        rot_quat_weight[gi, 0, 0] += wi


def finalize_motion_window_accum_np(
    accum: np.ndarray,
    weight_sum: np.ndarray,
    rot_quat_accum: np.ndarray,
    rot_quat_weight: np.ndarray,
    cfg: V46Config,
    source_hint: str,
) -> Tuple[np.ndarray, dict]:
    out = accum / np.maximum(weight_sum, 1e-8)
    valid = rot_quat_weight[:, 0, 0] > 1e-8
    q = np.zeros((accum.shape[0], NUM_JOINTS, 4), dtype=np.float32)
    q[:] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if np.any(valid):
        q[valid] = normalize_quat_np(rot_quat_accum[valid])
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(R).reshape(accum.shape[0], -1)
    out, report = enforce_edge151_contract_np(out, cfg, source_hint=source_hint, derive_contact=True, project_rot=True)
    report["rotation_overlap_mode"] = "quaternion_sign_aligned_weighted_average"
    report["scalar_overlap_mode"] = "hann_weighted_overlap_add"
    report["weight_sum_min"] = float(np.min(weight_sum)) if weight_sum.size else 0.0
    report["weight_sum_p05"] = float(np.percentile(weight_sum, 5)) if weight_sum.size else 0.0
    return out.astype(np.float32), report


def blend_motion_overlap_np(
    a: np.ndarray,
    b: np.ndarray,
    w_b: np.ndarray,
    cfg: V46Config,
    source_hint: str = "blend_motion_overlap",
) -> Tuple[np.ndarray, dict]:
    """Blend two overlap clips with quaternion rotation fusion, not Rot6D LERP.

    a and b must have the same temporal length. Scalar/root/contact channels
    use Euclidean weights; Rot6D channels are converted to quaternions with
    sign alignment and then mapped back to Rot6D. This is used for RAG event
    boundary blending, where adjacent retrieved clips can have large pose gaps.
    """
    a = np.asarray(a, dtype=np.float32)[:, :EDGE_DIM]
    b = np.asarray(b, dtype=np.float32)[:, :EDGE_DIM]
    L = int(min(len(a), len(b)))
    if L <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32), {"blend_mode": "empty"}
    a = a[:L]
    b = b[:L]
    wb = np.asarray(w_b, dtype=np.float32).reshape(-1, 1)[:L]
    wb = np.clip(wb, 0.0, 1.0)
    wa = 1.0 - wb
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(L, EDGE_DIM)
    accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, a, wa, 0, L)
    accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, b, wb, 0, L)
    out, report = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint=source_hint
    )
    report["blend_mode"] = "scalar_linear_quaternion_rotation"
    report["w_b_min"] = float(np.min(wb)) if wb.size else 0.0
    report["w_b_max"] = float(np.max(wb)) if wb.size else 0.0
    return out.astype(np.float32), report

def _clean_stem(path: str | Path) -> str:
    return Path(str(path)).stem.strip().lower().replace("-", "_").replace(" ", "_")


CHANG_E_CATEGORY_PROFILES: Dict[str, Dict[str, object]] = {
    # The category names follow the Chang-E paper/Table-1 terminology while
    # staying compatible with the user's actual filenames under EDGE/change.
    "thirty_six_postures": {
        "aliases": {"36pose", "36posture", "36postures", "thirtysix", "thirty_six"},
        "display": "Ji Yue Tian Thirty-Six Postures",
        "semantic_role": "pose_sequence",
        "energy": 0.42, "onset": 0.22, "lower": 0.38, "upper": 0.55,
        "turn": 0.18, "travel": 0.25, "calmness": 0.45,
    },
    "lotus_steps": {
        "aliases": {"lotus", "lotussteps", "lotus_step", "lotus_steps"},
        "display": "Lotus Steps",
        "semantic_role": "flowing_footwork",
        "energy": 0.35, "onset": 0.16, "lower": 0.58, "upper": 0.32,
        "turn": 0.12, "travel": 0.45, "calmness": 0.52,
    },
    "revelation_meditation": {
        "aliases": {"meditation", "mediation", "revelation", "revelation_meditation", "revelation_mediation"},
        "display": "Revelation Meditation",
        "semantic_role": "calm_meditative_flow",
        "energy": 0.18, "onset": 0.05, "lower": 0.18, "upper": 0.28,
        "turn": 0.08, "travel": 0.10, "calmness": 0.86,
    },
    "pipa_behind_back": {
        "aliases": {"pipa", "pipa1", "pipa2", "playing_pipa", "playing_the_pipa"},
        "display": "Playing the Pipa Behind the Back",
        "semantic_role": "instrument_upper_body_motif",
        "energy": 0.48, "onset": 0.22, "lower": 0.30, "upper": 0.78,
        "turn": 0.25, "travel": 0.26, "calmness": 0.35,
    },
    "lei_gong_drum": {
        "aliases": {"drum", "lei_gong", "leigong", "lei_gong_drum"},
        "display": "Lei Gong Drum",
        "semantic_role": "percussive_high_energy",
        "energy": 0.78, "onset": 0.72, "lower": 0.72, "upper": 0.70,
        "turn": 0.20, "travel": 0.42, "calmness": 0.15,
    },
    "ribbon_flow": {
        "aliases": {"ribbon", "sash", "silk", "whirl", "sogdian", "sogdian_whirl"},
        "display": "Ribbon / Sogdian Whirl Flow",
        "semantic_role": "flowing_turning_motif",
        "energy": 0.62, "onset": 0.36, "lower": 0.50, "upper": 0.66,
        "turn": 0.70, "travel": 0.58, "calmness": 0.25,
    },
}


# V46.11: stronger classification semantics for Chang-E motion-only BVH.
# The taxonomy is deliberately multi-label: a source file name gives the primary
# dance family, while motion descriptors keep event-level dynamics.  This lets
# the RAG router use interpretable labels without pretending that filename tags
# are paired music supervision.
ENERGY_LABELS = ("calm", "moderate", "high", "percussive")
RHYTHM_LABELS = ("sustained", "lyrical", "accented", "percussive")
BODY_FOCUS_LABELS = ("pose", "lower_body", "upper_body", "full_body", "turning_flow")
SPATIAL_LABELS = ("in_place", "traveling", "turning")
MUSIC_ALIGNMENT_LABELS = (
    "calm_meditative", "lyrical_flow", "pose_hold", "instrument_phrase",
    "percussive_accent", "turning_climax", "footwork_flow",
)

CATEGORY_CLASS_OVERRIDES: Dict[str, Dict[str, object]] = {
    "thirty_six_postures": {
        "energy_label": "moderate",
        "rhythm_label": "sustained",
        "body_focus_label": "pose",
        "spatial_label": "in_place",
        "music_alignment_label": "pose_hold",
        "music_alignment_tags": ["pose_hold", "calm_meditative", "lyrical_flow"],
        "preferred_music_roles": ["intro", "normal", "release", "calm"],
        "preferred_dance_keys": ["thirty_six_postures", "lotus_steps", "revelation_meditation"],
    },
    "lotus_steps": {
        "energy_label": "moderate",
        "rhythm_label": "lyrical",
        "body_focus_label": "lower_body",
        "spatial_label": "traveling",
        "music_alignment_label": "footwork_flow",
        "music_alignment_tags": ["footwork_flow", "lyrical_flow", "calm_meditative"],
        "preferred_music_roles": ["normal", "release", "calm"],
        "preferred_dance_keys": ["lotus_steps", "ribbon_flow", "thirty_six_postures"],
    },
    "revelation_meditation": {
        "energy_label": "calm",
        "rhythm_label": "sustained",
        "body_focus_label": "full_body",
        "spatial_label": "in_place",
        "music_alignment_label": "calm_meditative",
        "music_alignment_tags": ["calm_meditative", "pose_hold"],
        "preferred_music_roles": ["intro", "calm", "release"],
        "preferred_dance_keys": ["revelation_meditation", "thirty_six_postures", "lotus_steps"],
    },
    "pipa_behind_back": {
        "energy_label": "moderate",
        "rhythm_label": "accented",
        "body_focus_label": "upper_body",
        "spatial_label": "in_place",
        "music_alignment_label": "instrument_phrase",
        "music_alignment_tags": ["instrument_phrase", "lyrical_flow", "percussive_accent"],
        "preferred_music_roles": ["normal", "build_up", "climax"],
        "preferred_dance_keys": ["pipa_behind_back", "ribbon_flow", "lei_gong_drum"],
    },
    "lei_gong_drum": {
        "energy_label": "percussive",
        "rhythm_label": "percussive",
        "body_focus_label": "full_body",
        "spatial_label": "traveling",
        "music_alignment_label": "percussive_accent",
        "music_alignment_tags": ["percussive_accent", "turning_climax"],
        "preferred_music_roles": ["build_up", "climax"],
        "preferred_dance_keys": ["lei_gong_drum", "pipa_behind_back", "ribbon_flow"],
    },
    "ribbon_flow": {
        "energy_label": "high",
        "rhythm_label": "lyrical",
        "body_focus_label": "turning_flow",
        "spatial_label": "turning",
        "music_alignment_label": "turning_climax",
        "music_alignment_tags": ["turning_climax", "lyrical_flow", "footwork_flow"],
        "preferred_music_roles": ["normal", "build_up", "climax"],
        "preferred_dance_keys": ["ribbon_flow", "lotus_steps", "pipa_behind_back"],
    },
}


def canonicalize_chang_e_key(key: object) -> str:
    """Canonicalize Chang-E category names for semantic RAG.

    The released/local filenames may contain spelling variants such as
    ``female_mediation.bvh``.  Source fields keep the raw filename for
    auditability, but all internal action/music labels should use the
    Chang-E paper terminology: ``revelation_meditation``.
    """
    key_s = str(key or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        key_s = re.sub(r"_take\d+$", "", key_s)
    except Exception:
        pass
    aliases = {
        "mediation": "revelation_meditation",
        "female_mediation": "revelation_meditation",
        "male_mediation": "revelation_meditation",
        "revelation_mediation": "revelation_meditation",
        "meditation": "revelation_meditation",
    }
    if key_s in aliases:
        return aliases[key_s]
    for k, prof in CHANG_E_CATEGORY_PROFILES.items():
        if key_s == k or key_s in set(prof.get("aliases", set())):
            return k
    return key_s


def _safe_profile_key(meta: dict) -> str:
    key = meta.get("dance_key") or meta.get("parent_label") or meta.get("label") or "unknown"
    return canonicalize_chang_e_key(key)


def _label_index(label: str, labels: Sequence[str]) -> int:
    try:
        return list(labels).index(str(label))
    except ValueError:
        return -1


def strong_action_semantics_from_meta(meta: dict, desc: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Return multi-label semantic metadata for an event.

    Filename category gives a stable cultural prior; descriptor statistics refine
    it at event level.  The labels are used for reporting and RAG routing, not as
    ground-truth paired music supervision.
    """
    key = _safe_profile_key(meta)
    prof = dict(CATEGORY_CLASS_OVERRIDES.get(key, {}))
    base_prof = CHANG_E_CATEGORY_PROFILES.get(key, {})
    energy = float(base_prof.get("energy", 0.40))
    onset = float(base_prof.get("onset", 0.20))
    travel = float(base_prof.get("travel", 0.25))
    turn = float(base_prof.get("turn", 0.15))
    lower = float(base_prof.get("lower", energy))
    upper = float(base_prof.get("upper", energy))
    calm = float(base_prof.get("calmness", max(0.0, 0.75 - energy)))
    if desc is not None and len(desc) >= 19:
        # Normalize rough descriptor channels into semantic refiners.  This is
        # intentionally weak: filename category remains the cultural prior.
        travel = max(travel, float(np.clip(desc[1] / 1.5, 0.0, 1.0)))
        energy = max(energy, float(np.clip(desc[5] / 0.18, 0.0, 1.0)))
        lower = max(lower, float(np.clip(desc[7] / 0.12, 0.0, 1.0)))
        upper = max(upper, float(np.clip(desc[8] / 0.12, 0.0, 1.0)))
        turn = max(turn, float(np.clip(abs(desc[17]) / 0.25, 0.0, 1.0)))
        calm = max(0.0, min(calm, 1.0 - min(0.9, energy * 0.65))) if energy > 0.65 else calm
    if "energy_label" not in prof:
        prof["energy_label"] = "calm" if energy < 0.28 else ("high" if energy > 0.62 else "moderate")
    if "rhythm_label" not in prof:
        prof["rhythm_label"] = "percussive" if onset > 0.55 else ("accented" if onset > 0.30 else ("sustained" if calm > 0.65 else "lyrical"))
    if "body_focus_label" not in prof:
        if turn > 0.58:
            prof["body_focus_label"] = "turning_flow"
        elif upper > lower * 1.35:
            prof["body_focus_label"] = "upper_body"
        elif lower > upper * 1.25:
            prof["body_focus_label"] = "lower_body"
        else:
            prof["body_focus_label"] = "full_body"
    if "spatial_label" not in prof:
        prof["spatial_label"] = "turning" if turn > 0.55 else ("traveling" if travel > 0.40 else "in_place")
    if "music_alignment_label" not in prof:
        if calm > 0.72:
            prof["music_alignment_label"] = "calm_meditative"
        elif onset > 0.55:
            prof["music_alignment_label"] = "percussive_accent"
        elif turn > 0.58:
            prof["music_alignment_label"] = "turning_climax"
        elif str(prof.get("body_focus_label")) == "upper_body":
            prof["music_alignment_label"] = "instrument_phrase"
        else:
            prof["music_alignment_label"] = "lyrical_flow"
    tags = list(dict.fromkeys([str(prof.get("music_alignment_label"))] + [str(x) for x in prof.get("music_alignment_tags", [])]))
    prof["music_alignment_tags"] = tags
    prof.setdefault("preferred_music_roles", ["normal"])
    prof.setdefault("preferred_dance_keys", [key])
    prof["classification_text"] = (
        f"action={key}; energy={prof.get('energy_label')}; rhythm={prof.get('rhythm_label')}; "
        f"body={prof.get('body_focus_label')}; spatial={prof.get('spatial_label')}; "
        f"music_align={prof.get('music_alignment_label')}"
    )
    return prof


def class_semantic_vector_from_meta(meta: dict, cfg: Optional[V46Config] = None) -> np.ndarray:
    """A 32D classification prior aligned with audio slot feature channels.

    This is stronger than the old name_semantic vector because it encodes
    multi-label action family, energy/rhythm/body-focus/spatial/music-affinity.
    It remains 32D to keep V44/V46 checkpoints compatible with the existing MLPs.
    """
    key = _safe_profile_key(meta)
    base = filename_semantic_vector_from_meta(meta, cfg).copy()
    cls = strong_action_semantics_from_meta(meta)
    energy_i = _label_index(str(cls.get("energy_label")), ENERGY_LABELS)
    rhythm_i = _label_index(str(cls.get("rhythm_label")), RHYTHM_LABELS)
    body_i = _label_index(str(cls.get("body_focus_label")), BODY_FOCUS_LABELS)
    spatial_i = _label_index(str(cls.get("spatial_label")), SPATIAL_LABELS)
    align_i = _label_index(str(cls.get("music_alignment_label")), MUSIC_ALIGNMENT_LABELS)
    # High-level one-hot / ordinal labels in high channels; low channels still
    # preserve slot-compatible continuous semantics.
    base[22] = 0.0 if energy_i < 0 else energy_i / max(1, len(ENERGY_LABELS) - 1)
    base[23] = 0.0 if rhythm_i < 0 else rhythm_i / max(1, len(RHYTHM_LABELS) - 1)
    base[24] = 0.0 if body_i < 0 else body_i / max(1, len(BODY_FOCUS_LABELS) - 1)
    base[25] = 0.0 if spatial_i < 0 else spatial_i / max(1, len(SPATIAL_LABELS) - 1)
    base[26] = 0.0 if align_i < 0 else align_i / max(1, len(MUSIC_ALIGNMENT_LABELS) - 1)
    # Category-specific compact code; stable across rebuilds.
    known = list(CHANG_E_CATEGORY_PROFILES.keys())
    ci = known.index(key) if key in known else -1
    base[27] = 0.0 if ci < 0 else ci / max(1, len(known) - 1)
    # Explicit affinity bits used by retrieval fallback.
    tags = set(str(x) for x in cls.get("music_alignment_tags", []))
    base[28] = 1.0 if "calm_meditative" in tags or str(cls.get("energy_label")) == "calm" else 0.0
    base[29] = 1.0 if "percussive_accent" in tags or str(cls.get("rhythm_label")) == "percussive" else 0.0
    base[30] = 1.0 if "turning_climax" in tags or str(cls.get("spatial_label")) == "turning" else 0.0
    base[31] = 1.0
    return base.astype(np.float32)













CHANG_E_CATEGORY_PROFILES = {
    "flying_apsaras": {"aliases": {"flying", "apsaras", "flying_apsara", "flying_apsaras", "feitian", "fei_tian", "sky_dance"}, "energy": 0.52, "onset": 0.28, "travel": 0.32, "turn": 0.38, "lower": 0.38, "upper": 0.72, "floorwork": 0.10, "jump": 0.35, "spin": 0.35, "pose_hold": 0.25, "instrument": 0.0, "prop": 0.85, "display": "Flying Apsaras", "semantic_role": "aerial_graceful_flow", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "upper_body", "spatial_label": "aerial_leaning", "music_alignment_label": "lyrical_flow", "music_alignment_tags": ["lyrical_flow", "turning_climax", "calm_meditative", "aerial_curve"], "preferred_music_roles": ["intro", "build_up", "climax"], "preferred_dance_keys": ["flying_apsaras", "sogdian_whirl", "lotus_steps"], "cultural_motif": "flying_apsara", "prop_proxy_label": "sash_ribbon_proxy", "locomotion_label": "floating_leaning", "support_label": "low_contact_flight_like", "event_family": "aerial_curve", "motion_stage_role": "opening_or_climax", "natural_duration_range_sec": [2.0, 5.5]},
    "lotus_steps": {"aliases": {"lotus", "lotussteps", "lotus_step", "lotus_steps"}, "energy": 0.48, "onset": 0.35, "travel": 0.62, "turn": 0.20, "lower": 0.78, "upper": 0.38, "floorwork": 0.05, "jump": 0.12, "spin": 0.10, "pose_hold": 0.20, "instrument": 0.0, "prop": 0.0, "display": "Lotus Steps", "semantic_role": "flowing_footwork", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "lower_body", "spatial_label": "traveling", "music_alignment_label": "footwork_flow", "music_alignment_tags": ["footwork_flow", "lyrical_flow", "calm_meditative"], "preferred_music_roles": ["normal", "development"], "preferred_dance_keys": ["lotus_steps", "flying_apsaras", "sogdian_whirl"], "cultural_motif": "lotus_step", "prop_proxy_label": "none", "locomotion_label": "traveling_steps", "support_label": "alternating_foot_support", "event_family": "footwork_flow", "motion_stage_role": "development", "natural_duration_range_sec": [1.5, 4.0]},
    "thirty_six_postures": {"aliases": {"36pose", "36posture", "36postures", "thirtysix", "thirty_six", "thirty_six_postures", "jiyuetian"}, "energy": 0.36, "onset": 0.18, "travel": 0.12, "turn": 0.12, "lower": 0.28, "upper": 0.42, "floorwork": 0.18, "jump": 0.02, "spin": 0.05, "pose_hold": 0.90, "instrument": 0.0, "prop": 0.0, "display": "Ji Yue Tian Thirty-Six Postures", "semantic_role": "iconic_pose_sequence", "energy_label": "moderate", "rhythm_label": "sustained", "body_focus_label": "pose", "spatial_label": "in_place", "music_alignment_label": "pose_hold", "music_alignment_tags": ["pose_hold", "calm_meditative", "lyrical_flow"], "preferred_music_roles": ["intro", "release", "resolution"], "preferred_dance_keys": ["thirty_six_postures", "revelation_meditation", "lotus_steps"], "cultural_motif": "jiyuetian_pose", "prop_proxy_label": "none", "locomotion_label": "in_place_pose", "support_label": "static_or_low_motion_support", "event_family": "pose_motif", "motion_stage_role": "anchor_or_resolution", "natural_duration_range_sec": [1.2, 3.8]},
    "revelation_meditation": {"aliases": {"meditation", "mediation", "revelation", "revelation_meditation", "revelation_mediation"}, "energy": 0.20, "onset": 0.08, "travel": 0.10, "turn": 0.08, "lower": 0.20, "upper": 0.36, "floorwork": 0.38, "jump": 0.0, "spin": 0.03, "pose_hold": 0.78, "instrument": 0.0, "prop": 0.0, "display": "Revelation Meditation", "semantic_role": "calm_meditative_flow", "energy_label": "calm", "rhythm_label": "sustained", "body_focus_label": "full_body", "spatial_label": "in_place", "music_alignment_label": "calm_meditative", "music_alignment_tags": ["calm_meditative", "pose_hold", "lyrical_flow"], "preferred_music_roles": ["intro", "calm", "release", "resolution"], "preferred_dance_keys": ["revelation_meditation", "thirty_six_postures", "flying_apsaras"], "cultural_motif": "buddhist_meditation", "prop_proxy_label": "none", "locomotion_label": "slow_weight_shift", "support_label": "stable_support", "event_family": "calm_flow", "motion_stage_role": "intro_or_resolution", "natural_duration_range_sec": [2.0, 6.0]},
    "sogdian_whirl": {"aliases": {"ribbon", "ribbon_flow", "sash", "silk", "whirl", "sogdian", "sogdian_whirl", "turn", "turning"}, "energy": 0.72, "onset": 0.40, "travel": 0.50, "turn": 0.90, "lower": 0.68, "upper": 0.65, "floorwork": 0.02, "jump": 0.20, "spin": 0.95, "pose_hold": 0.15, "instrument": 0.0, "prop": 0.75, "display": "Sogdian Whirl / Ribbon Flow", "semantic_role": "flowing_turning_motif", "energy_label": "high", "rhythm_label": "lyrical", "body_focus_label": "turning_flow", "spatial_label": "turning", "music_alignment_label": "turning_climax", "music_alignment_tags": ["turning_climax", "lyrical_flow", "footwork_flow"], "preferred_music_roles": ["build_up", "climax"], "preferred_dance_keys": ["sogdian_whirl", "flying_apsaras", "lotus_steps"], "cultural_motif": "sogdian_whirl", "prop_proxy_label": "ribbon_sash_proxy", "locomotion_label": "turning_travel", "support_label": "alternating_or_pivot_support", "event_family": "turning_flow", "motion_stage_role": "climax", "natural_duration_range_sec": [1.6, 4.5]},
    "pipa_behind_back": {"aliases": {"pipa", "pipa1", "pipa2", "playing_pipa", "playing_the_pipa", "pipa_behind_back"}, "energy": 0.46, "onset": 0.42, "travel": 0.16, "turn": 0.20, "lower": 0.30, "upper": 0.82, "floorwork": 0.06, "jump": 0.05, "spin": 0.10, "pose_hold": 0.45, "instrument": 1.0, "prop": 0.70, "display": "Playing the Pipa Behind the Back", "semantic_role": "instrument_upper_body_motif", "energy_label": "moderate", "rhythm_label": "accented", "body_focus_label": "upper_body", "spatial_label": "in_place", "music_alignment_label": "instrument_phrase", "music_alignment_tags": ["instrument_phrase", "lyrical_flow", "percussive_accent"], "preferred_music_roles": ["motif", "normal", "build_up"], "preferred_dance_keys": ["pipa_behind_back", "sogdian_whirl", "lei_gong_drum"], "cultural_motif": "pipa_instrument_pose", "prop_proxy_label": "pipa_proxy", "locomotion_label": "upper_body_phrase", "support_label": "stable_support", "event_family": "instrument_motif", "motion_stage_role": "motif_recall", "natural_duration_range_sec": [1.6, 4.5]},
    "lei_gong_drum": {"aliases": {"drum", "lei_gong", "leigong", "lei_gong_drum"}, "energy": 0.82, "onset": 0.88, "travel": 0.52, "turn": 0.35, "lower": 0.75, "upper": 0.76, "floorwork": 0.04, "jump": 0.32, "spin": 0.20, "pose_hold": 0.10, "instrument": 0.65, "prop": 0.55, "display": "Lei Gong Drum", "semantic_role": "percussive_high_energy", "energy_label": "percussive", "rhythm_label": "percussive", "body_focus_label": "full_body", "spatial_label": "traveling", "music_alignment_label": "percussive_accent", "music_alignment_tags": ["percussive_accent", "turning_climax", "footwork_flow"], "preferred_music_roles": ["accent", "climax"], "preferred_dance_keys": ["lei_gong_drum", "pipa_behind_back", "sogdian_whirl"], "cultural_motif": "thunder_drum", "prop_proxy_label": "drum_proxy", "locomotion_label": "accented_travel", "support_label": "strong_foot_contact", "event_family": "percussive_accent", "motion_stage_role": "accent_or_climax", "natural_duration_range_sec": [1.2, 3.5]},
    "unknown": {"aliases": set(), "energy": 0.45, "onset": 0.30, "travel": 0.30, "turn": 0.20, "lower": 0.45, "upper": 0.45, "floorwork": 0.0, "jump": 0.0, "spin": 0.0, "pose_hold": 0.25, "instrument": 0.0, "prop": 0.0, "display": "Unknown Chang-E Motion", "semantic_role": "unknown_motion", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "full_body", "spatial_label": "in_place", "music_alignment_label": "lyrical_flow", "music_alignment_tags": ["lyrical_flow"], "preferred_music_roles": ["normal"], "preferred_dance_keys": ["lotus_steps", "thirty_six_postures"], "cultural_motif": "unknown", "prop_proxy_label": "unknown", "locomotion_label": "unknown", "support_label": "unknown", "event_family": "unknown", "motion_stage_role": "development", "natural_duration_range_sec": [1.5, 4.0]},
}

ENERGY_LABELS = ["calm", "moderate", "high", "percussive"]
RHYTHM_LABELS = ["sustained", "lyrical", "accented", "percussive"]
BODY_FOCUS_LABELS = ["pose", "lower_body", "upper_body", "full_body", "turning_flow"]
SPATIAL_LABELS = ["in_place", "traveling", "turning", "aerial_leaning"]
MUSIC_ALIGNMENT_LABELS = ["calm_meditative", "lyrical_flow", "pose_hold", "instrument_phrase", "percussive_accent", "turning_climax", "footwork_flow", "aerial_curve"]
EVENT_FAMILY_LABELS = ["calm_flow", "pose_motif", "footwork_flow", "turning_flow", "instrument_motif", "percussive_accent", "aerial_curve", "unknown"]
STAGE_ROLE_LABELS = ["intro", "development", "build_up", "motif_recall", "anchor_or_resolution", "intro_or_resolution", "opening_or_climax", "accent_or_climax", "climax", "resolution"]
CATEGORY_CLASS_OVERRIDES = {}


def canonicalize_chang_e_key(key: object) -> str:
    key_s = str(key or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        key_s = re.sub(r"_take\d+$", "", key_s)
    except Exception:
        pass
    aliases = {"mediation": "revelation_meditation", "female_mediation": "revelation_meditation", "male_mediation": "revelation_meditation", "meditation": "revelation_meditation", "36pose": "thirty_six_postures", "36postures": "thirty_six_postures", "thirtysix": "thirty_six_postures", "lotus": "lotus_steps", "pipa": "pipa_behind_back", "drum": "lei_gong_drum", "leigong": "lei_gong_drum", "ribbon": "sogdian_whirl", "ribbon_flow": "sogdian_whirl", "sogdian": "sogdian_whirl", "whirl": "sogdian_whirl", "flying": "flying_apsaras", "apsaras": "flying_apsaras", "feitian": "flying_apsaras"}
    if key_s in aliases:
        return aliases[key_s]
    for k, prof in CHANG_E_CATEGORY_PROFILES.items():
        if key_s == k or key_s in set(prof.get("aliases", set())):
            return k
    return key_s if key_s in CHANG_E_CATEGORY_PROFILES else "unknown"


def _safe_profile_key(meta: dict) -> str:
    return canonicalize_chang_e_key(meta.get("dance_key") or meta.get("parent_label") or meta.get("label") or meta.get("source_bvh") or "unknown")


def _label_index(label: str, labels: Sequence[str]) -> int:
    try:
        return list(labels).index(str(label))
    except ValueError:
        return -1


def _parse_numeric_semantic(meta: dict) -> Dict[str, float]:
    keys = ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]
    vals = [x for x in re.split(r"[;, ]+", str(meta.get("semantic_numeric", "") or "")) if x]
    out = {}
    for k, v in zip(keys, vals):
        try: out[k] = float(v)
        except Exception: pass
    return out


def strong_action_semantics_from_meta(meta: dict, desc: Optional[np.ndarray] = None) -> Dict[str, object]:
    key = _safe_profile_key(meta)
    base_prof = dict(CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"]))
    numeric = {k: float(base_prof.get(k, 0.0)) for k in ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]}
    numeric.update(_parse_numeric_semantic(meta))
    if desc is not None and len(desc) >= 20:
        numeric["travel"] = max(numeric["travel"], float(np.clip(desc[1] / 1.2, 0.0, 1.0)))
        numeric["energy"] = max(numeric["energy"], float(np.clip(desc[5] / 0.14, 0.0, 1.0)))
        numeric["lower"] = max(numeric["lower"], float(np.clip(desc[7] / 0.10, 0.0, 1.0)))
        numeric["upper"] = max(numeric["upper"], float(np.clip(desc[8] / 0.10, 0.0, 1.0)))
        numeric["turn"] = max(numeric["turn"], float(np.clip(abs(desc[17]) / 0.22, 0.0, 1.0)))
        numeric["jump"] = max(numeric["jump"], float(np.clip(desc[18] / 0.20, 0.0, 1.0)))
        numeric["pose_hold"] = max(numeric["pose_hold"], float(np.clip(1.0 - desc[5] / 0.12, 0.0, 1.0)))
    prof = dict(base_prof)
    for field in ["energy_label", "rhythm_label", "body_focus_label", "spatial_label", "music_alignment_label", "semantic_role", "cultural_motif", "prop_proxy_label", "locomotion_label", "support_label", "event_family", "motion_stage_role"]:
        if meta.get(field):
            prof[field] = str(meta.get(field))
    if meta.get("music_alignment_tags"):
        prof["music_alignment_tags"] = [x for x in re.split(r"[;|,]", str(meta.get("music_alignment_tags"))) if x]
    if meta.get("preferred_dance_keys"):
        prof["preferred_dance_keys"] = [canonicalize_chang_e_key(x) for x in re.split(r"[;|,]", str(meta.get("preferred_dance_keys"))) if x]
    tags = list(dict.fromkeys([str(prof.get("music_alignment_label"))] + [str(x) for x in prof.get("music_alignment_tags", [])]))
    prof["music_alignment_tags"] = tags
    prof.setdefault("preferred_music_roles", base_prof.get("preferred_music_roles", ["normal"]))
    prof.setdefault("preferred_dance_keys", base_prof.get("preferred_dance_keys", [key]))
    prof["semantic_numeric"] = ";".join(str(float(numeric[k])) for k in ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"])
    prof["classification_text"] = f"action={key}; motif={prof.get('cultural_motif')}; family={prof.get('event_family')}; stage={prof.get('motion_stage_role')}; music_align={prof.get('music_alignment_label')}; numeric={prof['semantic_numeric']}"
    return refine_chang_e_event_semantics(meta, desc, prof)


def _float_meta(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        v = meta.get(key, default)
        if v is None or str(v).lower() in {"nan", "none", "null", ""}:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _bounded01(x: float) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.0


def chang_e_event_quality_from_numbers(nums: Dict[str, float], family: str, duration: float, natural_range: Sequence[float]) -> float:
    """Quality gate for converting long Chang-E BVH into 72BVH-like RAG events."""
    energy = _bounded01(nums.get("energy", 0.0)); travel = _bounded01(nums.get("travel", 0.0))
    turn = _bounded01(nums.get("turn", 0.0)); lower = _bounded01(nums.get("lower", 0.0)); upper = _bounded01(nums.get("upper", 0.0))
    pose_hold = _bounded01(nums.get("pose_hold", 0.0)); jump = _bounded01(nums.get("jump", 0.0)); onset = _bounded01(nums.get("onset", 0.0))
    contact_ratio = _bounded01(nums.get("contact_ratio", 0.5))
    root_y = max(0.0, float(nums.get("root_y_range", 0.0)))
    lo, hi = 1.5, 4.0
    try:
        if natural_range and len(natural_range) >= 2:
            lo, hi = float(natural_range[0]), float(natural_range[-1])
    except Exception:
        pass
    dur = max(1e-3, float(duration or 0.0))
    center = max(1e-3, 0.5 * (lo + hi))
    dur_score = 1.0 if (lo <= dur <= hi) else float(np.exp(-abs(np.log(dur / center))))
    content = max(energy, travel, turn, lower, upper, onset, jump)
    if family in {"pose_motif", "calm_flow"}:
        content = max(content * 0.65, pose_hold)
    # V46.31: stationary Dunhuang postures / meditation motifs are supposed to
    # have long stable support. Do not score contact_ratio=1.0 as bad gait.
    if family in {"pose_motif", "calm_flow"} or pose_hold > 0.70:
        contact_score = 1.0 if contact_ratio >= 0.70 else float(contact_ratio / 0.70)
    else:
        contact_score = 1.0 - min(1.0, abs(contact_ratio - 0.46) / 0.54)
    root_y_penalty = max(0.0, min(0.25, (root_y - 0.35) * 0.35))
    dead_penalty = 0.0
    if family not in {"pose_motif", "calm_flow"} and content < 0.20 and pose_hold < 0.45:
        dead_penalty = 0.25
    q = 0.42 * content + 0.22 * pose_hold + 0.20 * dur_score + 0.16 * contact_score - root_y_penalty - dead_penalty
    return float(np.clip(q, 0.02, 1.0))


def chang_e_semantic_event_starts(seq: np.ndarray, cfg: V46Config) -> List[int]:
    """Boundary-aware starts for Chang-E long BVH.

    The old 72BVH data behaved well because each file was already a compact
    action unit. Chang-E files are long performances, so we preserve uniform
    coverage and add motion-novelty anchors around energy/yaw/contact changes.
    """
    x = np.asarray(seq, dtype=np.float32)
    T = int(x.shape[0])
    win = max(1, min(int(getattr(cfg, "window_len", 120)), T))
    hop = max(1, int(getattr(cfg, "hop_len", max(1, win // 2))))
    minf = max(1, int(getattr(cfg, "min_event_frames", 45)))
    if T <= max(win, minf):
        return [0]
    starts = set([0, max(0, T - win)])
    for st in range(0, max(1, T - minf + 1), hop):
        starts.add(int(min(max(0, st), max(0, T - minf))))
    try:
        joints = fk_24_np(x)
        v = np.zeros_like(joints)
        if T > 1:
            v[1:] = joints[1:] - joints[:-1]
        energy = np.linalg.norm(v.reshape(T, -1, 3), axis=-1).mean(axis=1)
        root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
        root_v = np.zeros((T,), dtype=np.float32)
        if T > 1:
            root_v[1:] = np.linalg.norm(root[1:, [0, 2]] - root[:-1, [0, 2]], axis=-1)
        yaw = root_yaw_np(x)
        yaw_v = np.zeros((T,), dtype=np.float32)
        if T > 1:
            yaw_v[1:] = np.abs(angle_diff(yaw[1:], yaw[:-1]))
        foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
        floor = np.percentile(foot[..., 1].reshape(-1), 5)
        near = (foot[..., 1] < floor + 0.05).mean(axis=1)
        novelty = energy + 0.65 * root_v + 0.55 * yaw_v + 0.20 * np.abs(np.diff(near, prepend=near[:1]))
        if novelty.size > 7:
            k = np.ones(5, dtype=np.float32) / 5.0
            novelty = np.convolve(novelty, k, mode="same")
        thr = float(np.percentile(novelty, 72)) if novelty.size else 0.0
        order = np.argsort(-novelty)
        extra = 0
        max_extra = int(getattr(cfg, "chang_e_boundary_max_extra_starts", 96))
        min_sep = max(8, min(hop, win // 3))
        selected_centers: List[int] = []
        for c in order.tolist():
            if novelty[int(c)] < thr or extra >= max_extra:
                break
            c = int(c)
            if any(abs(c - q) < min_sep for q in selected_centers):
                continue
            selected_centers.append(c)
            starts.add(int(np.clip(c - win // 2, 0, max(0, T - minf))))
            starts.add(int(np.clip(c - win // 3, 0, max(0, T - minf))))
            extra += 2
    except Exception:
        pass
    out = sorted(starts)
    merged: List[int] = []
    min_start_sep = max(6, min(hop // 2, win // 4))
    tail = max(0, T - win)
    # V46.31: if the sequence is only slightly longer than one window, [0:win]
    # and [tail:T] are >95% overlapping twins. Keep one centered representative
    # rather than polluting the RAG DB with near-identical embeddings.
    if 0 < tail < min_start_sep:
        return [int(max(0, tail // 2))]
    for st in out:
        st = int(st)
        # V46.31: keep coverage endpoints without creating near-duplicate twins.
        # When a novelty start lies too close to the required tail window, replace
        # the previous start with the exact tail start instead of appending both.
        if st == 0:
            if not merged:
                merged.append(0)
            continue
        if st == tail:
            if merged and abs(st - merged[-1]) < min_start_sep:
                # V46.31: protect the unique opener for very short sequences.
                # If T is only slightly larger than win, tail can be only a few
                # frames after 0. Replacing [0] with [tail] permanently drops
                # opening frames. Keep the opener and add the tail only in this
                # unique two-endpoint case; otherwise replace the near-duplicate.
                if len(merged) == 1 and merged[-1] == 0 and st != 0:
                    merged.append(st)
                elif not (len(merged) == 1 and merged[-1] == 0):
                    merged[-1] = st
            elif not merged or merged[-1] != st:
                merged.append(st)
            continue
        if not merged or abs(st - merged[-1]) >= min_start_sep:
            merged.append(st)
    if merged:
        if abs(tail - merged[-1]) < min_start_sep:
            if len(merged) == 1 and merged[-1] == 0 and tail != 0:
                merged.append(tail)
            elif not (len(merged) == 1 and merged[-1] == 0):
                merged[-1] = tail
        elif merged[-1] != tail:
            merged.append(tail)
    else:
        merged = [0, tail] if tail > 0 else [0]
    return sorted(set(int(x) for x in merged))


def refine_chang_e_event_semantics(meta: dict, desc: Optional[np.ndarray], prof: Dict[str, object]) -> Dict[str, object]:
    # V46.31 window-level semantics for Chang-E event slicing.
    # Chang-E is a long, category-complete MoCap corpus; each local window is
    # converted into a curated semantic event comparable to the old 72BVH units.
    out = dict(prof)
    key = _safe_profile_key(meta)
    nums = _parse_numeric_semantic(out)
    if desc is not None and len(desc) >= 20:
        nums["duration"] = float(desc[0])
        nums["travel"] = max(float(nums.get("travel", 0.0)), float(np.clip(desc[1] / 1.15, 0.0, 1.0)))
        nums["energy"] = max(float(nums.get("energy", 0.0)), float(np.clip(desc[5] / 0.135, 0.0, 1.0)))
        nums["lower"] = max(float(nums.get("lower", 0.0)), float(np.clip(desc[7] / 0.10, 0.0, 1.0)))
        nums["upper"] = max(float(nums.get("upper", 0.0)), float(np.clip(desc[8] / 0.10, 0.0, 1.0)))
        nums["turn"] = max(float(nums.get("turn", 0.0)), float(np.clip(abs(desc[17]) / 0.20, 0.0, 1.0)))
        nums["root_y_range"] = float(desc[18])
        nums["contact_ratio"] = float(desc[10])
        local_hold = float(np.clip(1.0 - desc[5] / 0.115, 0.0, 1.0)) * float(np.clip(0.55 + desc[10], 0.0, 1.0))
        nums["pose_hold"] = max(float(nums.get("pose_hold", 0.0)), local_hold)
        nums["jump"] = max(float(nums.get("jump", 0.0)), float(np.clip((desc[18] - 0.035) / 0.16, 0.0, 1.0)))
        nums["spin"] = max(float(nums.get("spin", 0.0)), float(np.clip(abs(desc[17]) / 0.22, 0.0, 1.0)))
    frac_mid = _float_meta(meta, "event_position_mid", _float_meta(meta, "event_position_fraction", 0.5))
    duration = float(nums.get("duration", _float_meta(meta, "duration", 0.0)))
    energy = float(nums.get("energy", 0.0)); onset = float(nums.get("onset", 0.0))
    travel = float(nums.get("travel", 0.0)); turn = float(nums.get("turn", 0.0))
    lower = float(nums.get("lower", 0.0)); upper = float(nums.get("upper", 0.0))
    pose_hold = float(nums.get("pose_hold", 0.0)); jump = float(nums.get("jump", 0.0)); spin = float(nums.get("spin", 0.0))
    contact_ratio = float(nums.get("contact_ratio", 0.5))
    if pose_hold > 0.72 and energy < 0.60:
        family = "pose_motif"
    elif turn > 0.68 or spin > 0.68:
        family = "turning_flow"
    elif onset > 0.72 or (energy > 0.78 and key == "lei_gong_drum"):
        family = "percussive_accent"
    elif key == "pipa_behind_back" and upper >= lower * 1.12:
        family = "instrument_motif"
    elif travel > 0.50 or lower > upper * 1.18:
        family = "footwork_flow"
    elif key == "flying_apsaras" or jump > 0.45:
        family = "aerial_curve"
    elif energy < 0.34:
        family = "calm_flow"
    else:
        family = str(out.get("event_family", "footwork_flow"))
    out["event_family"] = family
    if frac_mid < 0.12:
        stage = "intro"
    elif frac_mid > 0.88:
        stage = "resolution"
    elif family in {"percussive_accent", "turning_flow"} and energy > 0.55:
        stage = "climax"
    elif family == "instrument_motif":
        stage = "motif_recall"
    elif pose_hold > 0.70:
        stage = "anchor_or_resolution"
    elif energy > 0.58 or travel > 0.55:
        stage = "build_up"
    else:
        stage = "development"
    out["motion_stage_role"] = stage
    if contact_ratio < 0.18 or jump > 0.45:
        out["support_label"] = "low_contact_flight_like"
    elif pose_hold > 0.72:
        out["support_label"] = "stable_support"
    elif turn > 0.55:
        out["support_label"] = "alternating_or_pivot_support"
    elif travel > 0.45 or lower > 0.60:
        out["support_label"] = "alternating_foot_support"
    else:
        out.setdefault("support_label", "stable_support")
    if turn > 0.60:
        out["locomotion_label"] = "turning_travel"; out["spatial_label"] = "turning"
    elif travel > 0.50:
        out["locomotion_label"] = "traveling_steps"; out["spatial_label"] = "traveling"
    elif pose_hold > 0.70:
        out["locomotion_label"] = "in_place_pose"; out["spatial_label"] = "in_place"
    elif energy < 0.34:
        out["locomotion_label"] = "slow_weight_shift"; out["spatial_label"] = "in_place"
    family_to_align = {"calm_flow": "calm_meditative", "pose_motif": "pose_hold", "footwork_flow": "footwork_flow", "turning_flow": "turning_climax", "instrument_motif": "instrument_phrase", "percussive_accent": "percussive_accent", "aerial_curve": "lyrical_flow"}
    out["music_alignment_label"] = family_to_align.get(family, str(out.get("music_alignment_label", "lyrical_flow")))
    tags = [out["music_alignment_label"], family, stage, str(out.get("support_label", "")), str(out.get("locomotion_label", ""))]
    tags += [str(x) for x in out.get("music_alignment_tags", [])]
    out["music_alignment_tags"] = list(dict.fromkeys([x for x in tags if x]))
    out["event_position_mid"] = float(frac_mid)
    natural_range = out.get("natural_duration_range_sec", CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"]).get("natural_duration_range_sec", [1.5, 4.0]))
    q = chang_e_event_quality_from_numbers(nums, family, duration, natural_range)
    out["event_quality_score"] = float(q)
    out["semantic_confidence"] = float(np.clip(0.25 + 0.50 * q + 0.25 * max(energy, pose_hold, turn, travel, upper, lower), 0.10, 1.0))
    keys = ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]
    out["semantic_numeric"] = ";".join(str(float(nums.get(k, 0.0))) for k in keys)
    out["classification_text"] = (
        f"action={key}; motif={out.get('cultural_motif')}; family={out.get('event_family')}; "
        f"stage={out.get('motion_stage_role')}; support={out.get('support_label')}; "
        f"locomotion={out.get('locomotion_label')}; music_align={out.get('music_alignment_label')}; "
        f"event_mid={float(frac_mid):.3f}; quality={q:.3f}; semantic_conf={out['semantic_confidence']:.3f}; numeric={out['semantic_numeric']}"
    )
    return out


def class_semantic_vector_from_meta(meta: dict, cfg: Optional[V46Config] = None) -> np.ndarray:
    key = _safe_profile_key(meta)
    base = filename_semantic_vector_from_meta(meta, cfg).copy()
    cls = strong_action_semantics_from_meta(meta)
    nums = _parse_numeric_semantic(cls)
    for k in ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]:
        nums.setdefault(k, float(CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"]).get(k, 0.0)))
    known = [k for k in CHANG_E_CATEGORY_PROFILES.keys() if k != "unknown"]
    ci = known.index(key) if key in known else -1
    align_i = _label_index(str(cls.get("music_alignment_label")), MUSIC_ALIGNMENT_LABELS)
    family_i = _label_index(str(cls.get("event_family")), EVENT_FAMILY_LABELS)
    stage_i = _label_index(str(cls.get("motion_stage_role")), STAGE_ROLE_LABELS)
    v = base.astype(np.float32)
    for i,k in enumerate(["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]):
        v[8+i] = nums[k]
    v[20] = 0.0 if ci < 0 else ci / max(1, len(known) - 1)
    v[21] = 0.0 if align_i < 0 else align_i / max(1, len(MUSIC_ALIGNMENT_LABELS) - 1)
    v[22] = 0.0 if family_i < 0 else family_i / max(1, len(EVENT_FAMILY_LABELS) - 1)
    v[23] = 0.0 if stage_i < 0 else stage_i / max(1, len(STAGE_ROLE_LABELS) - 1)
    tags = set(str(x) for x in cls.get("music_alignment_tags", []))
    v[28] = 1.0 if ("calm_meditative" in tags or cls.get("event_family") == "calm_flow") else 0.0
    v[29] = 1.0 if ("percussive_accent" in tags or nums["onset"] > 0.65) else 0.0
    v[30] = 1.0 if ("turning_climax" in tags or nums["spin"] > 0.65 or nums["turn"] > 0.65) else 0.0
    v[31] = 1.0
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def audio_slot_classification_from_pseudo(pseudo: np.ndarray, duration: float, energy: float, onset: float, dyn: float) -> Dict[str, object]:
    """Infer slot-level music labels for unpaired alignment and schedule reports."""
    if energy < 0.035 and onset < 0.015:
        label = "calm_meditative"
        role = "calm"
        preferred = ["revelation_meditation", "thirty_six_postures", "lotus_steps"]
    elif onset > 0.10 or (energy > 0.08 and dyn > 0.04):
        label = "percussive_accent"
        role = "climax" if energy > 0.08 else "build_up"
        preferred = ["lei_gong_drum", "pipa_behind_back", "ribbon_flow"]
    elif energy > 0.065:
        label = "turning_climax"
        role = "build_up"
        preferred = ["ribbon_flow", "pipa_behind_back", "lei_gong_drum"]
    elif duration > 5.0 and energy < 0.055:
        label = "pose_hold"
        role = "release"
        preferred = ["thirty_six_postures", "revelation_meditation", "lotus_steps"]
    else:
        label = "lyrical_flow"
        role = "normal"
        preferred = ["lotus_steps", "thirty_six_postures", "ribbon_flow", "pipa_behind_back"]
    if label == "percussive_accent":
        energy_label, rhythm_label = "high", "percussive"
    elif label == "calm_meditative":
        energy_label, rhythm_label = "calm", "sustained"
    elif label == "pose_hold":
        energy_label, rhythm_label = "calm", "sustained"
    else:
        energy_label, rhythm_label = "moderate", "lyrical"
    return {
        "role": role,
        "music_alignment_label": label,
        "energy_label": energy_label,
        "rhythm_label": rhythm_label,
        "preferred_dance_keys": preferred,
        "preferred_semantic_roles": [CHANG_E_CATEGORY_PROFILES.get(k, {}).get("semantic_role", CATEGORY_CLASS_OVERRIDES.get(k, {}).get("semantic_role", "")) for k in preferred],
    }




# V46.12 external classical-music semantic interface.
# Supported labels are deliberately identical to RAG event music_alignment labels,
# so a trained classical model can supervise retrieval without paired BVH/audio.
MUSIC_SEMANTIC_LABELS = MUSIC_ALIGNMENT_LABELS
MUSIC_LABEL_TO_ACTION = {
    "calm_meditative": ["revelation_meditation", "thirty_six_postures", "lotus_steps"],
    "pose_hold": ["thirty_six_postures", "revelation_meditation", "lotus_steps"],
    "lyrical_flow": ["lotus_steps", "ribbon_flow", "revelation_meditation", "thirty_six_postures"],
    "instrument_phrase": ["pipa_behind_back", "lotus_steps", "ribbon_flow"],
    "percussive_accent": ["lei_gong_drum", "pipa_behind_back", "ribbon_flow"],
    "turning_climax": ["ribbon_flow", "lei_gong_drum", "pipa_behind_back"],
    "footwork_flow": ["lotus_steps", "ribbon_flow", "thirty_six_postures"],
}
MUSIC_PROXY_NAME_LABELS = {
    "pipa": "instrument_phrase",
    "guzhen": "lyrical_flow",
    "guzheng": "lyrical_flow",
    "xiao": "calm_meditative",
    "gu": "percussive_accent",
    "drum": "percussive_accent",
    "luo": "percussive_accent",
    "gong": "percussive_accent",
}


def _split_path_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts: List[str] = []
    for chunk in text.replace(";", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def canonical_music_label(label: object) -> str:
    text = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "meditation": "calm_meditative",
        "meditative": "calm_meditative",
        "calm": "calm_meditative",
        "tranquil": "calm_meditative",
        "hold": "pose_hold",
        "pose": "pose_hold",
        "sustain": "pose_hold",
        "lyrical": "lyrical_flow",
        "flow": "lyrical_flow",
        "melodic": "lyrical_flow",
        "instrument": "instrument_phrase",
        "pipa": "instrument_phrase",
        "plucked": "instrument_phrase",
        "percussive": "percussive_accent",
        "accent": "percussive_accent",
        "drum": "percussive_accent",
        "climax": "turning_climax",
        "turn": "turning_climax",
        "turning": "turning_climax",
        "footwork": "footwork_flow",
        "steps": "footwork_flow",
        "step": "footwork_flow",
    }
    if text in aliases:
        return aliases[text]
    if text in MUSIC_SEMANTIC_LABELS:
        return text
    return text if text else "lyrical_flow"


def normalize_music_probs(probs: object = None, top_label: object = None, temperature: float = 0.65) -> Dict[str, float]:
    out = {k: 0.0 for k in MUSIC_SEMANTIC_LABELS}
    if isinstance(probs, dict):
        for k, v in probs.items():
            ck = canonical_music_label(k)
            if ck in out:
                try:
                    out[ck] += max(0.0, float(v))
                except Exception:
                    pass
    elif probs is not None:
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        for i, v in enumerate(arr[:len(MUSIC_SEMANTIC_LABELS)]):
            out[MUSIC_SEMANTIC_LABELS[i]] += max(0.0, float(v))
    if sum(out.values()) <= 1e-8 and top_label is not None:
        label = canonical_music_label(top_label)
        if label in out:
            out[label] = 1.0
    if sum(out.values()) <= 1e-8:
        out["lyrical_flow"] = 1.0
    # Temperature-sharpen while retaining soft alternatives from external model.
    temp = max(0.05, float(temperature))
    vals = np.asarray([out[k] for k in MUSIC_SEMANTIC_LABELS], dtype=np.float32)
    vals = np.power(np.maximum(vals, 0.0) + 1e-8, 1.0 / temp)
    vals = vals / max(float(vals.sum()), 1e-8)
    return {k: float(vals[i]) for i, k in enumerate(MUSIC_SEMANTIC_LABELS)}


def music_semantic_slot_from_probs(probs: Dict[str, float], duration: float = 4.0, source: str = "external") -> Dict[str, object]:
    top_label = max(probs.items(), key=lambda kv: kv[1])[0]
    preferred = MUSIC_LABEL_TO_ACTION.get(top_label, ["lotus_steps", "thirty_six_postures"])
    if top_label == "percussive_accent":
        role, energy_label, rhythm_label = "climax", "percussive", "percussive"
    elif top_label == "turning_climax":
        role, energy_label, rhythm_label = "build_up", "high", "accented"
    elif top_label in {"calm_meditative", "pose_hold"}:
        role, energy_label, rhythm_label = "calm" if top_label == "calm_meditative" else "release", "calm", "sustained"
    elif top_label == "instrument_phrase":
        role, energy_label, rhythm_label = "normal", "moderate", "accented"
    elif top_label == "footwork_flow":
        role, energy_label, rhythm_label = "normal", "moderate", "lyrical"
    else:
        role, energy_label, rhythm_label = "normal", "moderate", "lyrical"
    return {
        "role": role,
        "music_alignment_label": top_label,
        "energy_label": energy_label,
        "rhythm_label": rhythm_label,
        "preferred_dance_keys": preferred,
        "preferred_semantic_roles": [str(CHANG_E_CATEGORY_PROFILES.get(k, {}).get("semantic_role", "")) for k in preferred],
        "music_semantic_probs": {k: float(v) for k, v in probs.items()},
        "music_semantic_top_label": top_label,
        "external_music_semantic_source": source,
    }


def music_probs_to_pseudo_feature(probs: Dict[str, float], duration: float, cfg: V46Config) -> np.ndarray:
    # Label prototypes occupy the same descriptor layout used by motion events.
    prototypes = {
        "calm_meditative": (0.020, 0.010, 0.012, 0.85),
        "pose_hold": (0.030, 0.012, 0.010, 0.78),
        "lyrical_flow": (0.055, 0.035, 0.040, 0.42),
        "instrument_phrase": (0.070, 0.065, 0.055, 0.30),
        "percussive_accent": (0.105, 0.125, 0.100, 0.12),
        "turning_climax": (0.095, 0.080, 0.110, 0.08),
        "footwork_flow": (0.065, 0.045, 0.070, 0.34),
    }
    energy = onset = dyn = calm = 0.0
    for label, p in probs.items():
        e, o, d, c = prototypes.get(label, prototypes["lyrical_flow"])
        energy += float(p) * e
        onset += float(p) * o
        dyn += float(p) * d
        calm += float(p) * c
    pseudo = np.zeros(32, dtype=np.float32)
    pseudo[0] = float(duration)
    pseudo[1] = energy * 2.0
    pseudo[2] = energy
    pseudo[3] = max(energy, dyn)
    pseudo[4] = dyn
    pseudo[5] = energy + onset
    pseudo[6] = onset + dyn
    pseudo[7] = energy + 0.5 * onset
    pseudo[8] = energy
    pseudo[9] = 1.0 + onset
    pseudo[10] = calm
    pseudo[13] = max(0.02, onset)
    pseudo[14] = onset
    pseudo[16] = onset
    pseudo[17] = dyn
    pseudo[18] = dyn
    sem = music_semantic_slot_from_probs(probs, duration, source="external")
    pseudo[22] = _label_index(str(sem["energy_label"]), ENERGY_LABELS) / max(1, len(ENERGY_LABELS) - 1)
    pseudo[23] = _label_index(str(sem["rhythm_label"]), RHYTHM_LABELS) / max(1, len(RHYTHM_LABELS) - 1)
    pseudo[26] = _label_index(str(sem["music_alignment_label"]), MUSIC_ALIGNMENT_LABELS) / max(1, len(MUSIC_ALIGNMENT_LABELS) - 1)
    pseudo[28] = float(probs.get("calm_meditative", 0.0) + 0.5 * probs.get("pose_hold", 0.0))
    pseudo[29] = float(probs.get("percussive_accent", 0.0) + 0.4 * probs.get("instrument_phrase", 0.0))
    pseudo[30] = float(probs.get("turning_climax", 0.0) + 0.3 * probs.get("footwork_flow", 0.0))
    pseudo[31] = 1.0
    return pseudo.astype(np.float32)


def sidecar_music_semantic_candidates(audio_path: str | Path, cfg: V46Config) -> List[Path]:
    p = Path(audio_path)
    names = [
        f"{p.stem}.music_semantic.json", f"{p.stem}_music_semantic.json", f"{p.stem}.semantic.json", f"{p.stem}_semantic.json",
        f"{p.stem}.music_semantic.npz", f"{p.stem}_music_semantic.npz", f"{p.stem}.semantic.npz", f"{p.stem}_semantic.npz",
        f"{p.stem}.json", f"{p.stem}.npz",
    ]
    cands: List[Path] = [p.with_name(n) for n in names]
    for d in _split_path_list(getattr(cfg, "external_music_semantic_dirs", "")):
        dp = Path(d)
        for n in names:
            cands.append(dp / n)
    return cands


def run_external_music_semantic_cmd(audio_path: str | Path, cfg: V46Config) -> Optional[Path]:
    cmd = str(getattr(cfg, "external_music_semantic_cmd", "") or "").strip()
    if not cmd:
        return None
    cache = ensure_dir(getattr(cfg, "external_music_semantic_cache_dir", "output/v46_external_music_semantic_cache"))
    audio_p = Path(audio_path)
    out_json = cache / f"{audio_p.stem}.music_semantic.json"
    out_npz = cache / f"{audio_p.stem}.music_semantic.npz"
    if out_json.exists() or out_npz.exists():
        return out_json if out_json.exists() else out_npz
    expanded = cmd.format(audio=str(audio_p), out_json=str(out_json), out_npz=str(out_npz), stem=audio_p.stem)
    try:
        subprocess.run(expanded, shell=True, check=True)
    except Exception as exc:
        print(f"[V46.12 WARN] external music semantic command failed for {audio_p}: {exc}", file=sys.stderr)
        return None
    if out_json.exists() or out_npz.exists():
        return out_json if out_json.exists() else out_npz
    return None


def parse_external_music_semantic_file(path: str | Path, cfg: V46Config) -> Optional[Tuple[List[dict], np.ndarray]]:
    p = Path(path)
    try:
        if p.suffix.lower() == ".npz":
            data = np.load(p, allow_pickle=True)
            label_names = [canonical_music_label(x) for x in (data["label_names"] if "label_names" in data.files else np.asarray(MUSIC_SEMANTIC_LABELS, dtype=object)).tolist()]
            probs_arr = np.asarray(data["slot_probs"] if "slot_probs" in data.files else data["probs"], dtype=np.float32)
            starts = np.asarray(data["slot_start"] if "slot_start" in data.files else (data["start"] if "start" in data.files else np.arange(len(probs_arr))*4.0), dtype=np.float32)
            ends = np.asarray(data["slot_end"] if "slot_end" in data.files else (data["end"] if "end" in data.files else starts + 4.0), dtype=np.float32)
            labels = data["slot_label"].tolist() if "slot_label" in data.files else [label_names[int(np.argmax(r))] for r in probs_arr]
            raw_slots = []
            for i in range(len(probs_arr)):
                probs = {label_names[j]: float(probs_arr[i, j]) for j in range(min(len(label_names), probs_arr.shape[1]))}
                raw_slots.append({"slot_id": i, "start": float(starts[i]), "end": float(ends[i]), "top_label": labels[i], "probs": probs})
        else:
            obj = load_json(p)
            raw_slots = obj.get("slots", obj.get("segments", obj if isinstance(obj, list) else []))
            if not isinstance(raw_slots, list):
                return None
        slots: List[dict] = []
        feats: List[np.ndarray] = []
        for i, s in enumerate(raw_slots):
            if not isinstance(s, dict):
                continue
            start = float(s.get("start_sec", s.get("start", s.get("t0", i * 4.0))))
            end = float(s.get("end_sec", s.get("end", s.get("t1", start + float(getattr(cfg, "unpaired_audio_slot_seconds", 4.0))))))
            duration = max(end - start, 1e-3)
            top = s.get("top_label", s.get("label", s.get("music_alignment_label", None)))
            probs = normalize_music_probs(s.get("probs", s.get("probabilities", s.get("slot_probs", None))), top, getattr(cfg, "external_music_semantic_temperature", 0.65))
            pseudo = music_probs_to_pseudo_feature(probs, duration, cfg)
            sem = music_semantic_slot_from_probs(probs, duration, source=str(p))
            item = {"slot_id": int(s.get("slot_id", i)), "start": start, "end": end, "duration": duration,
                    "energy": float(pseudo[2]), "onset": float(pseudo[16])}
            item.update(sem)
            slots.append(item)
            feats.append(pseudo)
        if not feats:
            return None
        return slots, np.stack(feats).astype(np.float32)
    except Exception as exc:
        print(f"[V46.12 WARN] failed parsing external music semantic {p}: {exc}", file=sys.stderr)
        return None


def filename_proxy_music_semantic(audio_path: str | Path, cfg: V46Config, slot_seconds: float) -> Optional[Tuple[List[dict], np.ndarray]]:
    if not bool(getattr(cfg, "external_music_semantic_filename_proxy", True)):
        return None
    stem = Path(audio_path).stem.lower()
    label = None
    for key, lab in MUSIC_PROXY_NAME_LABELS.items():
        if key in stem:
            label = lab
            break
    if label is None:
        return None
    try:
        sr, wav = read_wav_mono(audio_path)
        total = len(wav) / max(float(sr), 1.0)
    except Exception:
        total = float(slot_seconds)
    n = max(1, int(math.ceil(total / max(float(slot_seconds), 1e-3))))
    probs = normalize_music_probs(None, label, getattr(cfg, "external_music_semantic_temperature", 0.65))
    slots, feats = [], []
    for i in range(n):
        start = i * float(slot_seconds)
        end = min(total, (i + 1) * float(slot_seconds)) if total > 0 else (i + 1) * float(slot_seconds)
        dur = max(end - start, 1e-3)
        pseudo = music_probs_to_pseudo_feature(probs, dur, cfg)
        sem = music_semantic_slot_from_probs(probs, dur, source=f"filename_proxy:{Path(audio_path).name}")
        item = {"slot_id": i, "start": start, "end": end, "duration": dur, "energy": float(pseudo[2]), "onset": float(pseudo[16])}
        item.update(sem)
        slots.append(item)
        feats.append(pseudo)
    return slots, np.stack(feats).astype(np.float32)


def load_external_music_semantic_slots(audio_path: str | Path, cfg: V46Config, slot_seconds: float) -> Optional[Tuple[List[dict], np.ndarray]]:
    if not bool(getattr(cfg, "external_music_semantic_enable", True)):
        return None
    for cand in sidecar_music_semantic_candidates(audio_path, cfg):
        if cand.exists() and cand.is_file():
            parsed = parse_external_music_semantic_file(cand, cfg)
            if parsed is not None:
                return parsed
    cmd_out = run_external_music_semantic_cmd(audio_path, cfg)
    if cmd_out is not None:
        parsed = parse_external_music_semantic_file(cmd_out, cfg)
        if parsed is not None:
            return parsed
    if bool(getattr(cfg, "external_music_semantic_proxy_enable", True)):
        prox = filename_proxy_music_semantic(audio_path, cfg, slot_seconds)
        if prox is not None:
            return prox
    if bool(getattr(cfg, "external_music_semantic_required", False)):
        raise RuntimeError(f"External music semantic is required but no JSON/NPZ/command output was found for {audio_path}")
    return None

def semantic_label_match_bonus(slot: dict, db: dict, cfg: V46Config) -> np.ndarray:
    """V46.31 interpretable music-router bonus for Chang-E semantic Event-RAG."""
    n = len(db.get("paths", []))
    bonus = np.zeros(n, dtype=np.float32)
    if not bool(getattr(cfg, "classification_semantic_enable", True)) or n == 0:
        return bonus
    dance_keys = np.asarray(db.get("dance_keys", np.array(["unknown"] * n, dtype=object)), dtype=object)
    roles = np.asarray(db.get("semantic_roles", np.array(["unknown"] * n, dtype=object)), dtype=object)
    energy = np.asarray(db.get("energy_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    rhythm = np.asarray(db.get("rhythm_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    align = np.asarray(db.get("music_alignment_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    families = np.asarray(db.get("event_families", np.array(["unknown"] * n, dtype=object)), dtype=object)
    stages = np.asarray(db.get("motion_stage_roles", np.array(["unknown"] * n, dtype=object)), dtype=object)
    motifs = np.asarray(db.get("cultural_motifs", np.array(["unknown"] * n, dtype=object)), dtype=object)
    locomotion = np.asarray(db.get("locomotion_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    support = np.asarray(db.get("support_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    quality = np.asarray(db.get("event_quality_scores", np.ones(n, dtype=np.float32)), dtype=np.float32)
    preferred = [canonicalize_chang_e_key(x) for x in slot.get("preferred_dance_keys", [])]
    preferred_roles = [str(x) for x in slot.get("preferred_semantic_roles", [])]
    slot_align = str(slot.get("music_alignment_label", slot.get("music_semantic_top_label", "")))
    slot_energy = str(slot.get("energy_label", "")); slot_rhythm = str(slot.get("rhythm_label", "")); slot_role = str(slot.get("role", "normal"))
    route_family_map = {"calm_meditative": ["calm_flow", "pose_motif", "aerial_curve"], "pose_hold": ["pose_motif", "calm_flow", "instrument_motif"], "lyrical_flow": ["aerial_curve", "footwork_flow", "instrument_motif", "calm_flow"], "footwork_flow": ["footwork_flow", "turning_flow", "aerial_curve"], "instrument_phrase": ["instrument_motif", "aerial_curve", "pose_motif"], "percussive_accent": ["percussive_accent", "turning_flow", "instrument_motif"], "turning_climax": ["turning_flow", "aerial_curve", "percussive_accent"], "aerial_curve": ["aerial_curve", "turning_flow", "footwork_flow"]}
    route_stage_map = {"intro": ["intro", "intro_or_resolution", "anchor_or_resolution"], "calm": ["intro", "intro_or_resolution", "resolution", "anchor_or_resolution"], "normal": ["development", "build_up", "motif_recall"], "development": ["development", "build_up"], "build_up": ["build_up", "development", "opening_or_climax"], "motif": ["motif_recall", "development"], "motif_recall": ["motif_recall", "anchor_or_resolution"], "accent": ["accent_or_climax", "climax", "build_up"], "climax": ["climax", "accent_or_climax", "opening_or_climax"], "release": ["resolution", "anchor_or_resolution", "intro_or_resolution"], "resolution": ["resolution", "anchor_or_resolution", "intro_or_resolution"]}
    route_support_map = {"calm_meditative": ["stable_support", "static_or_low_motion_support"], "pose_hold": ["stable_support", "static_or_low_motion_support"], "footwork_flow": ["alternating_foot_support", "alternating_or_pivot_support"], "turning_climax": ["alternating_or_pivot_support", "low_contact_flight_like"], "percussive_accent": ["strong_foot_contact", "alternating_foot_support"], "lyrical_flow": ["alternating_foot_support", "low_contact_flight_like", "stable_support"], "instrument_phrase": ["stable_support", "alternating_foot_support"]}
    route_loco_map = {"calm_meditative": ["slow_weight_shift", "in_place_pose", "floating_leaning"], "pose_hold": ["in_place_pose", "slow_weight_shift"], "footwork_flow": ["traveling_steps", "turning_travel"], "turning_climax": ["turning_travel", "floating_leaning"], "percussive_accent": ["accented_travel", "turning_travel", "traveling_steps"], "lyrical_flow": ["floating_leaning", "traveling_steps", "upper_body_phrase"], "instrument_phrase": ["upper_body_phrase", "in_place_pose"]}
    for k in preferred:
        if k:
            bonus += (dance_keys == k).astype(np.float32) * float(getattr(cfg, "preferred_dance_key_bonus", 0.28))
    for r in preferred_roles:
        if r:
            bonus += (roles == r).astype(np.float32) * 0.25
    if slot_align:
        bonus += (align == slot_align).astype(np.float32) * 0.45
        for rank, fam in enumerate(route_family_map.get(slot_align, [])):
            bonus += (families == fam).astype(np.float32) * float(getattr(cfg, "event_family_bonus", 0.58)) / float(rank + 1)
        for rank, sup in enumerate(route_support_map.get(slot_align, [])):
            bonus += (support == sup).astype(np.float32) * float(getattr(cfg, "route_support_bonus", 0.12)) / float(rank + 1)
        for rank, loc in enumerate(route_loco_map.get(slot_align, [])):
            bonus += (locomotion == loc).astype(np.float32) * float(getattr(cfg, "route_locomotion_bonus", 0.14)) / float(rank + 1)
    if slot_role:
        for rank, st in enumerate(route_stage_map.get(slot_role, [])):
            bonus += (stages == st).astype(np.float32) * float(getattr(cfg, "motion_stage_role_bonus", 0.36)) / float(rank + 1)
    if slot_energy:
        bonus += (energy == slot_energy).astype(np.float32) * 0.14
    if slot_rhythm:
        bonus += (rhythm == slot_rhythm).astype(np.float32) * 0.12
    if slot_align == "instrument_phrase":
        bonus += np.isin(motifs, ["pipa_instrument_pose", "thunder_drum"]).astype(np.float32) * 0.16
    conf = np.asarray(db.get("semantic_confidence", np.ones(n, dtype=np.float32)), dtype=np.float32)
    q_gate = np.clip(0.45 + 0.55 * quality, 0.25, 1.15)
    bonus *= np.clip(0.65 + 0.35 * conf, 0.5, 1.15) * q_gate
    # V46.31: never normalize by the current candidate max.  That dynamic
    # min-max scaling can turn a weak accidental match in a vague slot into a
    # full-strength routing reward.  Use a fixed saturating scale instead.
    scale = max(0.25, float(getattr(cfg, "route_semantic_bonus_scale", 1.50)))
    bonus = 1.0 - np.exp(-np.maximum(bonus, 0.0) / scale)
    return np.clip(bonus, 0.0, 1.0).astype(np.float32)


def parse_change_bvh_semantics(path: str | Path) -> Dict[str, object]:
    """Parse meaningful Chang-E filename semantics from EDGE/change/*.bvh.

    Examples:
      female_36pose_1.bvh -> gender=female, category=thirty_six_postures, take=1
      male_pipa_2.bvh     -> gender=male,   category=pipa_behind_back, take=2
      female_lotus.bvh    -> gender=female, category=lotus_steps

    The full filename stem is deliberately used as source_uid/source_group for
    source-aware RAG and leakage prevention.  Category/gender/take are separate
    semantic attributes used for routing and reporting, not for source grouping.
    """
    stem = _clean_stem(path)
    tokens = [t for t in stem.split("_") if t]
    gender = "unknown"
    rest = tokens[:]
    if rest and rest[0] in {"male", "female"}:
        gender = rest[0]
        rest = rest[1:]

    take_id: Optional[int] = None
    if rest and rest[-1].isdigit():
        take_id = int(rest[-1])
        rest = rest[:-1]
    base = "_".join(rest) if rest else stem

    category_key = "unknown"
    for key, prof in CHANG_E_CATEGORY_PROFILES.items():
        aliases = set(prof.get("aliases", set()))
        if base in aliases or any(tok in aliases for tok in rest):
            category_key = key
            break
    if category_key == "unknown":
        category_key = canonicalize_chang_e_key(base or "unknown")
    else:
        category_key = canonicalize_chang_e_key(category_key)

    prof = CHANG_E_CATEGORY_PROFILES.get(category_key, {})
    display = str(prof.get("display", category_key.replace("_", " ").title()))
    semantic_role = str(prof.get("semantic_role", "unknown_motion"))
    source_uid = stem
    take_text = f" take {take_id}" if take_id is not None else ""
    gender_text = "female" if gender == "female" else ("male" if gender == "male" else "unknown-gender")
    semantic_text = f"{gender_text} {display}{take_text}; role={semantic_role}"
    event_label = category_key
    if take_id is not None:
        event_label = f"{category_key}_take{take_id}"
    return {
        "source_uid": source_uid,
        "source_group": source_uid,
        "gender": gender,
        "dance_key": category_key,
        "dance_category": display,
        "semantic_role": semantic_role,
        "semantic_text": semantic_text,
        "take_id": take_id if take_id is not None else -1,
        "source_take": take_id if take_id is not None else -1,
        "label": event_label,
        "parent_label": category_key,
        "raw_stem": stem,
    }


def source_group_from_path(path: str | Path) -> str:
    # V46.9 default: each meaningful Chang-E BVH stem is a separate source
    # group. This yields 12 source groups for the user's current change/*.bvh
    # instead of collapsing singleton files into one directory-level group.
    return str(parse_change_bvh_semantics(path).get("source_group"))


def infer_label_from_filename(path: str | Path) -> str:
    return str(parse_change_bvh_semantics(path).get("label"))


def filename_semantic_vector_from_meta(meta: dict, cfg: Optional[V46Config] = None) -> np.ndarray:
    """Convert filename/category semantics into a 32D slot-compatible prior.

    The vector uses the same coarse channel layout as audio_slots()/event_descriptor():
    duration, root/travel, energy, dynamics, joint/lower/upper density, contact
    calmness, foot speed, turn/onset and root-y dynamics.  It is not a paired
    label; it is a weak prior that makes unpaired music-to-event matching aware
    that e.g. drum should prefer onset-rich slots and meditation should prefer
    calm slots.
    """
    key = canonicalize_chang_e_key(meta.get("dance_key") or meta.get("parent_label") or meta.get("label") or "unknown")
    prof = CHANG_E_CATEGORY_PROFILES.get(key, {})
    duration = float(meta.get("duration", 0.0) or 0.0)
    energy = float(prof.get("energy", 0.40))
    onset = float(prof.get("onset", 0.20))
    lower = float(prof.get("lower", energy))
    upper = float(prof.get("upper", energy))
    turn = float(prof.get("turn", 0.15))
    travel = float(prof.get("travel", 0.25))
    calm = float(prof.get("calmness", max(0.0, 0.75 - energy)))
    v = np.zeros(32, dtype=np.float32)
    v[0] = duration
    v[1] = travel * 2.0
    v[2] = energy
    v[3] = min(1.0, energy + 0.25 * onset)
    v[4] = energy * 0.45 + onset * 0.55
    v[5] = energy + onset
    v[6] = min(1.0, energy + 0.45 * onset)
    v[7] = lower
    v[8] = upper
    v[9] = lower / max(upper, 1e-4)
    v[10] = calm
    v[11] = calm * 0.8
    v[12] = calm * 0.8
    v[13] = max(0.02, onset)
    v[14] = max(0.02, onset * 1.25)
    v[15] = turn
    v[16] = abs(turn) + 0.25 * onset
    v[17] = max(abs(turn), onset)
    v[18] = energy * (0.25 + 0.5 * (1.0 - calm))
    v[19] = 0.15 + 0.25 * energy
    # Gender is not a quality score. It is only kept as a small style flag so
    # the router can report source diversity; it should not dominate retrieval.
    gender = str(meta.get("gender", "unknown"))
    v[20] = 0.15 if gender == "female" else (0.35 if gender == "male" else 0.25)
    take = float(meta.get("take_id", -1) if meta.get("take_id", -1) is not None else -1)
    v[21] = max(0.0, take) / 4.0 if take >= 0 else 0.0
    # Duplicate coarse stats to high channels for robust normalized matching.
    v[22] = energy
    v[23] = onset
    v[24] = travel
    v[25] = calm
    v[26] = lower
    v[27] = upper
    v[28] = turn
    v[29] = energy + 0.5 * onset
    v[30] = (lower + upper) * 0.5
    v[31] = 1.0
    return v.astype(np.float32)


def motion_feature_z_for_alignment(db: dict, cfg: V46Config, weight: Optional[float] = None) -> np.ndarray:
    """Return descriptor z mixed with filename and strong classification priors."""
    desc_z = np.asarray(db["desc_z"], dtype=np.float32)
    if not bool(getattr(cfg, "filename_semantic_enable", True)):
        return desc_z
    mean = np.asarray(db["desc_mean"], dtype=np.float32)
    std = np.asarray(db["desc_std"], dtype=np.float32)
    parts = []
    if "name_semantic" in db:
        name = np.asarray(db["name_semantic"], dtype=np.float32)
        parts.append((name - mean) / np.maximum(std, 1e-6))
    if bool(getattr(cfg, "classification_semantic_enable", True)) and "class_semantic" in db:
        cls = np.asarray(db["class_semantic"], dtype=np.float32)
        cls_z = (cls - mean) / np.maximum(std, 1e-6)
        if parts:
            ratio = float(getattr(cfg, "classification_semantic_ratio", 0.70))
            ratio = max(0.0, min(1.0, ratio))
            sem_z = (1.0 - ratio) * parts[0] + ratio * cls_z
        else:
            sem_z = cls_z
    elif parts:
        sem_z = parts[0]
    else:
        return desc_z
    sem_z = np.clip(sem_z, -8.0, 8.0).astype(np.float32)
    w = float(getattr(cfg, "filename_semantic_weight", 0.35) if weight is None else weight)
    w = max(0.0, min(0.85, w))
    return ((1.0 - w) * desc_z + w * sem_z).astype(np.float32)


def scan_motion_files(motion_dirs: Sequence[str], exts: Sequence[str] = (".npy", ".npz", ".pkl", ".pickle", ".bvh")) -> List[str]:
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



def resolve_path_from_roots(value: object, roots: Sequence[str | Path]) -> Optional[str]:
    """Resolve manifest paths robustly against manifest dir, repo root, and motion dirs."""
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    p = Path(text)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        for r in roots:
            if not r:
                continue
            candidates.append(Path(r) / p)
            candidates.append(Path(r) / p.name)
    # Direct hit.
    for c in candidates:
        if c.exists():
            return str(c)
    # Fallback: recursive name search inside roots for screenshot-style change/*.bvh.
    name = p.name
    if name:
        for r in roots:
            rr = Path(r)
            if rr.exists() and rr.is_dir():
                hits = list(rr.rglob(name))
                if hits:
                    return str(hits[0])
    return None


def _coerce_int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return int(float(text))
    except Exception:
        return default


def _coerce_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return float(text)
    except Exception:
        return default


def resample_motion_to_config_fps(motion: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:
    """Resample BVH-derived EDGE-like arrays to cfg.fps before event slicing.

    V46.15 fix:
    channel 0 may temporarily carry BVH native fps from load_bvh_file(), but it
    must never be overwritten with target fps after resampling because EDGE
    channel 0 is a contact channel.  The DB writer rebuilds contacts from FK.
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        return x.astype(np.float32), {"resampled": False, "reason": "too_short"}

    ch0 = x[:, 0]
    finite_ch0 = ch0[np.isfinite(ch0)]
    if finite_ch0.size:
        ch0_med = float(np.nanmedian(finite_ch0))
        ch0_p05 = float(np.nanpercentile(finite_ch0, 5))
        ch0_p95 = float(np.nanpercentile(finite_ch0, 95))
        # V46.31: FPS metadata written by the BVH loader is nearly constant, but
        # a few boundary/cropping frames may contain small jitter.  Use the middle
        # 90% band for the main constant-column test and keep raw std only for
        # diagnostics.  This avoids silently skipping required high-FPS -> 30 FPS
        # resampling because of a few outlier rows.
        trimmed = finite_ch0[(finite_ch0 >= ch0_p05) & (finite_ch0 <= ch0_p95)]
        ch0_std = float(np.nanstd(finite_ch0))
        ch0_trimmed_std = float(np.nanstd(trimmed)) if trimmed.size else ch0_std
    else:
        ch0_med = ch0_p05 = ch0_p95 = ch0_std = ch0_trimmed_std = 0.0
    looks_like_fps_metadata = bool(
        ch0_med > 2.0 and ch0_p05 > 1.0 and ch0_p95 < 400.0
        and ch0_trimmed_std < max(0.75, ch0_med * 0.08)
    )
    native = ch0_med if looks_like_fps_metadata else float(cfg.fps)
    target = float(cfg.fps)

    if (not bool(getattr(cfg, "bvh_resample_to_config_fps", True))) or abs(native - target) < 1e-3:
        y = x.copy().astype(np.float32)
        return y, {
            "resampled": False,
            "native_fps": float(native),
            "target_fps": float(target),
            "frames": int(len(x)),
            "fps_metadata_detected": looks_like_fps_metadata,
            "channel0_trimmed_std": float(ch0_trimmed_std),
            "note": "channel0_not_overwritten_EDGE_contact_contract",
        }

    new_len = max(2, int(round(x.shape[0] * target / max(native, 1e-6))))
    y = resample_motion_np(x, new_len).astype(np.float32)
    return y, {
        "resampled": True,
        "native_fps": float(native),
        "target_fps": float(target),
        "frames_before": int(len(x)),
        "frames_after": int(new_len),
        "fps_metadata_detected": looks_like_fps_metadata,
        "channel0_trimmed_std": float(ch0_trimmed_std),
        "note": "channel0_preserved_until_event_contract_guard_rebuilds_contacts",
    }


def read_manifest_records(manifest_path: Optional[str], motion_dirs: Sequence[str], cfg: V46Config) -> List[dict]:
    """
    Read an optional manifest.csv and return normalized source-aware records.

    Supported columns follow the user's prior manifest document:
      source_bvh, fragment_file, fragment_index, label,
      start_frame, end_frame, start_time, end_time, duration_sec, bvh_fps.
    The manifest is an upstream semantic slicing index, not a finished RAG DB;
    we keep its fields as parent metadata and still compute tensor/contact/boundary
    descriptors here.
    """
    if not manifest_path or not bool(getattr(cfg, "manifest_enable", True)):
        return []
    mp = Path(manifest_path)
    if not mp.exists():
        return []
    roots: List[Path] = [mp.parent, Path.cwd()]
    for d in motion_dirs or []:
        roots.append(Path(d))
    rows: List[dict] = []
    with open(mp, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for ridx, row in enumerate(reader):
            fragment = resolve_path_from_roots(row.get("fragment_file") or row.get("path") or row.get("file"), roots)
            source = resolve_path_from_roots(row.get("source_bvh") or row.get("source_file") or row.get("bvh"), roots)
            load_path = fragment or source
            if not load_path:
                continue
            start_frame = _coerce_int(row.get("start_frame"), None)
            end_frame = _coerce_int(row.get("end_frame"), None)
            label = str(row.get("label") or row.get("motion_event") or row.get("fragment_name") or Path(load_path).stem)
            source_name = str(row.get("source_bvh") or Path(source or load_path).name)
            sem = parse_change_bvh_semantics(source or load_path)
            item = {
                "manifest_id": ridx,
                "manifest_path": str(mp),
                "load_path": str(load_path),
                "source_file": str(source or load_path),
                "source_bvh": source_name,
                "fragment_file": str(fragment) if fragment else None,
                "fragment_index": _coerce_int(row.get("fragment_index"), ridx),
                "label": label,
                "parent_label": label,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec_manifest": _coerce_float(row.get("duration_sec"), None),
                "bvh_fps_manifest": _coerce_float(row.get("bvh_fps"), None),
                "raw_manifest_row": {k: row.get(k) for k in row.keys()},
            }
            item.update({k: v for k, v in sem.items() if k not in {"label", "parent_label"}})
            # Manifest labels are manual semantic labels and should remain primary.
            rows.append(item)
    return rows


def iter_source_records(args: argparse.Namespace, cfg: V46Config) -> Tuple[List[dict], dict]:
    """Return load records from manifest when available, otherwise from motion_dirs."""
    motion_dirs = list(getattr(args, "motion_dirs", None) or [])
    manifest_records = read_manifest_records(getattr(args, "manifest", None), motion_dirs, cfg)
    if manifest_records:
        return manifest_records, {"input_mode": "manifest", "manifest_records": len(manifest_records), "motion_dirs": motion_dirs}
    files = scan_motion_files(motion_dirs)
    if not files:
        raise FileNotFoundError(f"No motion files found in: {motion_dirs}")
    out_records = []
    for f in files:
        sem = parse_change_bvh_semantics(f)
        rec = {"load_path": f, "source_file": f, "source_bvh": Path(f).name, "fragment_index": 0}
        rec.update(sem)
        out_records.append(rec)
    return out_records, {"input_mode": "direct_files", "files": len(files), "motion_dirs": motion_dirs}


def add_event_to_db_lists(
    clip: np.ndarray,
    event_idx: int,
    out_path: Path,
    cfg: V46Config,
    source: str,
    matched_audio: Optional[str],
    st: int,
    base_meta: dict,
    descs: List[np.ndarray], entries: List[np.ndarray], exits: List[np.ndarray], c0s: List[np.ndarray], c1s: List[np.ndarray],
    music_feats: List[np.ndarray], music_masks: List[float], meta: List[dict],
) -> None:
    clip, contract_report = enforce_edge151_contract_np(
        clip,
        cfg,
        source_hint=str(base_meta.get("source_file", out_path)),
        derive_contact=True,
        project_rot=True,
    )
    np.save(out_path, clip.astype(np.float32))
    desc = event_descriptor(clip, cfg.fps)
    entry, exit_, c0, c1 = motion_boundary_state(clip)
    if matched_audio:
        try:
            music_feat = audio_feature_for_motion_clip(matched_audio, st, clip.shape[0], cfg)
            music_mask = 1.0
        except Exception as exc:
            print(f"[V46.11 WARN] audio feature failed for {matched_audio}: {exc}", file=sys.stderr)
            music_feat = np.zeros(32, dtype=np.float32)
            music_mask = 0.0
    else:
        music_feat = np.zeros(32, dtype=np.float32)
        music_mask = 0.0
    descs.append(desc)
    entries.append(entry)
    exits.append(exit_)
    c0s.append(c0)
    c1s.append(c1)
    music_feats.append(music_feat.astype(np.float32))
    music_masks.append(float(music_mask))
    item = {
        "event_id": event_idx,
        "path": str(out_path),
        "source_file": str(base_meta.get("source_file", base_meta.get("load_path", ""))),
        "source_bvh": str(base_meta.get("source_bvh", Path(str(base_meta.get("source_file", "source"))).name)),
        "source_group": source,
        "matched_audio": matched_audio,
        "has_real_audio_feature": bool(music_mask > 0.5),
        "seq_id": int(base_meta.get("seq_id", 0)),
        "start": int(st),
        "end": int(st + clip.shape[0]),
        "frames": int(clip.shape[0]),
        "duration": float(clip.shape[0] / max(float(cfg.fps), 1e-6)),
        "label": str(base_meta.get("label", infer_label_from_filename(base_meta.get("source_file", out_path)))),
        "parent_label": str(base_meta.get("parent_label", base_meta.get("label", "unknown"))),
        "fragment_index": int(base_meta.get("fragment_index", 0) or 0),
        "manifest_id": base_meta.get("manifest_id"),
        "manifest_path": base_meta.get("manifest_path"),
        "input_mode": base_meta.get("input_mode", "direct_files"),
        "edge151_contract_report": contract_report,
    }
    sem = parse_change_bvh_semantics(base_meta.get("source_file", base_meta.get("source_bvh", out_path)))
    item.update(strong_action_semantics_from_meta({**sem, **item}, desc))
    # Keep source_uid/source_group from filename unless a manifest-specific source
    # explicitly supplied them.  Keep manifest label if present; otherwise use
    # filename category/take label.
    for k in ["source_uid", "gender", "dance_key", "dance_category", "semantic_role", "semantic_text", "take_id", "source_take", "raw_stem"]:
        item[k] = base_meta.get(k, sem.get(k))
    strong_sem = strong_action_semantics_from_meta(item, desc)
    item.update(strong_sem)
    if item.get("semantic_text"):
        item["semantic_text"] = str(item["semantic_text"]) + "; " + str(strong_sem.get("classification_text", ""))
    if not item.get("label") or item.get("label") == "unknown":
        item["label"] = str(sem.get("label", "unknown"))
    if not item.get("parent_label") or item.get("parent_label") == "unknown":
        item["parent_label"] = str(sem.get("parent_label", item.get("label", "unknown")))
    if "resample_report" in base_meta:
        item["resample_report"] = base_meta["resample_report"]
    meta.append(item)

def build_db(args: argparse.Namespace) -> int:
    cfg = V46Config.from_json(args.config).apply_env()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = ensure_dir(args.out_db)
    audio_files = collect_audio_files(getattr(args, "audio_dirs", None))
    records, input_report = iter_source_records(args, cfg)
    audio_match_cache: Dict[str, Optional[str]] = {}

    meta: List[dict] = []
    descs: List[np.ndarray] = []
    entries: List[np.ndarray] = []
    exits: List[np.ndarray] = []
    c0s: List[np.ndarray] = []
    c1s: List[np.ndarray] = []
    music_feats: List[np.ndarray] = []
    music_masks: List[float] = []
    resample_reports: List[dict] = []

    npy_dir = ensure_dir(out_dir / "events")
    event_idx = 0
    for rec in records:
        f = str(rec["load_path"])
        seqs = load_motion_file(f)
        if not seqs:
            continue
        # Manifest rows can reference a full source_bvh plus original high-FPS frame range.
        start_frame = rec.get("start_frame")
        end_frame = rec.get("end_frame")
        source_key_path = rec.get("source_file") or f
        source = source_group_from_path(source_key_path)
        if source_key_path not in audio_match_cache:
            audio_match_cache[str(source_key_path)] = find_matching_audio_for_motion(str(source_key_path), audio_files)
        matched_audio = audio_match_cache.get(str(source_key_path))

        for seq_id, seq_raw in enumerate(seqs):
            seq = np.asarray(seq_raw, dtype=np.float32)
            if start_frame is not None or end_frame is not None:
                a = max(0, int(start_frame or 0))
                b = min(seq.shape[0], int(end_frame if end_frame is not None else seq.shape[0]))
                if b > a:
                    seq = seq[a:b]
            seq, res_report = resample_motion_to_config_fps(seq, cfg)
            resample_reports.append(res_report)
            T = seq.shape[0]
            if T < cfg.min_event_frames:
                continue
            # Manifest rows are motif-level semantic clips.  If enabled, split long
            # motif clips into event-level windows while preserving parent metadata;
            # otherwise preserve each motif as one event.
            if rec.get("manifest_id") is not None and not bool(getattr(cfg, "manifest_secondary_event_split", True)):
                starts = [0]
                win = min(T, cfg.max_event_frames)
            else:
                win = min(int(cfg.window_len), T)
                if bool(getattr(cfg, "chang_e_boundary_event_split", True)):
                    starts = chang_e_semantic_event_starts(seq, cfg)
                else:
                    starts = [0] if T <= cfg.max_event_frames else list(range(0, max(1, T - cfg.min_event_frames + 1), cfg.hop_len))
            for st in starts:
                endf = min(T, st + win)
                if endf - st < cfg.min_event_frames:
                    continue
                clip = seq[st:endf].astype(np.float32)
                if clip.shape[0] > cfg.max_event_frames:
                    clip = clip[: cfg.max_event_frames]
                path = npy_dir / f"event_{event_idx:07d}.npy"
                base_meta = dict(rec)
                event_mid = (float(st) + 0.5 * float(endf - st)) / max(float(T), 1.0)
                base_meta.update({
                    "seq_id": seq_id,
                    "resample_report": res_report,
                    "input_mode": input_report.get("input_mode"),
                    "event_start": int(st),
                    "event_end": int(endf),
                    "event_source_frames": int(T),
                    "event_position_mid": float(event_mid),
                    "event_position_fraction": float(event_mid),
                })
                add_event_to_db_lists(
                    clip=clip,
                    event_idx=event_idx,
                    out_path=path,
                    cfg=cfg,
                    source=source,
                    matched_audio=matched_audio,
                    st=st,
                    base_meta=base_meta,
                    descs=descs,
                    entries=entries,
                    exits=exits,
                    c0s=c0s,
                    c1s=c1s,
                    music_feats=music_feats,
                    music_masks=music_masks,
                    meta=meta,
                )
                event_idx += 1

    if not meta:
        raise RuntimeError("No valid EDGE 151D motion events were built.")
    desc = np.stack(descs).astype(np.float32)
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1e-6
    desc_z = (desc - mean) / std
    name_semantic = np.stack([filename_semantic_vector_from_meta(m, cfg) for m in meta]).astype(np.float32)
    class_semantic = np.stack([class_semantic_vector_from_meta(m, cfg) for m in meta]).astype(np.float32)
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
        labels=np.array([m.get("label", "unknown") for m in meta], dtype=object),
        parent_labels=np.array([m.get("parent_label", m.get("label", "unknown")) for m in meta], dtype=object),
        source_bvh=np.array([m.get("source_bvh", "") for m in meta], dtype=object),
        source_uids=np.array([m.get("source_uid", m.get("source_group", "")) for m in meta], dtype=object),
        genders=np.array([m.get("gender", "unknown") for m in meta], dtype=object),
        dance_keys=np.array([m.get("dance_key", "unknown") for m in meta], dtype=object),
        dance_categories=np.array([m.get("dance_category", "unknown") for m in meta], dtype=object),
        semantic_roles=np.array([m.get("semantic_role", "unknown") for m in meta], dtype=object),
        semantic_texts=np.array([m.get("semantic_text", "") for m in meta], dtype=object),
        energy_labels=np.array([m.get("energy_label", "unknown") for m in meta], dtype=object),
        rhythm_labels=np.array([m.get("rhythm_label", "unknown") for m in meta], dtype=object),
        body_focus_labels=np.array([m.get("body_focus_label", "unknown") for m in meta], dtype=object),
        spatial_labels=np.array([m.get("spatial_label", "unknown") for m in meta], dtype=object),
        music_alignment_labels=np.array([m.get("music_alignment_label", "unknown") for m in meta], dtype=object),
        classification_texts=np.array([m.get("classification_text", "") for m in meta], dtype=object),
        event_families=np.array([m.get("event_family", "unknown") for m in meta], dtype=object),
        motion_stage_roles=np.array([m.get("motion_stage_role", "unknown") for m in meta], dtype=object),
        cultural_motifs=np.array([m.get("cultural_motif", "unknown") for m in meta], dtype=object),
        prop_proxy_labels=np.array([m.get("prop_proxy_label", "unknown") for m in meta], dtype=object),
        locomotion_labels=np.array([m.get("locomotion_label", "unknown") for m in meta], dtype=object),
        support_labels=np.array([m.get("support_label", "unknown") for m in meta], dtype=object),
        event_position_mid=np.array([float(m.get("event_position_mid", 0.5)) for m in meta], dtype=np.float32),
        semantic_confidence=np.array([float(m.get("semantic_confidence", 0.5)) for m in meta], dtype=np.float32),
        event_quality_scores=np.array([float(m.get("event_quality_score", 0.5)) for m in meta], dtype=np.float32),
        natural_duration_min=np.array([float((m.get("natural_duration_range_sec", [1.5, 4.0]) or [1.5, 4.0])[0]) for m in meta], dtype=np.float32),
        natural_duration_max=np.array([float((m.get("natural_duration_range_sec", [1.5, 4.0]) or [1.5, 4.0])[-1]) for m in meta], dtype=np.float32),
        take_ids=np.array([int(m.get("take_id", -1) if m.get("take_id", -1) is not None else -1) for m in meta], dtype=np.int32),
        name_semantic=name_semantic.astype(np.float32),
        class_semantic=class_semantic.astype(np.float32),
        durations=np.array([m["duration"] for m in meta], dtype=np.float32),
        frames=np.array([m["frames"] for m in meta], dtype=np.int32),
        music=np.stack(music_feats).astype(np.float32),
        music_mask=np.array(music_masks, dtype=np.float32),
    )
    audio_coverage = float(np.mean(music_masks)) if music_masks else 0.0
    split_audit = {
        "num_sources_total": int(len(set(str(m.get("source_group")) for m in meta))),
        "num_labels_total": int(len(set(str(m.get("label")) for m in meta))),
        "events_per_source_min": int(min([sum(str(x.get("source_group")) == s for x in meta) for s in set(str(m.get("source_group")) for m in meta)])),
        "events_per_source_max": int(max([sum(str(x.get("source_group")) == s for x in meta) for s in set(str(m.get("source_group")) for m in meta)])),
        "num_source_uids_total": int(len(set(str(m.get("source_uid", m.get("source_group"))) for m in meta))),
        "category_counts": {str(k): int(sum(str(m.get("dance_key")) == str(k) for m in meta)) for k in sorted(set(str(m.get("dance_key")) for m in meta))},
        "gender_counts": {str(k): int(sum(str(m.get("gender")) == str(k) for m in meta)) for k in sorted(set(str(m.get("gender")) for m in meta))},
        "energy_label_counts": {str(k): int(sum(str(m.get("energy_label")) == str(k) for m in meta)) for k in sorted(set(str(m.get("energy_label")) for m in meta))},
        "rhythm_label_counts": {str(k): int(sum(str(m.get("rhythm_label")) == str(k) for m in meta)) for k in sorted(set(str(m.get("rhythm_label")) for m in meta))},
        "body_focus_counts": {str(k): int(sum(str(m.get("body_focus_label")) == str(k) for m in meta)) for k in sorted(set(str(m.get("body_focus_label")) for m in meta))},
        "music_alignment_label_counts": {str(k): int(sum(str(m.get("music_alignment_label")) == str(k) for m in meta)) for k in sorted(set(str(m.get("music_alignment_label")) for m in meta))},
        "source_group_semantics": "full_filename_stem; category/gender/take are separate semantic metadata; V46.11 adds multi-label action/music-alignment classes",
        "train_val_group_overlap": 0,  # no random sample split is produced here; downstream split must remain source-disjoint.
    }
    report = {
        "version": "v46_11_canonical_strong_class_semantic_source_aware_db",
        "config": dataclasses.asdict(cfg),
        "events": meta,
        "num_events": len(meta),
        "audio_feature_coverage": audio_coverage,
        "audio_files_seen": len(audio_files),
        "input_report": input_report,
        "resample_summary": {
            "resampled_count": int(sum(1 for r in resample_reports if r.get("resampled"))),
            "native_fps_values": sorted(set(round(float(r.get("native_fps", cfg.fps)), 6) for r in resample_reports if "native_fps" in r))[:12],
            "target_fps": float(cfg.fps),
        },
        "split_audit": split_audit,
        "fk_tree_source": FK_TREE_SOURCE,
    }
    save_json(report, out_dir / "events_meta.json")
    print(json.dumps({"db": str(db_path), "num_events": len(meta), "out_dir": str(out_dir), "audio_feature_coverage": audio_coverage, "audio_files_seen": len(audio_files), "input_mode": input_report.get("input_mode"), "resampled_count": report["resample_summary"]["resampled_count"]}, ensure_ascii=False, indent=2))
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


def collect_audio_files(audio_dirs: Optional[Sequence[str]]) -> List[str]:
    """Collect wav/mp3-like files for optional real-audio V44 pairing."""
    if not audio_dirs:
        return []
    exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
    files: List[str] = []
    for d in audio_dirs:
        if not d:
            continue
        p = Path(d)
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
        elif p.exists():
            for ext in exts:
                files.extend(glob.glob(str(p / "**" / f"*{ext}"), recursive=True))
    return sorted(set(files))


def _match_key(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    keep = []
    for ch in stem:
        keep.append(ch if ch.isalnum() else "_")
    tokens = [t for t in "".join(keep).split("_") if t and t not in {"motion", "audio", "music", "wav", "npy", "clip"}]
    return "_".join(tokens)


def find_matching_audio_for_motion(motion_path: str | Path, audio_files: Sequence[str]) -> Optional[str]:
    """Best-effort same-stem / token-overlap pairing for Chang-E/change exports."""
    if not audio_files:
        return None
    key = _match_key(motion_path)
    if not key:
        return None
    audio_keys = [(_match_key(a), a) for a in audio_files]
    for ak, a in audio_keys:
        if ak == key:
            return a
    for ak, a in audio_keys:
        if ak and (ak in key or key in ak):
            return a
    kt = set(key.split("_"))
    best = (0.0, None)
    for ak, a in audio_keys:
        at = set(ak.split("_"))
        if not at:
            continue
        score = len(kt & at) / max(len(kt | at), 1)
        if score > best[0]:
            best = (score, a)
    return best[1] if best[0] >= 0.35 else None


def audio_feature_for_motion_clip(audio_path: str | Path, start_frame: int, frames: int, cfg: V46Config) -> np.ndarray:
    """Extract a 32D real audio descriptor aligned to a motion clip interval."""
    sr, wav = read_wav_mono(audio_path)
    st = max(0, int(round((start_frame / max(float(cfg.fps), 1e-6)) * sr)))
    ed = min(wav.size, int(round(((start_frame + frames) / max(float(cfg.fps), 1e-6)) * sr)))
    if ed - st < max(64, int(0.10 * sr)):
        st, ed = 0, wav.size
    return audio_global_features(wav[st:ed], sr).astype(np.float32)


def standardize_features(x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if mask is not None and np.asarray(mask).astype(bool).any():
        base = x[np.asarray(mask).astype(bool)]
    else:
        base = x
    mean = base.mean(axis=0, keepdims=True).astype(np.float32)
    std = (base.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return ((x - mean) / std).astype(np.float32), mean, std


def audio_slots(path: str | Path, cfg: V46Config, slot_seconds: float = 4.0, slots_json: Optional[str] = None) -> Tuple[List[dict], np.ndarray]:
    if slots_json and Path(slots_json).exists():
        # V46.12: slots_json can be either the old feature JSON or a new
        # external classical-music semantic JSON/NPZ.
        if str(slots_json).lower().endswith(".npz"):
            parsed = parse_external_music_semantic_file(slots_json, cfg)
            if parsed is not None:
                return parsed
        data = load_json(slots_json)
        slots = data.get("slots", data if isinstance(data, list) else [])
        if slots and isinstance(slots[0], dict) and ("probs" in slots[0] or "top_label" in slots[0] or "music_alignment_label" in slots[0]):
            parsed = parse_external_music_semantic_file(slots_json, cfg)
            if parsed is not None:
                return parsed
        feats = []
        for s in slots:
            base = np.asarray(s.get("feature", []), dtype=np.float32)
            if base.size < 32:
                base = np.pad(base, (0, 32 - base.size))
            feats.append(base[:32])
        return slots, np.stack(feats).astype(np.float32)
    external = load_external_music_semantic_slots(path, cfg, slot_seconds)
    if external is not None:
        return external
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
        slot_sem = audio_slot_classification_from_pseudo(pseudo, dur, energy, onset, dyn)
        # Encode music alignment labels in the same high channels as class_semantic.
        pseudo[22] = _label_index(slot_sem["energy_label"], ENERGY_LABELS) / max(1, len(ENERGY_LABELS) - 1)
        pseudo[23] = _label_index(slot_sem["rhythm_label"], RHYTHM_LABELS) / max(1, len(RHYTHM_LABELS) - 1)
        pseudo[26] = _label_index(slot_sem["music_alignment_label"], MUSIC_ALIGNMENT_LABELS) / max(1, len(MUSIC_ALIGNMENT_LABELS) - 1)
        slot_item = {"slot_id": i, "start": st / sr, "end": ed / sr, "duration": dur, "energy": energy, "onset": onset}
        slot_item.update(slot_sem)
        slots.append(slot_item)
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
    # V46.8 keeps this only as an explicitly marked smoke-test fallback.  For
    # motion-only Chang-E BVH, prefer unpaired real-audio semantic OT below.
    m = desc.copy().astype(np.float32)
    rng = np.random.default_rng(1234)
    m[:, 1:9] *= rng.normal(1.0, noise, size=m[:, 1:9].shape).astype(np.float32)
    m[:, 15:18] *= rng.normal(1.0, noise * 1.5, size=m[:, 15:18].shape).astype(np.float32)
    m += rng.normal(0.0, noise * 0.15, size=m.shape).astype(np.float32)
    return m.astype(np.float32)


def semantic_dims_and_weights() -> Tuple[np.ndarray, np.ndarray]:
    """Feature weights shared by unpaired audio-motion OT and retrieval fallback."""
    dims = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 22, 23, 24, 25, 26, 28, 29, 30], dtype=np.int64)
    weights = np.array([
        0.45,  # duration
        0.80, 0.95, 0.75, 0.70,  # root/energy/dynamics
        1.15, 1.00, 1.05, 0.90,  # joint/lower/upper energy
        0.45, 0.35,              # lower/upper ratio + contact calmness
        0.75, 0.75,              # foot speed
        0.90, 0.95, 0.65,        # onset/turn/root-y dynamics
        0.55, 0.55, 0.42, 0.42, 0.70,  # explicit class channels
        0.35, 0.42, 0.42,        # calm/percussive/turn affinity bits
    ], dtype=np.float32)
    return dims, weights


def load_unpaired_audio_feature_pool(audio_dirs: Optional[Sequence[str]], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:
    """Load real, unpaired music clips and convert them to slot-level 32D features."""
    files = collect_audio_files(audio_dirs)
    feats: List[np.ndarray] = []
    meta: List[dict] = []
    for f in files:
        try:
            slots, sf = audio_slots(f, cfg, slot_seconds=float(cfg.unpaired_audio_slot_seconds))
        except Exception as exc:
            print(f"[V46.11 WARN] failed unpaired audio feature extraction {f}: {exc}", file=sys.stderr)
            continue
        for slot, feat in zip(slots, sf):
            feats.append(feat.astype(np.float32))
            meta.append({"audio": str(f), "slot": dict(slot)})
    if not feats:
        return np.zeros((0, 32), dtype=np.float32), []
    return np.stack(feats).astype(np.float32), meta


def build_unpaired_audio_motion_pairs(db: dict, audio_dirs: Optional[Sequence[str]], cfg: V46Config) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]]:
    """
    Build semantic pseudo pairs for the realistic Chang-E case: BVH motions and
    music exist, but they are not synchronized.

    This is not paired supervision.  It uses real music slot features and
    motion descriptors, then solves a lightweight semantic assignment: energetic
    / onset-rich music slots are matched to high-energy / turning / expressive
    motion events, calm slots to low-density events, and durations are softly
    matched.  The checkpoint explicitly records this as
    ``unpaired_audio_semantic_ot``.
    """
    audio_raw, audio_meta = load_unpaired_audio_feature_pool(audio_dirs, cfg)
    if audio_raw.shape[0] < int(cfg.unpaired_min_audio_slots):
        return None

    motion_z = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "classification_ot_weight", getattr(cfg, "filename_semantic_ot_weight", 0.35))))
    desc_mean = np.asarray(db["desc_mean"], dtype=np.float32)
    desc_std = np.asarray(db["desc_std"], dtype=np.float32)
    music_z_all = ((audio_raw - desc_mean) / np.maximum(desc_std, 1e-6)).astype(np.float32)
    # Small motion-only datasets can have nearly-zero std on some descriptor
    # channels. Clip both modalities before OT/training to avoid a single
    # degenerate dimension dominating the semantic assignment.
    music_z_all = np.clip(music_z_all, -8.0, 8.0).astype(np.float32)
    motion_z = np.clip(motion_z, -8.0, 8.0).astype(np.float32)

    dims, weights = semantic_dims_and_weights()
    mz = music_z_all[:, dims]
    dz = motion_z[:, dims]
    # Weighted squared distance gives a stable OT-like semantic cost without
    # requiring paired labels.  Add a tiny deterministic jitter to break ties.
    diff = mz[:, None, :] - dz[None, :, :]
    cost = np.sum((diff * weights[None, None, :]) ** 2, axis=-1)
    rng = np.random.default_rng(int(cfg.seed) + 4607)
    cost = cost + rng.normal(0.0, 1e-5, size=cost.shape).astype(np.float32)

    topk = max(1, min(int(cfg.unpaired_positive_topk), motion_z.shape[0]))
    pairs_per = max(1, min(int(cfg.unpaired_pairs_per_audio_slot), topk))
    music_pairs: List[np.ndarray] = []
    motion_pairs: List[np.ndarray] = []
    pair_preview: List[dict] = []
    for ai in range(cost.shape[0]):
        # Take top-k compatible motions, then sample a few. This avoids one
        # audio slot collapsing to a single hub event and improves source spread.
        top = np.argpartition(cost[ai], topk - 1)[:topk]
        top = top[np.argsort(cost[ai, top])]
        chosen = top[:pairs_per]
        for mi in chosen:
            music_pairs.append(music_z_all[ai])
            motion_pairs.append(motion_z[int(mi)])
        if len(pair_preview) < 16:
            pair_preview.append({
                "audio": audio_meta[ai]["audio"],
                "slot_id": int(audio_meta[ai]["slot"].get("slot_id", ai)),
                "slot_energy": float(audio_meta[ai]["slot"].get("energy", 0.0)),
                "slot_music_semantic_label": str(audio_meta[ai]["slot"].get("music_semantic_top_label", audio_meta[ai]["slot"].get("music_alignment_label", ""))),
                "slot_external_music_semantic_source": str(audio_meta[ai]["slot"].get("external_music_semantic_source", "")),
                "top_motion_ids": [int(x) for x in top[:min(5, len(top))].tolist()],
                "top_costs": [float(cost[ai, int(x)]) for x in top[:min(5, len(top))].tolist()],
            })

    if len(music_pairs) < 2:
        return None
    music = np.stack(music_pairs).astype(np.float32)
    motion = np.stack(motion_pairs).astype(np.float32)
    has_external_sem = any(str(m.get("slot", {}).get("external_music_semantic_source", "")).startswith(("/", "filename_proxy", "output", ".")) for m in audio_meta)
    report_mode = "external_classical_music_semantic_ot" if has_external_sem else "unpaired_audio_semantic_ot"
    report = {
        "mode": report_mode,
        "audio_files": sorted(set(m["audio"] for m in audio_meta)),
        "num_audio_slots": int(audio_raw.shape[0]),
        "num_motion_events": int(motion_z.shape[0]),
        "num_training_pairs": int(music.shape[0]),
        "positive_topk": int(topk),
        "pairs_per_audio_slot": int(pairs_per),
        "semantic_dims": [int(x) for x in dims.tolist()],
        "semantic_weights": [float(x) for x in weights.tolist()],
        "pair_preview": pair_preview,
    }
    return music, motion, desc_mean.astype(np.float32), desc_std.astype(np.float32), report


def load_db(db_path: str | Path) -> dict:
    data = np.load(db_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def train_contrastive(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for V44 training.")
    cfg = V46Config.from_json(args.config).apply_env()
    sem_dirs = getattr(args, "music_semantic_dirs", None)
    if sem_dirs:
        cfg.external_music_semantic_dirs = os.pathsep.join([str(x) for x in sem_dirs])
    if getattr(args, "external_music_semantic_cmd", None):
        cfg.external_music_semantic_cmd = str(args.external_music_semantic_cmd)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    db = load_db(args.db)
    motion_db = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "filename_semantic_weight", 0.35)))
    supervision_mode = "weak_motion_proxy"
    pair_report: Dict[str, object] = {}

    if args.music_feature_npz and Path(args.music_feature_npz).exists():
        mf_npz = np.load(args.music_feature_npz)
        key = "music_z" if "music_z" in mf_npz.files else "music"
        mf = mf_npz[key].astype(np.float32)
        if mf.shape != motion_db.shape:
            raise ValueError(f"music feature shape {mf.shape} != motion feature shape {motion_db.shape}")
        music, music_mean, music_std = standardize_features(mf)
        motion = motion_db
        supervision_mode = "external_real_music_features"
        pair_report = {"mode": supervision_mode, "num_training_pairs": int(motion.shape[0])}
    elif "music" in db and "music_mask" in db and float(np.mean(np.asarray(db["music_mask"], dtype=np.float32) > 0.5)) >= float(cfg.audio_pair_min_coverage):
        mask = np.asarray(db["music_mask"], dtype=np.float32) > 0.5
        music_raw = np.asarray(db["music"], dtype=np.float32)
        music, music_mean, music_std = standardize_features(music_raw, mask=mask)
        motion = motion_db[mask]
        music = music[mask]
        supervision_mode = "paired_audio_features_from_db"
        pair_report = {"mode": supervision_mode, "paired_coverage": float(mask.mean()), "num_training_pairs": int(motion.shape[0])}
    else:
        unpaired_dirs = getattr(args, "unpaired_audio_dirs", None) or getattr(args, "audio_dirs", None)
        unpaired = build_unpaired_audio_motion_pairs(db, unpaired_dirs, cfg) if bool(cfg.unpaired_audio_enable) else None
        if unpaired is not None:
            music, motion, music_mean, music_std, pair_report = unpaired
            supervision_mode = str(pair_report.get("mode", "unpaired_audio_semantic_ot"))
        else:
            if bool(cfg.contrastive_require_real_music) or bool(cfg.unpaired_disable_motion_proxy):
                raise RuntimeError(
                    "No paired audio and no usable unpaired audio were found. "
                    "Put target music under test_music_bank/ or data/music/, or set "
                    "V46_UNPAIRED_DISABLE_MOTION_PROXY=0 for a smoke-test-only fallback."
                )
            music_raw = make_weak_music_features_from_motion(np.asarray(db["desc"], dtype=np.float32))
            mean = np.asarray(db["desc_mean"], dtype=np.float32)
            std = np.asarray(db["desc_std"], dtype=np.float32)
            music = (music_raw - mean) / std
            motion = motion_db
            music_mean, music_std = mean, std
            supervision_mode = "weak_motion_descriptor_proxy"
            pair_report = {"mode": supervision_mode, "warning": "motion descriptor self-distillation; do not report as real music-motion pairing", "num_training_pairs": int(motion.shape[0])}

    N = int(motion.shape[0])
    if N < 2:
        raise RuntimeError("V44 contrastive needs at least two training pairs.")
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
            print(f"[V44 contrastive] epoch={ep} loss={np.mean(losses):.5f} scale={model.logit_scale.exp().item():.2f} mode={supervision_mode}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "version": "v44_12_external_classical_music_semantic_grounding",
        "state_dict": model.state_dict(),
        "config": dataclasses.asdict(cfg),
        "feat_dim": motion.shape[1],
        "embed_dim": cfg.embed_dim,
        "supervision_mode": supervision_mode,
        "music_mean": music_mean.astype(np.float32),
        "music_std": music_std.astype(np.float32),
        "pair_report": pair_report,
    }
    torch.save(ckpt, out)
    print(json.dumps({"contrastive_ckpt": str(out), "num_pairs": int(N), "supervision_mode": supervision_mode, "pair_report": pair_report}, ensure_ascii=False, indent=2))
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


def sample_motion_window(paths: np.ndarray, target_len: int, cfg: Optional[V46Config] = None) -> np.ndarray:
    """Sample a training window and keep the EDGE-151D contract after resampling."""
    p = str(random.choice(paths.tolist()))
    m = np.load(p).astype(np.float32)
    if m.shape[0] == target_len:
        out = m
    elif m.shape[0] > target_len:
        st = random.randint(0, m.shape[0] - target_len)
        out = m[st:st + target_len]
    else:
        out = resample_motion_np(m, target_len)
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"sample_motion_window:{p}", derive_contact=True, project_rot=True)
    return out.astype(np.float32)


def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06, cfg: Optional[V46Config] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create seam corruption for V45/V46 without corrupting EDGE contact channels."""
    x = clean.copy().astype(np.float32)
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
            drift = np.zeros((1, D), dtype=np.float32)
            drift[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.5, size=(1, 3))
            drift[:, ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.18, size=(1, ROT6D_END - ROT6D_START))
            x[c:] += np.linspace(0, 1, T - c, dtype=np.float32)[:, None] * drift

    noise = np.zeros_like(x, dtype=np.float32)
    noise[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.10, size=(T, 3)).astype(np.float32)
    noise[:, ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.05, size=(T, ROT6D_END - ROT6D_START)).astype(np.float32)
    x += noise
    x, _ = enforce_edge151_contract_np(x, cfg, source_hint="degrade_for_refiner", derive_contact=True, project_rot=True)
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
            clean = sample_motion_window(paths, cfg.window_len, cfg)
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
            clean, _ = enforce_edge151_contract_np(
                clean, cfg, source_hint=f"train_diffusion_clean:{paths[idx]}", derive_contact=True, project_rot=True
            )
            retr, seam = degrade_for_refiner(clean, severity=0.045, cfg=cfg)
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
    model.music_mean = np.asarray(ckpt.get("music_mean", np.zeros((1, ckpt.get("feat_dim", 32)), dtype=np.float32)), dtype=np.float32)
    model.music_std = np.asarray(ckpt.get("music_std", np.ones((1, ckpt.get("feat_dim", 32)), dtype=np.float32)), dtype=np.float32)
    model.supervision_mode = ckpt.get("supervision_mode", "unknown")
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
    """V46.31 retrieval: contrastive similarity + curated Chang-E semantic event router."""
    desc = np.asarray(db["desc"], dtype=np.float32)
    desc_z = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "classification_retrieval_weight", getattr(cfg, "filename_semantic_retrieval_weight", 0.20))))
    mean = np.asarray(db["desc_mean"], dtype=np.float32)
    std = np.asarray(db["desc_std"], dtype=np.float32)
    if contrastive is not None and hasattr(contrastive, "music_mean") and hasattr(contrastive, "music_std"):
        music_mean = np.asarray(getattr(contrastive, "music_mean"), dtype=np.float32)
        music_std = np.asarray(getattr(contrastive, "music_std"), dtype=np.float32)
        music_z = (slot_feat - music_mean) / np.maximum(music_std, 1e-6)
    else:
        music_z = (slot_feat - mean) / std
    music_z = np.clip(music_z, -8.0, 8.0).astype(np.float32)
    desc_z = np.clip(desc_z, -8.0, 8.0).astype(np.float32)
    music_emb, motion_emb = embed_with_contrastive(contrastive, music_z, desc_z, cfg)
    sources = np.asarray(db["source_groups"], dtype=object)
    durations = np.asarray(db["durations"], dtype=np.float32)
    entries = np.asarray(db["entry"], dtype=np.float32); exits = np.asarray(db["exit"], dtype=np.float32)
    centry = np.asarray(db["contact_entry"], dtype=np.float32); cexit = np.asarray(db["contact_exit"], dtype=np.float32)
    dance_keys = np.asarray(db.get("dance_keys", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)
    labels_arr = np.asarray(db.get("labels", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)
    align_arr = np.asarray(db.get("music_alignment_labels", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)
    families = np.asarray(db.get("event_families", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)
    stages = np.asarray(db.get("motion_stage_roles", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)
    locomotion = np.asarray(db.get("locomotion_labels", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)
    support = np.asarray(db.get("support_labels", np.array(["unknown"] * len(desc), dtype=object)), dtype=object)
    sem_conf = np.asarray(db.get("semantic_confidence", np.ones(len(desc), dtype=np.float32)), dtype=np.float32)
    event_quality = np.asarray(db.get("event_quality_scores", np.ones(len(desc), dtype=np.float32)), dtype=np.float32)
    nat_min = np.asarray(db.get("natural_duration_min", np.ones(len(desc), dtype=np.float32) * 1.5), dtype=np.float32)
    nat_max = np.asarray(db.get("natural_duration_max", np.ones(len(desc), dtype=np.float32) * 4.0), dtype=np.float32)
    beams: List[Tuple[float, List[int], Dict[str, int]]] = [(0.0, [], {})]
    reports: List[dict] = []
    for i, slot in enumerate(slots):
        sim = music_emb[i] @ motion_emb.T
        slot_dur = max(float(slot.get("duration", durations.mean() if len(durations) else 1.0)), 1e-4)
        dur_cost = np.abs(np.log(np.maximum(durations, 1e-4) / slot_dur))
        class_bonus = semantic_label_match_bonus(slot, db, cfg)
        in_range = ((slot_dur >= nat_min) & (slot_dur <= nat_max)).astype(np.float32)
        center = np.maximum((nat_min + nat_max) * 0.5, 1e-4)
        natural_score = in_range + (1.0 - in_range) * np.exp(-np.abs(np.log(slot_dur / center))).astype(np.float32)
        quality_term = np.clip(event_quality, 0.0, 1.0)
        low_quality_penalty = np.maximum(0.0, float(getattr(cfg, "chang_e_min_event_quality", 0.22)) - quality_term)
        base_score = (sim - cfg.retrieval_warp_penalty * dur_cost + float(getattr(cfg, "semantic_routing_weight", 0.72)) * class_bonus + float(getattr(cfg, "route_natural_duration_weight", 0.20)) * natural_score + float(getattr(cfg, "event_quality_weight", 0.22)) * quality_term + 0.04 * np.clip(sem_conf, 0.0, 1.0) - 0.75 * low_quality_penalty)
        cand = np.argsort(-base_score)[: max(cfg.top_k, cfg.beam_size, int(getattr(cfg, "route_debug_topk", 10)))].tolist()
        new_beams: List[Tuple[float, List[int], Dict[str, int]]] = []
        for score, path, usage in beams:
            prev = path[-1] if path else None
            for idx in cand:
                sc = float(base_score[idx])
                src = str(sources[idx]); dk = str(dance_keys[idx]); fam = str(families[idx]); stg = str(stages[idx])
                sc -= float(getattr(cfg, "route_source_repeat_penalty", cfg.retrieval_source_penalty)) * usage.get("src::" + src, 0)
                sc -= float(getattr(cfg, "route_dance_key_repeat_penalty", 0.16)) * usage.get("dance::" + dk, 0)
                # V46.31: family diversity is local-window based and capped.
                # For 3-5 minute dances, global family counts inevitably grow;
                # an unbounded penalty can overpower the music semantic match and
                # force wrong rare families in the later song.
                fam_recent_window = max(1, int(getattr(cfg, "route_family_recent_window", 8)))
                fam_recent_count = sum(1 for p_idx in path[-fam_recent_window:] if str(families[p_idx]) == fam)
                fam_pen = float(getattr(cfg, "route_family_balance_penalty", 0.18)) * max(0, fam_recent_count - 1)
                fam_pen = min(float(getattr(cfg, "route_family_penalty_cap", 0.25)), fam_pen)
                sc -= fam_pen
                # V46.31: source-run hard penalty is consecutive-run only.
                # Global source usage above remains a soft diversity prior; do not
                # blacklist high-quality sources for the entire later song merely
                # because they were selected twice earlier.
                run_count = 0
                for p_idx in reversed(path):
                    if str(sources[p_idx]) == src:
                        run_count += 1
                    else:
                        break
                if run_count >= 2:
                    sc -= float(getattr(cfg, "route_source_run_hard_penalty", 0.30))
                if str(slot.get("role", "")) in {"motif", "motif_recall"} and usage.get("fam::" + fam, 0) > 0:
                    sc += float(getattr(cfg, "route_motif_recall_bonus", 0.12))
                if i == 0 and stg in {"intro", "intro_or_resolution"}:
                    sc += float(getattr(cfg, "route_stage_sequence_weight", 0.16))
                elif i >= len(slots) - 2 and stg in {"resolution", "anchor_or_resolution", "intro_or_resolution"}:
                    sc += float(getattr(cfg, "route_stage_sequence_weight", 0.16))
                elif str(slot.get("role", "")) in {"build_up", "climax", "accent"} and stg in {"build_up", "climax", "accent_or_climax"}:
                    sc += float(getattr(cfg, "route_stage_sequence_weight", 0.16)) * 0.8
                if prev is not None:
                    sc -= cfg.retrieval_transition_penalty * transition_cost(exits[prev], entries[idx], cexit[prev], centry[idx])
                    if src == str(sources[prev]):
                        sc -= cfg.retrieval_repeat_penalty
                    if fam == str(families[prev]):
                        sc -= float(getattr(cfg, "route_family_repeat_penalty", 0.12))
                ns = dict(usage)
                ns["src::" + src] = ns.get("src::" + src, 0) + 1; ns["dance::" + dk] = ns.get("dance::" + dk, 0) + 1; ns["fam::" + fam] = ns.get("fam::" + fam, 0) + 1
                new_beams.append((score + sc, path + [int(idx)], ns))
        new_beams.sort(key=lambda x: x[0], reverse=True); beams = new_beams[: cfg.beam_size]
        preview_n = max(1, min(int(getattr(cfg, "classification_report_topk", 8)), len(cand)))
        reports.append({"slot": i, "start": slot.get("start"), "end": slot.get("end"), "duration": slot.get("duration"), "slot_role": slot.get("role"), "slot_music_alignment_label": slot.get("music_alignment_label"), "slot_music_semantic_top_label": slot.get("music_semantic_top_label", slot.get("music_alignment_label")), "slot_preferred_dance_keys": slot.get("preferred_dance_keys", []), "top_candidate": int(cand[0]), "top_candidate_label": str(labels_arr[cand[0]]), "top_candidate_dance_key": str(dance_keys[cand[0]]), "top_candidate_event_family": str(families[cand[0]]), "top_candidate_stage_role": str(stages[cand[0]]), "top_candidate_support_label": str(support[cand[0]]), "top_candidate_locomotion_label": str(locomotion[cand[0]]), "top_candidate_event_quality": float(event_quality[cand[0]]), "top_candidate_music_alignment_label": str(align_arr[cand[0]]), "beam_best_score": float(beams[0][0]), "routing_policy": "V46.31 curated semantic Event-RAG: contrastive/descriptor + family/stage/support/locomotion + quality + diversity", "candidate_preview": [{"event_id": int(j), "score": float(base_score[int(j)]), "semantic_route_bonus": float(class_bonus[int(j)]), "natural_duration_score": float(natural_score[int(j)]), "event_quality": float(event_quality[int(j)]), "source": str(sources[int(j)]), "label": str(labels_arr[int(j)]), "dance_key": str(dance_keys[int(j)]), "event_family": str(families[int(j)]), "motion_stage_role": str(stages[int(j)]), "support_label": str(support[int(j)]), "locomotion_label": str(locomotion[int(j)]), "music_alignment_label": str(align_arr[int(j)])} for j in cand[:preview_n]]})
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


def concat_events_v46_31_overlap(event_paths: Sequence[str], target_durations: Sequence[float], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:
    """Concatenate retrieved RAG events under the EDGE-151D contract.

    V46.31 fix:
    The overlap cross-fade is now compensated locally per segment, not by a
    whole-song global resample.  Every music slot keeps its assigned net frame
    budget after overlap trimming, so local beat/phrase boundaries do not drift.
    We also keep the V46.28/V46.29 yaw-aligned overlap start, no root-Y ramp,
    ov==1 midpoint weighting, and safe one-frame overlap slicing.
    """
    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    target_lens = [max(cfg.min_event_frames, int(round(float(d) * cfg.fps))) for d in target_durations]
    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        m_raw = np.load(str(p)).astype(np.float32)
        m, pre_report = enforce_edge151_contract_np(
            m_raw, cfg, source_hint=f"concat_load:{p}", derive_contact=True, project_rot=True
        )
        target_len = int(target_lens[i])
        # V46.31: compensate overlap locally.  Incoming clips lose ov frames
        # when m = m[ov:] removes the overlapped prefix.  Rather than globally
        # resampling the entire final song, pre-extend non-first clips by the
        # maximum plausible overlap and then locally normalize their post-overlap
        # remainder back to target_len.  This preserves per-slot music timing.
        overlap_budget = int(max(0, getattr(cfg, "overlap", 0))) if pieces else 0
        local_resample_len = int(max(cfg.min_event_frames, target_len + overlap_budget))
        warp = local_resample_len / max(1, m.shape[0])
        m = resample_motion_np(m, local_resample_len).astype(np.float32)
        m, post_resample_report = enforce_edge151_contract_np(
            m, cfg, source_hint=f"concat_resample_local_timing:{p}", derive_contact=True, project_rot=True
        )
        used_overlap = 0
        align_report = None
        blend_report = None
        local_timing_report = {
            "expected_net_frames": int(target_len),
            "local_resample_frames_before_overlap": int(local_resample_len),
            "overlap_budget_frames": int(overlap_budget),
            "overlap_trim_frames": 0,
            "post_overlap_frames_before_local_fix": int(local_resample_len),
            "local_timing_fix_applied": False,
            "local_timing_fix_mode": "none",
        }
        if pieces:
            ov = min(int(cfg.overlap), len(pieces[-1]) // 3, len(m) // 3)
            used_overlap = int(max(0, ov))
            if ov > 0:
                # Align incoming m[0] to the previous overlap start in both yaw
                # and XZ position.  This avoids both speed surge and cross-heading
                # tearing inside the quaternion overlap window.
                ref = pieces[-1][-ov].copy()
                try:
                    yaw_ref = float(root_yaw_np(pieces[-1][-ov:][:1])[0])
                    yaw_m = float(root_yaw_np(m[:1])[0])
                    dyaw = float(np.arctan2(np.sin(yaw_ref - yaw_m), np.cos(yaw_ref - yaw_m)))
                except Exception:
                    yaw_ref, yaw_m, dyaw = 0.0, 0.0, 0.0
                m = rotate_motion_around_y_np(m, dyaw, pivot_xz=m[0, [ROOT_X_IDX, ROOT_Z_IDX]])
                delta_xz = ref[[ROOT_X_IDX, ROOT_Z_IDX]] - m[0, [ROOT_X_IDX, ROOT_Z_IDX]]
                m[:, ROOT_X_IDX] += float(delta_xz[0])
                m[:, ROOT_Z_IDX] += float(delta_xz[1])
                # Deliberately do not apply any root-Y ramp.  Height/contact
                # continuity is handled only inside the real overlap blend.
                m, align_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_overlap_start_yaw_align:{p}", derive_contact=True, project_rot=True
                )
                if align_report is None:
                    align_report = {}
                align_report.update({
                    "overlap_alignment_mode": "yaw_and_xz_to_overlap_start_no_root_y_ramp",
                    "overlap_ref_frame": "previous_event[-overlap]",
                    "yaw_ref": float(yaw_ref),
                    "yaw_incoming_before": float(yaw_m),
                    "dyaw_applied": float(dyaw),
                    "delta_xz_applied": [float(delta_xz[0]), float(delta_xz[1])],
                    "root_y_ramp_applied": False,
                })
                a = pieces[-1][-ov:].copy()
                b = m[:ov].copy()
                if ov == 1:
                    w_b = np.asarray([[0.5]], dtype=np.float32)
                else:
                    w_b = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
                blend, blend_report = blend_motion_overlap_np(
                    a, b, w_b, cfg, source_hint=f"concat_overlap_quat:{Path(str(p)).name}"
                )
                pieces[-1] = np.concatenate([pieces[-1][:-ov], blend], axis=0)
                pieces[-1], _ = enforce_edge151_contract_np(
                    pieces[-1], cfg, source_hint="concat_piece_after_quat_overlap", derive_contact=True, project_rot=True
                )
                m = m[ov:]
                local_timing_report["overlap_trim_frames"] = int(ov)
                local_timing_report["post_overlap_frames_before_local_fix"] = int(m.shape[0])
            else:
                m = align_next_to_prev(pieces[-1], m)
                m, align_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_align_no_overlap:{p}", derive_contact=True, project_rot=True
                )
                local_timing_report["post_overlap_frames_before_local_fix"] = int(m.shape[0])

            # V46.31: after overlap handling, repair only this incoming segment's
            # net length.  This prevents whole-song interpolation from smearing
            # contact steps and preserves local music slot boundaries.
            if int(m.shape[0]) != int(target_len):
                m = resample_motion_np(m, int(target_len)).astype(np.float32)
                m, local_fix_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_local_timing_fix:{p}", derive_contact=True, project_rot=True
                )
                local_timing_report.update({
                    "local_timing_fix_applied": True,
                    "local_timing_fix_mode": "segment_local_resample_after_overlap_trim",
                    "frames_after_local_timing_fix": int(m.shape[0]),
                    "contract_after_local_timing_fix": local_fix_report,
                })
            else:
                local_timing_report["frames_after_local_timing_fix"] = int(m.shape[0])

        pieces.append(m.astype(np.float32))
        rep.append({
            "path": str(p),
            "target_frames": int(target_len),
            "source_frames": int(m_raw.shape[0]),
            "local_resample_frames": int(local_resample_len),
            "warp": float(warp),
            "overlap": int(used_overlap),
            "boundary_blend_mode": "quaternion_rotation" if used_overlap > 0 else "none",
            "contract_pre": pre_report,
            "contract_after_resample": post_resample_report,
            "contract_after_align": align_report,
            "contract_overlap_blend": blend_report,
            "segment_local_timing": local_timing_report,
        })

    final = np.concatenate(pieces, axis=0).astype(np.float32)
    total_target_frames = int(sum(target_lens))
    timing_report = {
        "target_total_frames": int(total_target_frames),
        "frames_before_terminal_guard": int(final.shape[0]),
        "timing_frame_delta_before_terminal_guard": int(total_target_frames - final.shape[0]),
        "timing_compensation_applied": False,
        "timing_compensation_mode": "segment_local_overlap_compensation_no_global_resample",
        "global_resample_applied": False,
    }
    # Terminal guard only.  It should normally be a no-op because each segment is
    # locally length-corrected.  If a pathological one-frame edge case remains,
    # trim or hold the last frame instead of globally resampling thousands of
    # frames, so local beat/contact timing is not redistributed across the song.
    if total_target_frames > 0 and int(final.shape[0]) != int(total_target_frames):
        delta = int(total_target_frames - final.shape[0])
        if delta > 0:
            pad = np.repeat(final[-1:, :], delta, axis=0).astype(np.float32)
            final = np.concatenate([final, pad], axis=0).astype(np.float32)
            mode = "terminal_hold_last_frame_pad_no_global_resample"
        else:
            final = final[:total_target_frames].astype(np.float32)
            mode = "terminal_trim_no_global_resample"
        timing_report.update({
            "timing_compensation_applied": True,
            "timing_compensation_mode": mode,
            "terminal_delta_frames": int(delta),
        })
    timing_report["frames_after_terminal_guard"] = int(final.shape[0])
    final, final_report = enforce_edge151_contract_np(
        final, cfg, source_hint="concat_final", derive_contact=True, project_rot=True
    )
    if rep:
        rep[-1]["concat_timing_compensation"] = timing_report
        rep[-1]["concat_final_contract"] = final_report
    return final.astype(np.float32), rep





# ===== V46.32 TRANSITION-BUDGET PATCH START =====
def _v46_env_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _v46_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _v46_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def quat_slerp_np(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path quaternion SLERP / nlerp fallback.

    q0, q1: [...,4], w: broadcastable [...,1]. Returns normalized [...,4].
    This function is intentionally NumPy-only so it is usable during concat.
    """
    q0 = normalize_quat_np(np.asarray(q0, dtype=np.float32))
    q1 = normalize_quat_np(np.asarray(q1, dtype=np.float32))
    w = np.asarray(w, dtype=np.float32)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    # Use normalized linear interpolation near zero angle; true SLERP elsewhere.
    near = dot > 0.9995
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - w) * theta0) / np.maximum(sin0, 1e-8)
    s1 = np.sin(w * theta0) / np.maximum(sin0, 1e-8)
    qs = s0 * q0 + s1 * q1
    ql = (1.0 - w) * q0 + w * q1
    out = np.where(near, ql, qs)
    return normalize_quat_np(out).astype(np.float32)


def _v46_root_velocity(m: np.ndarray, at_end: bool) -> np.ndarray:
    if m.shape[0] < 2:
        return np.zeros(3, dtype=np.float32)
    if at_end:
        return (m[-1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - m[-2, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)
    return (m[1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - m[0, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)


def transition_len_for_boundary(prev: np.ndarray, curr: np.ndarray, target_len: int, cfg: V46Config) -> int:
    """Choose a boundary transition budget for the *incoming* slot.

    The length is risk-aware but capped by the current music slot so that the
    retrieved core motion remains dominant.  This implements the paper position:
    real events preserve cultural vocabulary, local generation repairs only seams.
    """
    min_t = _v46_env_int("V46_TRANSITION_MIN_FRAMES", 8)
    max_t = _v46_env_int("V46_TRANSITION_MAX_FRAMES", 24)
    ratio = _v46_env_float("V46_TRANSITION_RATIO", 0.18)
    min_core = _v46_env_int("V46_TRANSITION_MIN_CORE_FRAMES", max(18, int(getattr(cfg, "min_event_frames", 36) * 0.55)))
    base = int(round(float(target_len) * float(ratio)))

    try:
        exit_j = fk_24_np(prev[-min(len(prev), 3):])[-1]
        entry_j = fk_24_np(curr[:min(len(curr), 3)])[0]
        pose_gap = float(np.linalg.norm(exit_j - entry_j, axis=-1).mean())
    except Exception:
        pose_gap = 0.0
    try:
        yaw_gap = abs(float(np.arctan2(np.sin(root_yaw_np(prev[-1:])[0] - root_yaw_np(curr[:1])[0]),
                                      np.cos(root_yaw_np(prev[-1:])[0] - root_yaw_np(curr[:1])[0]))))
    except Exception:
        yaw_gap = 0.0
    # Small risk schedule: larger pose/yaw gap gets a longer bridge.
    risk_extra = int(round(np.clip(pose_gap * 16.0 + yaw_gap * 4.0, 0.0, 10.0)))
    L = int(np.clip(base + risk_extra, min_t, max_t))
    L = min(L, max(0, int(target_len) - int(min_core)))
    return int(max(0, L))


def motion_inbetween_np(left_ctx: np.ndarray, right_ctx: np.ndarray, length: int, cfg: V46Config,
                        source_hint: str = "v46_32_inbetween") -> np.ndarray:
    """Generate a kinematic transition in EDGE-151D space.

    The bridge interpolates root trajectory with cubic Hermite and rotations with
    quaternion shortest-path interpolation.  Contact channels are rebuilt by FK in
    enforce_edge151_contract_np, so no invalid gray contacts are preserved.
    """
    L = int(length)
    if L <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32)
    a = np.asarray(left_ctx[-1], dtype=np.float32).copy()
    b = np.asarray(right_ctx[0], dtype=np.float32).copy()
    out = np.repeat(a[None, :], L, axis=0).astype(np.float32)

    # Phase excludes exact endpoints to avoid duplicating previous last or next first.
    u = (np.arange(1, L + 1, dtype=np.float32) / float(L + 1))[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2

    p0 = a[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    p1 = b[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    v0 = _v46_root_velocity(left_ctx, at_end=True)
    v1 = _v46_root_velocity(right_ctx, at_end=False)
    # Limit velocity to prevent a transition budget from launching the body.
    vmax = _v46_env_float("V46_TRANSITION_ROOT_VEL_CLAMP_MPF", 0.055)
    v0 = np.clip(v0, -vmax, vmax)
    v1 = np.clip(v1, -vmax, vmax)
    root = h00 * p0[None] + h10 * (L * v0[None]) + h01 * p1[None] + h11 * (L * v1[None])
    out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root.astype(np.float32)

    # Rotation SLERP for all joints.
    Ra = rot6d_to_matrix_np(a[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    Rb = rot6d_to_matrix_np(b[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    qa = matrix_to_quat_np(Ra)[None, :, :]
    qb = matrix_to_quat_np(Rb)[None, :, :]
    q = quat_slerp_np(np.repeat(qa, L, axis=0), np.repeat(qb, L, axis=0), u[:, None, :])
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(R).reshape(L, -1)

    # Contacts are not linearly interpolated. Rebuild from FK/contact thresholds.
    out[:, 0:4] = 0.0
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=source_hint, derive_contact=True, project_rot=True)
    return out.astype(np.float32)


def align_event_core_to_prev_np(prev: np.ndarray, curr: np.ndarray, cfg: V46Config) -> Tuple[np.ndarray, dict]:
    """Yaw + XZ align the incoming event core to the previous exit."""
    out = np.asarray(curr, dtype=np.float32).copy()
    rep: Dict[str, object] = {"mode": "none"}
    if prev.shape[0] == 0 or out.shape[0] == 0:
        return out, rep
    try:
        yaw_ref = float(root_yaw_np(prev[-1:])[0])
        yaw_m = float(root_yaw_np(out[:1])[0])
        dyaw = float(np.arctan2(np.sin(yaw_ref - yaw_m), np.cos(yaw_ref - yaw_m)))
    except Exception:
        yaw_ref, yaw_m, dyaw = 0.0, 0.0, 0.0
    out = rotate_motion_around_y_np(out, dyaw, pivot_xz=out[0, [ROOT_X_IDX, ROOT_Z_IDX]])
    delta_xz = prev[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += float(delta_xz[0])
    out[:, ROOT_Z_IDX] += float(delta_xz[1])
    out, contract = enforce_edge151_contract_np(out, cfg, source_hint="v46_32_align_event_core_to_prev", derive_contact=True, project_rot=True)
    rep = {"mode": "yaw_xz_entry_to_prev_exit", "yaw_ref": yaw_ref, "yaw_incoming_before": yaw_m,
           "dyaw_applied": dyaw, "delta_xz_applied": [float(delta_xz[0]), float(delta_xz[1])],
           "root_y_ramp_applied": False, "contract": contract}
    return out.astype(np.float32), rep


def concat_events(event_paths: Sequence[str], target_durations: Sequence[float], cfg: V46Config) -> Tuple[np.ndarray, List[dict]]:
    """V46.32 transition-budgeted concatenation.

    When V46_TRANSITION_BUDGET_ENABLE=1, every music slot keeps its exact frame
    budget.  For slot i>0, the first frames are a learned/refinable transition
    budget bridging previous exit and current entry; the remaining frames are the
    retrieved core motion after light resampling.  If disabled, fall back to the
    preserved V46.31 overlap implementation.
    """
    if not _v46_env_bool("V46_TRANSITION_BUDGET_ENABLE", True):
        return concat_events_v46_31_overlap(event_paths, target_durations, cfg)

    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    cursor = 0
    target_lens = [max(int(getattr(cfg, "min_event_frames", 36)), int(round(float(d) * float(cfg.fps)))) for d in target_durations]
    core_warp_min = _v46_env_float("V46_CORE_WARP_MIN", 0.70)
    core_warp_max = _v46_env_float("V46_CORE_WARP_MAX", 1.45)
    min_core = _v46_env_int("V46_TRANSITION_MIN_CORE_FRAMES", max(18, int(getattr(cfg, "min_event_frames", 36) * 0.55)))

    prev_tail: Optional[np.ndarray] = None
    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        target_len = int(target_lens[i])
        m_raw = np.load(str(p)).astype(np.float32)
        m, pre_report = enforce_edge151_contract_np(m_raw, cfg, source_hint=f"v46_32_concat_load:{p}", derive_contact=True, project_rot=True)

        # Incoming transition budget is charged to the current music slot, so
        # the global music boundary remains locked and total frames are exact.
        transition_len = 0
        align_report = None
        transition_report = {"enabled": False, "frames": 0}
        if prev_tail is not None:
            # First make a provisional core to estimate boundary risk.
            provisional_core_len = max(min_core, target_len - _v46_env_int("V46_TRANSITION_MIN_FRAMES", 8))
            provisional_core = resample_motion_np(m, provisional_core_len).astype(np.float32)
            provisional_core, _ = enforce_edge151_contract_np(provisional_core, cfg, source_hint=f"v46_32_provisional_core:{p}", derive_contact=True, project_rot=True)
            provisional_core, _ = align_event_core_to_prev_np(prev_tail, provisional_core, cfg)
            transition_len = transition_len_for_boundary(prev_tail, provisional_core, target_len, cfg)
        core_len = int(max(min_core, target_len - transition_len))
        if core_len + transition_len != target_len:
            # Preserve exact slot budget after respecting min_core.
            transition_len = int(max(0, target_len - core_len))

        src_len = max(1, int(m.shape[0]))
        warp = float(core_len / src_len)
        # Clamp only the core warp; if extreme, still keep target by transition
        # budget but report the violation.  This avoids hard failures in low-resource data.
        warp_clamped = False
        if bool(_v46_env_bool("V46_CORE_WARP_CLAMP_ENABLE", True)) and src_len > 1:
            desired = int(round(np.clip(core_len / src_len, core_warp_min, core_warp_max) * src_len))
            if desired != core_len and desired >= min_core and desired <= target_len:
                core_len = int(desired)
                transition_len = int(max(0, target_len - core_len))
                warp = float(core_len / src_len)
                warp_clamped = True

        core = resample_motion_np(m, core_len).astype(np.float32)
        core, core_report = enforce_edge151_contract_np(core, cfg, source_hint=f"v46_32_core_resample:{p}", derive_contact=True, project_rot=True)

        span_start = cursor
        bridge = np.zeros((0, EDGE_DIM), dtype=np.float32)
        if prev_tail is not None:
            core, align_report = align_event_core_to_prev_np(prev_tail, core, cfg)
            if transition_len > 0 and _v46_env_bool("V46_TRANSITION_INBETWEEN_ENABLE", True):
                bridge = motion_inbetween_np(prev_tail, core, transition_len, cfg, source_hint=f"v46_32_transition:{Path(str(p)).name}")
                transition_report = {"enabled": True, "mode": "kinematic_root_hermite_rot_slerp_contact_rebuild",
                                     "frames": int(transition_len), "span": [int(cursor), int(cursor + transition_len)],
                                     "mask_value": 1.0}
            elif transition_len > 0:
                bridge = resample_motion_np(np.stack([prev_tail[-1], core[0]], axis=0), transition_len).astype(np.float32)
                bridge, _ = enforce_edge151_contract_np(bridge, cfg, source_hint=f"v46_32_linear_transition_fallback:{p}", derive_contact=True, project_rot=True)
                transition_report = {"enabled": True, "mode": "linear_fallback_contract_projected", "frames": int(transition_len),
                                     "span": [int(cursor), int(cursor + transition_len)], "mask_value": 1.0}

        piece = np.concatenate([bridge, core], axis=0).astype(np.float32) if bridge.shape[0] else core.astype(np.float32)
        # Exact per-slot terminal guard.
        if piece.shape[0] != target_len:
            if piece.shape[0] < target_len:
                pad = np.repeat(piece[-1:, :], target_len - piece.shape[0], axis=0).astype(np.float32)
                piece = np.concatenate([piece, pad], axis=0)
                timing_fix = "slot_terminal_hold_pad"
            else:
                piece = piece[:target_len]
                timing_fix = "slot_terminal_trim"
        else:
            timing_fix = "none"
        piece, piece_report = enforce_edge151_contract_np(piece, cfg, source_hint=f"v46_32_piece_final:{p}", derive_contact=True, project_rot=True)
        pieces.append(piece.astype(np.float32))
        span_end = cursor + int(piece.shape[0])
        prev_tail = piece[-min(len(piece), max(3, int(transition_len) + 3)):].copy()

        rep.append({
            "path": str(p),
            "target_frames": int(target_len),
            "source_frames": int(m_raw.shape[0]),
            "core_frames": int(core_len),
            "transition_in_frames": int(transition_len),
            "transition_span": transition_report.get("span", []),
            "slot_span": [int(span_start), int(span_end)],
            "warp": float(warp),
            "core_warp_clamped": bool(warp_clamped),
            "boundary_blend_mode": "transition_budgeted_inbetweening" if transition_len > 0 else "none",
            "transition_budget_mode": "v46_32_slot_budget_core_resample_boundary_inbetween",
            "contract_pre": pre_report,
            "contract_core": core_report,
            "contract_after_align": align_report,
            "contract_piece": piece_report,
            "transition_report": transition_report,
            "slot_timing_fix": timing_fix,
        })
        cursor = span_end

    final = np.concatenate(pieces, axis=0).astype(np.float32) if pieces else np.zeros((0, EDGE_DIM), dtype=np.float32)
    total_target_frames = int(sum(target_lens))
    timing_report = {
        "target_total_frames": int(total_target_frames),
        "frames_before_terminal_guard": int(final.shape[0]),
        "timing_frame_delta_before_terminal_guard": int(total_target_frames - final.shape[0]),
        "timing_compensation_applied": False,
        "timing_compensation_mode": "v46_32_slot_local_transition_budget_exact",
        "global_resample_applied": False,
    }
    if total_target_frames > 0 and final.shape[0] != total_target_frames:
        delta = int(total_target_frames - final.shape[0])
        if delta > 0:
            final = np.concatenate([final, np.repeat(final[-1:, :], delta, axis=0)], axis=0).astype(np.float32)
            mode = "terminal_hold_last_frame_pad_no_global_resample"
        else:
            final = final[:total_target_frames].astype(np.float32)
            mode = "terminal_trim_no_global_resample"
        timing_report.update({"timing_compensation_applied": True, "timing_compensation_mode": mode, "terminal_delta_frames": int(delta)})
    timing_report["frames_after_terminal_guard"] = int(final.shape[0])
    final, final_report = enforce_edge151_contract_np(final, cfg, source_hint="v46_32_concat_final", derive_contact=True, project_rot=True)
    if rep:
        rep[-1]["concat_timing_compensation"] = timing_report
        rep[-1]["concat_final_contract"] = final_report
    return final.astype(np.float32), rep


def transition_mask_from_concat_report(T: int, concat_report: Sequence[dict], default_width: int = 24) -> Tuple[np.ndarray, List[int], List[dict]]:
    """Build refiner/diffusion mask from explicit V46.32 transition spans.

    Falls back to seam-center halos if a report comes from the old overlap path.
    """
    mask = np.zeros((int(T), 1), dtype=np.float32)
    centers: List[int] = []
    spans: List[dict] = []
    halo = _v46_env_int("V46_TRANSITION_MASK_HALO", 4)
    for r in concat_report:
        sp = r.get("transition_span", []) if isinstance(r, dict) else []
        if isinstance(sp, (list, tuple)) and len(sp) == 2:
            a = max(0, int(sp[0]) - halo)
            b = min(int(T), int(sp[1]) + halo)
            if b > a:
                mask[a:b, 0] = 1.0
                centers.append((a + b) // 2)
                spans.append({"start": int(a), "end": int(b), "source": "transition_span"})
    if not centers:
        acc = 0
        for r in list(concat_report)[:-1]:
            tf = int(r.get("target_frames", 0)) if isinstance(r, dict) else 0
            acc += tf
            centers.append(acc)
        mask = make_boundary_mask(int(T), centers, width=default_width)
        spans = [{"center": int(c), "start": int(max(0, c - default_width)), "end": int(min(T, c + default_width)), "source": "fallback_seam_center"} for c in centers]
    return mask.astype(np.float32), centers, spans
# ===== V46.32 TRANSITION-BUDGET PATCH END =====

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
    """Apply V45 with sliding-window inference and quaternion rotation fusion."""
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        seam_centers = []
        for a, b in contiguous_regions(seam_mask[:, 0] > 0.5):
            seam_centers.append((a + b) // 2)
        refined = analytic_residual_refine(motion, seam_centers)
        refined, _ = enforce_edge151_contract_np(
            refined, cfg, source_hint="apply_refiner_model:analytic", derive_contact=True, project_rot=True
        )
        return refined.astype(np.float32)

    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model = TemporalRefiner(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)

    with torch.no_grad():
        for st, ed in sliding_window_ranges(T, win, hop):
            chunk = motion[st:ed]
            mask = seam_mask[st:ed]
            orig_len = len(chunk)
            if orig_len < win:
                chunk_in = resample_motion_np(chunk, win)
                mask_in = resample_motion_np(mask, win)
            else:
                chunk_in = chunk
                mask_in = mask
            chunk_in, _ = enforce_edge151_contract_np(
                chunk_in, cfg, source_hint="apply_refiner_model:input_chunk", derive_contact=True, project_rot=True
            )
            x = torch.from_numpy(chunk_in[None]).float().to(cfg.device)
            c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
            sm = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
            delta = model(x, c, sm)
            y = x + delta * (0.2 + 0.8 * sm)
            y_np = y[0].detach().cpu().numpy()
            if orig_len < win:
                y_np = resample_motion_np(y_np, orig_len)
            y_np, _ = enforce_edge151_contract_np(
                y_np, cfg, source_hint="apply_refiner_model:output_chunk", derive_contact=True, project_rot=True
            )
            w = overlap_add_weight_np(orig_len, st, T, hop, win)
            accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, y_np, w, st, ed)

    out, _ = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint="apply_refiner_model:final"
    )
    return out.astype(np.float32)


def apply_diffusion_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: V46Config) -> np.ndarray:
    """Apply V46 diffusion with sliding-window inference and quaternion rotation fusion."""
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        motion, _ = enforce_edge151_contract_np(
            motion, cfg, source_hint="apply_diffusion_model:disabled", derive_contact=True, project_rot=True
        )
        return motion.astype(np.float32)

    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    Tdiff = int(ckpt.get("diffusion_steps", cfg.diffusion_steps))
    model = DiffusionDenoiser(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    betas, alphas, abar = make_beta_schedule(Tdiff, torch.device(cfg.device))

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)

    with torch.no_grad():
        for st, ed in sliding_window_ranges(T, win, hop):
            retr_np = motion[st:ed]
            mask_np = seam_mask[st:ed]
            orig_len = len(retr_np)
            if orig_len < win:
                retr_in = resample_motion_np(retr_np, win)
                mask_in = resample_motion_np(mask_np, win)
            else:
                retr_in = retr_np
                mask_in = mask_np
            retr_in, _ = enforce_edge151_contract_np(
                retr_in, cfg, source_hint="apply_diffusion_model:retrieval_chunk", derive_contact=True, project_rot=True
            )
            retr = torch.from_numpy(retr_in[None]).float().to(cfg.device)
            mask = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
            c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
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
                x = retr * (1.0 - 0.65 * mask) + x * (0.65 * mask)
            y = x[0].detach().cpu().numpy()
            if orig_len < win:
                y = resample_motion_np(y, orig_len)
            y, _ = enforce_edge151_contract_np(
                y, cfg, source_hint="apply_diffusion_model:output_chunk", derive_contact=True, project_rot=True
            )
            w = overlap_add_weight_np(orig_len, st, T, hop, win)
            accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, y, w, st, ed)

    out, _ = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint="apply_diffusion_model:final"
    )
    return out.astype(np.float32)


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
    V46.2 root-Y safety pass.

    Fixes three common post-process artifacts:
    1) Damping Snap Bug: landing damping length is the real contact duration,
       so the damping dip is exactly zero before the next flight frame.
    2) C1 Discontinuity: flight/parabola blending uses a Hanning/sin^2 gate,
       whose first derivative is zero at takeoff and landing boundaries.
    3) Micro-Flight Shock Bug: landing damping is applied only when the
       immediately preceding flight island is a real effective flight. The same
       cfg.root_y_min_flight_frames threshold gates both parabola and damping,
       so a 1-2 frame denoising gap cannot create a ghost landing hit.
    4) Space-Jump Bug: extremely long no-contact islands are treated as broken
       contact labels / bad upstream generation and are fused off before the
       ballistic parabola can launch the root several meters into the air.

    Only the legal EDGE root-Y channel is edited here; lower-body IK later writes
    legal lower-body rot6d channels.
    """
    out = motion.copy().astype(np.float32)
    if not bool(cfg.root_y_physics_enable) or out.shape[0] < 4:
        return out, {"enabled": False, "reason": "disabled_or_too_short"}

    root_y0 = out[:, ROOT_Y_IDX].copy()
    any_contact = contacts.any(axis=1)
    is_flight = ~any_contact
    fps = max(float(cfg.fps), 1e-6)
    min_effective_flight = max(1, int(cfg.root_y_min_flight_frames))
    max_biological_flight_s = float(max(cfg.root_y_max_flight_seconds, 1.0 / fps))
    max_biological_flight_frames = max(min_effective_flight, int(round(max_biological_flight_s * fps)))

    flight_applied = 0
    flight_skipped_micro = 0
    flight_skipped_space_jump = 0
    for start, end in contiguous_regions(is_flight):
        n = end - start
        if n < min_effective_flight:
            flight_skipped_micro += 1
            continue
        air_duration = n / fps
        # V46.3 biological fuse: a 2+ second no-contact interval in generated
        # long dance is almost always broken contact labels / bad retrieval, not
        # a valid human jump. Injecting a physical parabola would create a
        # multi-meter "space launch". Preserve the native root trajectory instead.
        if air_duration > max_biological_flight_s:
            flight_skipped_space_jump += 1
            continue
        left = max(0, start - 1)
        right = min(len(root_y0) - 1, end)
        if right <= left:
            continue
        y0 = float(root_y0[left])
        y1 = float(root_y0[right])
        duration = max((right - left) / fps, 1.0 / fps)
        v0 = (y1 - y0 + 0.5 * 9.81 * duration * duration) / duration
        for k, ti in enumerate(range(start, end)):
            # Use exact endpoint phases for zero-value/zero-slope blend.
            phase = 0.0 if n <= 1 else k / float(n - 1)
            gate = float(cfg.root_y_flight_strength) * float(c1_hanning_window01(phase))
            tau = (ti - left) / fps
            parabola = y0 + v0 * tau - 0.5 * 9.81 * tau * tau
            out[ti, ROOT_Y_IDX] = (1.0 - gate) * out[ti, ROOT_Y_IDX] + gate * parabola
        flight_applied += 1

    damping_applied = 0
    damping_skipped_micro_flight = 0
    damping_skipped_space_jump_flight = 0
    damping_preview: List[Dict[str, object]] = []
    for start, end in contiguous_regions(any_contact):
        # Only damp actual landings: a contact island that follows flight.
        if start <= 0 or any_contact[start - 1]:
            continue

        # V46.2 terminal guard: trace the immediately preceding no-contact
        # island. Damping is allowed only if this flight island was long enough
        # to receive the ballistic treatment. This preserves logical
        # conservation between parabola and landing damping, and prevents a
        # 1-2 frame per-foot-filter gap from creating a ghost impact dip.
        prev_flight_end = start
        prev_flight_start = prev_flight_end
        while prev_flight_start > 0 and not bool(any_contact[prev_flight_start - 1]):
            prev_flight_start -= 1
        prev_flight_len = prev_flight_end - prev_flight_start
        if prev_flight_len < min_effective_flight:
            damping_skipped_micro_flight += 1
            if len(damping_preview) < 24:
                damping_preview.append({
                    "start": int(start),
                    "end": int(end),
                    "frames": int(end - start),
                    "skipped": True,
                    "reason": "preceding_micro_flight",
                    "preceding_flight_start": int(prev_flight_start),
                    "preceding_flight_end": int(prev_flight_end),
                    "preceding_flight_frames": int(prev_flight_len),
                    "min_effective_flight_frames": int(min_effective_flight),
                })
            continue
        prev_flight_duration = prev_flight_len / fps
        if prev_flight_duration > max_biological_flight_s or prev_flight_len > max_biological_flight_frames:
            damping_skipped_space_jump_flight += 1
            if len(damping_preview) < 24:
                damping_preview.append({
                    "start": int(start),
                    "end": int(end),
                    "frames": int(end - start),
                    "skipped": True,
                    "reason": "preceding_space_jump_flight",
                    "preceding_flight_start": int(prev_flight_start),
                    "preceding_flight_end": int(prev_flight_end),
                    "preceding_flight_frames": int(prev_flight_len),
                    "preceding_flight_seconds": float(prev_flight_duration),
                    "max_biological_flight_seconds": float(max_biological_flight_s),
                })
            continue

        contact_len = end - start
        max_damp_frames = max(3, int(round(float(cfg.root_y_damping_max_seconds) * fps)))
        n = min(contact_len, max_damp_frames)
        if n <= 2:
            continue
        # V46.8 critical fix: damping is capped to an early post-touchdown window
        # instead of stretching across the whole contact island.  The chosen
        # window still has zero dip at both ends, so it is C0-safe at the next
        # untouched frame but cannot create a delayed squat in long support.
        max_abs_dip = 0.0
        for k, ti in enumerate(range(start, start + n)):
            phase = k / float(max(n - 1, 1))
            gate = float(c1_hanning_window01(phase))
            # Decay biases the cushion toward the landing instant while the
            # Hanning gate keeps both value and first derivative zero at ends.
            dip = float(cfg.root_y_damping_max_dip) * math.exp(-4.0 * phase) * gate
            out[ti, ROOT_Y_IDX] -= dip
            max_abs_dip = max(max_abs_dip, abs(dip))
        damping_applied += 1
        if len(damping_preview) < 24:
            damping_preview.append({
                "start": int(start),
                "end": int(end),
                "frames": int(contact_len),
                "damping_frames": int(n),
                "max_damping_frames": int(max_damp_frames),
                "max_dip_m": float(max_abs_dip),
                "capped_early_duration": True,
                "capped_seconds": float(cfg.root_y_damping_max_seconds),
                "preceding_flight_frames": int(prev_flight_len),
                "effective_flight_gated": True,
            })

    delta = out[:, ROOT_Y_IDX] - root_y0
    return out.astype(np.float32), {
        "enabled": True,
        "version": "v46_3_biological_max_flight_fused_root_y_physics",
        "fixes_damping_snap": True,
        "fixes_c1_discontinuity": True,
        "fixes_micro_flight_shock": True,
        "fixes_space_jump_bug": True,
        "flight_gate": "hanning_sin_squared",
        "damping_duration": "capped_early_contact_window",
        "damping_max_seconds": float(cfg.root_y_damping_max_seconds),
        "damping_requires_effective_preceding_flight": True,
        "ballistic_requires_biological_flight_duration": True,
        "max_biological_flight_seconds": float(max_biological_flight_s),
        "max_biological_flight_frames": int(max_biological_flight_frames),
        "min_effective_flight_frames": int(min_effective_flight),
        "flight_segments_applied": int(flight_applied),
        "flight_segments_skipped_micro": int(flight_skipped_micro),
        "flight_segments_skipped_space_jump": int(flight_skipped_space_jump),
        "landing_damping_applied": int(damping_applied),
        "landing_damping_skipped_micro_flight": int(damping_skipped_micro_flight),
        "landing_damping_skipped_space_jump_flight": int(damping_skipped_space_jump_flight),
        "damping_preview": damping_preview,
        "delta_mean": float(delta.mean()),
        "delta_p95_abs": float(np.percentile(np.abs(delta), 95)),
        "delta_max_abs": float(np.max(np.abs(delta))),
    }


def generate_ik_targets_np(native_foot: np.ndarray, contacts: np.ndarray, cfg: V46Config, root_xz: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Generate V43 lower-body IK targets with a V46.4 sliding-anchor cloud-step guard.

    The old span-only guard caused a Footskate Forgiveness Paradox: the worse a
    slow AI foot-drift became, the more likely it was to exceed the XZ threshold
    and be released from repair. This version never uses ``continue`` as an
    amnesty for large contact travel.

    Decision rule:
    - Large span + sufficiently high mean velocity (+ not absurdly bursty) is
      considered cloud-step only when it is also root/CoM-consistent: the root
      moves, the foot direction agrees with root direction, and foot-relative-to-root
      drift stays bounded. It then receives a moving local-window anchor.
    - Large span + low mean velocity is treated as AI dark drift / footskate and
      is still locked to a static contact-internal anchor for IK repair.
    - Short/static contacts use the same static anchor as before.

    Targets are initialized from native FK positions, and non-contact frames are
    never edited.
    """
    targets = native_foot.copy().astype(np.float32)
    locked_segments = 0
    sliding_anchor_segments = 0
    dark_drift_locked_segments = 0
    root_inconsistent_locked_segments = 0
    skipped_short = 0
    preview: List[Dict[str, object]] = []

    win = max(3, int(cfg.ik_sliding_anchor_window))
    if win % 2 == 0:
        win += 1
    half = win // 2
    speed_thr = float(cfg.ik_cloud_step_speed_mps)
    span_thr = float(cfg.ik_slide_release_m)
    min_frames = int(cfg.ik_slide_release_min_frames)
    speed_cv_max = float(cfg.ik_cloud_speed_cv_max)
    root_min_travel = float(cfg.ik_cloud_root_min_travel_m)
    direction_cos_min = float(cfg.ik_cloud_direction_cos_min)
    rel_span_max = float(cfg.ik_cloud_root_foot_rel_max_m)
    if root_xz is not None:
        root_xz = np.asarray(root_xz, dtype=np.float32)
        if root_xz.ndim != 2 or root_xz.shape[0] != native_foot.shape[0] or root_xz.shape[1] != 2:
            root_xz = None

    for f in range(native_foot.shape[1]):
        for start, end in contiguous_regions(contacts[:, f]):
            length = end - start
            if length < 3:
                skipped_short += 1
                continue

            seg = native_foot[start:end, f, :].astype(np.float32)
            seg_xz = seg[:, [0, 2]]
            span = float(np.linalg.norm(seg_xz.max(axis=0) - seg_xz.min(axis=0))) if length > 1 else 0.0
            step = np.linalg.norm(seg_xz[1:] - seg_xz[:-1], axis=-1) if length > 1 else np.zeros((0,), dtype=np.float32)
            arc = float(step.sum())
            duration_s = max((length - 1) / max(float(cfg.fps), 1e-6), 1.0 / max(float(cfg.fps), 1e-6))
            mean_speed_mps = float(arc / duration_s)
            inst_speed = step * float(cfg.fps) if step.size else np.zeros((0,), dtype=np.float32)
            speed_std = float(inst_speed.std()) if inst_speed.size else 0.0
            speed_mean = float(inst_speed.mean()) if inst_speed.size else 0.0
            speed_cv = float(speed_std / max(speed_mean, 1e-6)) if inst_speed.size else 0.0
            path_efficiency = float(span / max(arc, 1e-6)) if arc > 1e-6 else 1.0

            is_large_contact_travel = length >= min_frames and span > span_thr

            # V46.8 root-foot relative test.  Smooth high-speed foot travel is
            # not sufficient evidence for an intentional Dunhuang cloud-step:
            # severe AI footskate can also be smooth.  A true cloud-step should
            # be supported by root/CoM translation in a compatible direction and
            # should not show unbounded foot motion relative to the root.
            root_span = 0.0
            root_foot_rel_span = 0.0
            foot_root_cos = 0.0
            root_consistent = False
            if root_xz is not None and length > 1:
                root_seg = root_xz[start:end].astype(np.float32)
                root_span = float(np.linalg.norm(root_seg.max(axis=0) - root_seg.min(axis=0)))
                foot_delta = seg_xz[-1] - seg_xz[0]
                root_delta = root_seg[-1] - root_seg[0]
                denom = float(np.linalg.norm(foot_delta) * np.linalg.norm(root_delta))
                foot_root_cos = float(np.dot(foot_delta, root_delta) / max(denom, 1e-8)) if denom > 1e-8 else 0.0
                rel = seg_xz - root_seg
                root_foot_rel_span = float(np.linalg.norm(rel.max(axis=0) - rel.min(axis=0)))
                root_consistent = bool(
                    root_span >= root_min_travel
                    and foot_root_cos >= direction_cos_min
                    and root_foot_rel_span <= rel_span_max
                )
            else:
                # Backward-compatible fallback when root is unavailable.  It is
                # intentionally conservative: only very efficient, stable motion
                # can use sliding anchor without root evidence.
                root_consistent = bool(path_efficiency > 0.72 and speed_cv <= min(speed_cv_max, 1.0))

            velocity_consistent = bool(mean_speed_mps >= speed_thr and speed_cv <= speed_cv_max)
            is_cloud_step = bool(is_large_contact_travel and velocity_consistent and root_consistent)

            if is_cloud_step:
                # V46.4 critical fix: use a sliding local-window anchor rather
                # than releasing the target. This preserves intentional support
                # travel while smoothing high-frequency foot jitter.
                sliding_anchor_segments += 1
                for k, t in enumerate(range(start, end)):
                    lo = max(0, k - half)
                    hi = min(length, k + half + 1)
                    local_anchor = seg[lo:hi].mean(axis=0)
                    # Full XYZ mean is used intentionally: XZ follows the slide,
                    # Y is smoothed to avoid contact-height flicker.
                    targets[t, f] = local_anchor
                if len(preview) < 32:
                    preview.append({
                        "foot": int(f), "start": int(start), "end": int(end),
                        "frames": int(length), "mode": "sliding_anchor_cloud_step",
                        "xz_span_m": span, "arc_m": arc,
                        "mean_speed_mps": mean_speed_mps,
                        "speed_cv": speed_cv,
                        "path_efficiency": path_efficiency,
                        "root_span_m": root_span,
                        "root_foot_rel_span_m": root_foot_rel_span,
                        "foot_root_direction_cos": foot_root_cos,
                        "root_consistent": bool(root_consistent),
                        "span_threshold_m": span_thr,
                        "speed_threshold_mps": speed_thr,
                        "root_min_travel_m": root_min_travel,
                        "direction_cos_min": direction_cos_min,
                        "root_foot_rel_max_m": rel_span_max,
                        "window_frames": int(win),
                    })
                continue

            # Static anchor path. This deliberately catches large but slow AI
            # drift instead of forgiving it.
            if is_large_contact_travel:
                dark_drift_locked_segments += 1
                if velocity_consistent and not root_consistent:
                    root_inconsistent_locked_segments += 1

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
                    "xz_span_m": span, "arc_m": arc,
                    "mean_speed_mps": mean_speed_mps,
                    "speed_cv": speed_cv,
                    "path_efficiency": path_efficiency,
                    "root_span_m": root_span,
                    "root_foot_rel_span_m": root_foot_rel_span,
                    "foot_root_direction_cos": foot_root_cos,
                    "root_consistent": bool(root_consistent),
                    "velocity_consistent": bool(velocity_consistent),
                    "large_slow_drift_locked": bool(is_large_contact_travel),
                    "root_inconsistent_locked": bool(is_large_contact_travel and velocity_consistent and not root_consistent),
                    "anchor_source": "contact_internal_first_frames",
                })

    diff = np.linalg.norm(targets - native_foot, axis=-1)
    non_contact = ~contacts
    meta = {
        "version": "v46_4_sliding_anchor_cloud_step_target_generator",
        "fixes_footskate_forgiveness_paradox": True,
        "intentional_slide_guard": "root_aware_velocity_classified_sliding_anchor",
        "no_span_only_release": True,
        "fixes_smooth_dark_drift_cloudstep_false_positive": True,
        "cloud_step_speed_threshold_mps": float(speed_thr),
        "slide_span_threshold_m": float(span_thr),
        "sliding_anchor_window_frames": int(win),
        "cloud_speed_cv_max": float(speed_cv_max),
        "cloud_root_min_travel_m": float(root_min_travel),
        "cloud_direction_cos_min": float(direction_cos_min),
        "cloud_root_foot_rel_max_m": float(rel_span_max),
        "root_aware_guard_enabled": bool(root_xz is not None),
        "locked_segments": int(locked_segments),
        "sliding_anchor_segments": int(sliding_anchor_segments),
        "dark_drift_locked_segments": int(dark_drift_locked_segments),
        "root_inconsistent_locked_segments": int(root_inconsistent_locked_segments),
        "released_slide_segments": 0,
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
    targets, target_meta = generate_ik_targets_np(native_foot, contacts, cfg, root_xz=motion_base[:, [ROOT_X_IDX, ROOT_Z_IDX]])
    device = torch.device(cfg.device)
    out_all = motion_base.copy().astype(np.float32)
    reports = []
    T = motion.shape[0]
    chunk = int(cfg.ik_chunk)
    overlap = max(0, int(cfg.ik_chunk_overlap))
    stride = max(1, chunk - overlap)
    # V46.8: independent chunk solves are merged by weighted accumulation,
    # rather than half-overlap overwrite.  This avoids long-sequence IK seams at
    # chunk boundaries and preserves every overlapping frame as a blend.
    starts = list(range(0, T, stride))
    accum = np.zeros_like(out_all, dtype=np.float32)
    weight_sum = np.zeros((T, 1), dtype=np.float32)
    for st in starts:
        ed = min(T, st + chunk)
        if ed - st < 4:
            continue
        base_np = motion_base[st:ed].copy()
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
            weight = np.ones((L, 1), dtype=np.float32)
            ov = min(overlap, L // 2 if L > 1 else 0)
            if ov > 1 and st > 0:
                weight[:ov, 0] *= np.linspace(0.0, 1.0, ov, dtype=np.float32)
            if ov > 1 and ed < T:
                weight[-ov:, 0] *= np.linspace(1.0, 0.0, ov, dtype=np.float32)
            # Avoid exact zero-only coverage on pathological tiny chunks.
            weight = np.maximum(weight, 1e-4)
            accum[st:ed] += best_motion.astype(np.float32) * weight
            weight_sum[st:ed] += weight
        reports.append({"start": int(st), "end": int(ed), "best_loss": float(best_loss), "contact_ratio": float(contacts[st:ed].mean())})
    valid = weight_sum[:, 0] > 1e-8
    out_all = motion_base.copy().astype(np.float32)
    out_all[valid] = accum[valid] / weight_sum[valid]
    # Re-orthogonalize all rotation channels after optimization.
    if torch is not None:
        with torch.no_grad():
            x = torch.from_numpy(out_all[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)).float()
            out_all[:, ROT6D_START:ROT6D_END] = project_rot6d_torch(x).numpy().reshape(T, -1)
    audit_before = audit_motion_np(motion, cfg)
    audit_after = audit_motion_np(out_all, cfg)
    root_delta = np.linalg.norm(out_all[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]], axis=1)
    rollback_reasons: List[str] = []
    if audit_after["foot_skate_p95_mpf"] > max(audit_before["foot_skate_p95_mpf"] * float(cfg.rollback_skate_ratio), audit_before["foot_skate_p95_mpf"] + 0.002):
        rollback_reasons.append("foot_skate_p95_worse")
    if audit_after["mean_joint_jerk_p95"] > max(audit_before["mean_joint_jerk_p95"] * float(cfg.rollback_jerk_ratio), audit_before["mean_joint_jerk_p95"] + 1e-4):
        rollback_reasons.append("joint_jerk_p95_worse")
    if audit_after["foot_penetration_min_m"] < audit_before["foot_penetration_min_m"] - float(cfg.rollback_penetration_margin_m):
        rollback_reasons.append("floor_penetration_worse")
    if root_delta.size and float(root_delta.max()) > float(cfg.rollback_root_delta_max_m):
        rollback_reasons.append("root_delta_too_large")
    rollback = bool(rollback_reasons)
    final = motion.copy() if rollback else out_all
    report = {
        "version": "v46_8_true_lower_body_ik_root_aware_sliding_anchor_weighted_chunks_strict_rollback",
        "enabled": True,
        "writes_lower_body_rot6d": True,
        "root_y_physics": root_y_report,
        "ik_target_generator": target_meta,
        "lower_body_joints": list(map(int, LOWER_BODY_JOINTS)),
        "foot_joint_ids": list(map(int, DEFAULT_FOOT_JOINTS)),
        "floor_y": float(floor_y),
        "contact_ratio": float(contacts.mean()),
        "chunks": reports,
        "chunk_stitching": {
            "mode": "weighted_accumulation",
            "chunk": int(chunk),
            "overlap": int(overlap),
            "stride": int(stride),
            "coverage_min": float(weight_sum[:, 0].min()) if weight_sum.size else 0.0,
            "coverage_p95": float(np.percentile(weight_sum[:, 0], 95)) if weight_sum.size else 0.0,
        },
        "rollback_policy": {
            "mode": "or_gated_safety_rollback",
            "skate_ratio": float(cfg.rollback_skate_ratio),
            "jerk_ratio": float(cfg.rollback_jerk_ratio),
            "penetration_margin_m": float(cfg.rollback_penetration_margin_m),
            "root_delta_max_m": float(cfg.rollback_root_delta_max_m),
        },
        "root_delta_max_m": float(root_delta.max()) if root_delta.size else 0.0,
        "root_delta_p95_m": float(np.percentile(root_delta, 95)) if root_delta.size else 0.0,
        "rollback_reasons": rollback_reasons,
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
    sem_dirs = getattr(args, "music_semantic_dirs", None)
    if sem_dirs:
        cfg.external_music_semantic_dirs = os.pathsep.join([str(x) for x in sem_dirs])
    if getattr(args, "external_music_semantic_cmd", None):
        cfg.external_music_semantic_cmd = str(args.external_music_semantic_cmd)
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
    seam_mask, seam_positions, transition_spans = transition_mask_from_concat_report(motion.shape[0], concat_report, default_width=24)
    cond = np.mean(slot_feat, axis=0).astype(np.float32)
    # Normalize cond using database stats if available.
    cond = (cond - np.asarray(db["desc_mean"], dtype=np.float32)[0]) / np.asarray(db["desc_std"], dtype=np.float32)[0]
    stage_reports = {"retrieval": retrieval_report, "concat": concat_report, "seams": seam_positions, "transition_spans": transition_spans, "transition_budget_enabled": _v46_env_bool("V46_TRANSITION_BUDGET_ENABLE", True)}
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
    b.add_argument("--manifest", default=None, help="Optional manifest.csv. When present, its source_bvh/fragment_file/label/start_frame/end_frame fields become parent metadata for source-aware Event-RAG.")
    b.add_argument("--out_db", required=True, help="Output directory, containing events.npz and events/*.npy")
    b.add_argument("--audio_dirs", nargs="*", default=None, help="Optional paired audio/music directories. Same-stem files are used for real V44 music features.")
    b.set_defaults(func=build_db)

    c = sub.add_parser("train-contrastive", help="V44 music-motion contrastive training")
    c.add_argument("--db", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--music_feature_npz", default=None)
    c.add_argument("--unpaired_audio_dirs", nargs="*", default=None, help="Unpaired target/background music directories. Used for V46.8 semantic OT pseudo-pairs when BVH has no synchronized audio.")
    c.add_argument("--audio_dirs", nargs="*", default=None, help="Alias for --unpaired_audio_dirs; kept for script compatibility.")
    c.add_argument("--music_semantic_dirs", nargs="*", default=None, help="V46.12 directories containing external classical-music semantic JSON/NPZ sidecars.")
    c.add_argument("--external_music_semantic_cmd", default=None, help="Optional command template: use {audio}, {out_json}, {out_npz}, {stem} placeholders to call an external trained music semantic model.")
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
    g.add_argument("--music_semantic_dirs", nargs="*", default=None, help="V46.12 directories containing external classical-music semantic JSON/NPZ sidecars.")
    g.add_argument("--external_music_semantic_cmd", default=None, help="Optional command template for an external trained music semantic model.")
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
