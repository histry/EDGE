#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V20 shared motion utilities for EDGE-Dunhuang.

This file is intentionally numpy-first so it can be reused by database builders,
schedulers, evaluators and dataset builders without constructing EDGE.

151D contract:
  [0:4] contacts
  [4:7] root xyz
  [7:151] 24 joints x 6D rotation
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROOT = slice(4, 7)
ROT = slice(7, 151)
MOTION_DIM = 151

UPPER_JOINTS = list(range(14, 24))
TORSO_JOINTS = list(range(8, 14))
LOWER_JOINTS = list(range(0, 8))
ALL_JOINTS = list(range(24))

EVENT_TYPES = [
    "calm_flow",
    "high_tension",
    "build_up",
    "release",
    "support_shift",
    "turn_like",
    "arm_flourish",
    "pose_hold",
    "neutral_flow",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def moving_average(x: np.ndarray, k: int = 7) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 0 or len(x) == 0:
        return x
    k = int(max(1, k))
    if k <= 1 or len(x) <= 2:
        return x.astype(np.float32)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    if x.ndim == 1:
        xp = np.pad(x, (pad, pad), mode="edge")
        return np.convolve(xp, np.ones(k, dtype=np.float32) / k, mode="valid").astype(np.float32)
    xp = np.pad(x, ((pad, pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    out = np.stack([xp[i:i + k].mean(axis=0) for i in range(len(x))], axis=0)
    return out.astype(np.float32)


def minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def resize_feature(feat: np.ndarray, length: int) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat[:, None]
    if len(feat) == length:
        return feat.astype(np.float32)
    if len(feat) == 0:
        return np.zeros((length, 1), dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(feat), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, int(length), dtype=np.float32)
    out = np.stack([np.interp(dst, src, feat[:, i]) for i in range(feat.shape[1])], axis=-1)
    return out.astype(np.float32)


def canonical_resample_motion(motion: np.ndarray, length: int = 48) -> np.ndarray:
    motion = validate_motion(motion)
    return resize_feature(motion, length).astype(np.float32)


def validate_motion(x: np.ndarray, path: str | Path = "") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 1 and x.size == MOTION_DIM:
        x = x[None]
    if x.ndim != 2 or x.shape[-1] != MOTION_DIM:
        raise ValueError(f"Expected [T,{MOTION_DIM}] motion, got {x.shape} from {path}")
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)



def _axis_angle_to_rot6d_numpy(q: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotations to PyTorch3D-compatible 6D rotations.

    Input can be [T,72] or [T,24,3]. Output is [T,144].
    PyTorch3D's matrix_to_rotation_6d keeps the first two rows of the
    rotation matrix in row-major order. This matches dataset.quaternion.ax_to_6v.
    """
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 2 and q.shape[-1] == 72:
        q = q.reshape(q.shape[0], 24, 3)
    if q.ndim != 3 or q.shape[1:] != (24, 3):
        raise ValueError(f"axis-angle q should be [T,72] or [T,24,3], got {q.shape}")

    T = q.shape[0]
    aa = q.reshape(-1, 3).astype(np.float32)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.maximum(theta, 1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros_like(x)
    K = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], axis=-1).reshape(-1, 3, 3).astype(np.float32)
    I = np.eye(3, dtype=np.float32)[None]
    st = np.sin(theta).reshape(-1, 1, 1).astype(np.float32)
    ct = np.cos(theta).reshape(-1, 1, 1).astype(np.float32)
    R = I + st * K + (1.0 - ct) * (K @ K)

    # Small-angle fallback: axis is arbitrary when theta≈0; exact identity is safer.
    small = (theta.reshape(-1) < 1e-8)
    if np.any(small):
        R[small] = I

    rot6 = R[:, :2, :].reshape(T, 24, 6).reshape(T, 144)
    return rot6.astype(np.float32)


def _motion_from_pos_q_dict(obj: Any) -> Optional[np.ndarray]:
    """Convert local Dunhuang processed dict {'pos':[T,3], 'q':[T,72]} to 151D.

    Your data/dunhuang_bvh/processed/*.pkl files store root position and SMPL
    axis-angle rotations instead of the final EDGE 151D representation:
      pos: [T,3]
      q:   [T,72] = 24 joints x axis-angle(3)

    EDGE 151D is:
      [0:4] zero/proxy contacts, [4:7] pos, [7:151] 24 joints x 6D rotation.
    Contacts are filled with zeros here because the dynamic event DB mainly uses
    root/rotation curves; a later support estimator can overwrite them if needed.
    """
    if not isinstance(obj, dict) or "pos" not in obj or "q" not in obj:
        return None
    try:
        pos = np.asarray(obj["pos"], dtype=np.float32)
        q = np.asarray(obj["q"], dtype=np.float32)
    except Exception:
        return None
    if pos.ndim != 2 or pos.shape[-1] != 3:
        return None
    if q.ndim == 2 and q.shape[-1] == 144:
        rot6 = q.astype(np.float32)
    elif q.ndim == 3 and q.shape[1:] == (24, 6):
        rot6 = q.reshape(q.shape[0], 144).astype(np.float32)
    elif (q.ndim == 2 and q.shape[-1] == 72) or (q.ndim == 3 and q.shape[1:] == (24, 3)):
        rot6 = _axis_angle_to_rot6d_numpy(q)
    else:
        return None
    T = int(min(len(pos), len(rot6)))
    if T <= 0:
        return None
    contacts = np.zeros((T, 4), dtype=np.float32)
    motion = np.concatenate([contacts, pos[:T].astype(np.float32), rot6[:T].astype(np.float32)], axis=-1)
    return motion.astype(np.float32)


def _array_to_motion(arr: Any, path: str | Path = "") -> Optional[np.ndarray]:
    """Return a [T,151] motion if arr is numerically compatible, else None."""
    try:
        a = np.asarray(arr)
    except Exception:
        return None
    if a.dtype == object:
        return None
    if a.dtype.kind not in {"b", "i", "u", "f", "c"}:
        return None

    # Common EDGE / Dunhuang contracts.
    if a.ndim == 1 and a.size == MOTION_DIM:
        return a.reshape(1, MOTION_DIM).astype(np.float32)
    if a.ndim == 2:
        if a.shape[-1] == MOTION_DIM:
            return a.astype(np.float32)
        # Occasionally arrays are saved transposed as [151,T].
        if a.shape[0] == MOTION_DIM and a.shape[1] > 1:
            return a.T.astype(np.float32)
    if a.ndim == 3 and a.shape[-1] == MOTION_DIM:
        # [1,T,151] or [N,T,151]. For [N,T,151], flatten with no synthetic
        # interpolation; this keeps all frames available for dynamic segmentation.
        if a.shape[0] == 1:
            return a[0].astype(np.float32)
        return a.reshape(-1, MOTION_DIM).astype(np.float32)
    if a.ndim > 3 and a.shape[-1] == MOTION_DIM:
        return a.reshape(-1, MOTION_DIM).astype(np.float32)
    return None


def _short_structure(obj: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Compact structure summary for debugging unsupported pickle layouts."""
    if depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= 12:
                out["..."] = f"+{len(obj)-12} keys"
                break
            out[str(k)] = _short_structure(v, depth + 1, max_depth)
        return out
    if isinstance(obj, (list, tuple)):
        return [f"{type(obj).__name__}[len={len(obj)}]", _short_structure(obj[0], depth + 1, max_depth) if obj else "empty"]
    if isinstance(obj, np.ndarray):
        return f"ndarray shape={obj.shape} dtype={obj.dtype}"
    return type(obj).__name__


def _find_motion_recursive(obj: Any, path: str | Path = "", depth: int = 0, max_depth: int = 8) -> Optional[np.ndarray]:
    """Recursively find or construct a [T,151] motion in nested data."""
    if depth > max_depth:
        return None

    # Local Dunhuang processed files often store {"pos": [T,3], "q": [T,72]}.
    # Convert them to EDGE 151D before generic recursive search.
    m = _motion_from_pos_q_dict(obj)
    if m is not None:
        return m

    m = _array_to_motion(obj, path)
    if m is not None:
        return m

    # 0-d object numpy array containing a dict/list.
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        try:
            if obj.ndim == 0:
                return _find_motion_recursive(obj.item(), path, depth + 1, max_depth)
            for x in obj.flat:
                m = _find_motion_recursive(x, path, depth + 1, max_depth)
                if m is not None:
                    return m
        except Exception:
            return None

    if isinstance(obj, dict):
        # Search known motion keys first. Some processed Dunhuang files store
        # obj["motion"] as another dict, so this must recurse instead of directly
        # casting obj["motion"] to float.
        priority = [
            "motion", "motion_151", "poses", "unit_motions_physical",
            "arr_0", "target", "noisy", "data", "motion_data", "sequence",
            "samples", "x", "body", "smpl_motion",
        ]
        for k in priority:
            if k in obj:
                m = _find_motion_recursive(obj[k], path, depth + 1, max_depth)
                if m is not None:
                    return m
        for k, v in obj.items():
            if k in priority:
                continue
            m = _find_motion_recursive(v, path, depth + 1, max_depth)
            if m is not None:
                return m
        return None

    if isinstance(obj, (list, tuple)):
        for v in obj:
            m = _find_motion_recursive(v, path, depth + 1, max_depth)
            if m is not None:
                return m

    return None


def load_motion_any(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a [T,151] motion from npy/npz/pkl, including nested dict pkl files.

    This is intentionally defensive because local Dunhuang processed .pkl files
    can store keys such as ``motion`` as nested dictionaries. The old loader tried
    to cast that dict directly to float, causing:
        float() argument must be a string or a number, not 'dict'
    """
    path = Path(path)
    suffix = path.suffix.lower()
    meta: Dict[str, Any] = {"path": str(path), "stem": path.stem, "suffix": suffix}

    if suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
    elif suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        obj = {k: z[k] for k in z.files}
        meta["npz_keys"] = list(z.files)
    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            meta["pkl_keys"] = [str(k) for k in obj.keys()]
    else:
        raise ValueError(f"Unsupported motion file: {path}")

    motion = _find_motion_recursive(obj, path)
    if motion is None:
        raise ValueError(
            f"No numeric [T,{MOTION_DIM}] motion found in {path}. "
            f"Structure: {_short_structure(obj)}"
        )
    return validate_motion(motion, path), meta

def iter_motion_files(input_dir: str | Path, exts: Sequence[str] = (".npy", ".npz", ".pkl", ".pickle")) -> List[Path]:
    root = Path(input_dir)
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(set(files))


def localize_root(motion: np.ndarray, keep_y: bool = True) -> np.ndarray:
    out = validate_motion(motion).copy()
    if len(out):
        out[:, ROOT_X] -= out[0, ROOT_X]
        out[:, ROOT_Z] -= out[0, ROOT_Z]
        if not keep_y:
            out[:, ROOT_Y] -= out[0, ROOT_Y]
    return out.astype(np.float32)


def rot_view(motion: np.ndarray) -> np.ndarray:
    motion = validate_motion(motion)
    return motion[:, ROT].reshape(len(motion), 24, 6)


def rot_energy(motion: np.ndarray, joints: Sequence[int] = ALL_JOINTS) -> np.ndarray:
    r = rot_view(motion)
    if len(r) <= 1:
        return np.zeros((len(r),), dtype=np.float32)
    d = r[1:] - r[:-1]
    e = np.linalg.norm(d[:, list(joints)].reshape(len(d), -1), axis=-1).astype(np.float32)
    # align to frame length by duplicating first value
    return np.concatenate([e[:1], e], axis=0).astype(np.float32)


def root_y_curve(motion: np.ndarray) -> np.ndarray:
    return validate_motion(motion)[:, ROOT_Y].astype(np.float32)


def contact_switch_curve(motion: np.ndarray) -> np.ndarray:
    m = validate_motion(motion)
    c = np.clip(m[:, CONTACT], 0.0, 1.0)
    if len(c) <= 1:
        return np.zeros((len(c),), dtype=np.float32)
    d = np.abs(c[1:] - c[:-1]).mean(axis=-1).astype(np.float32)
    return np.concatenate([d[:1], d], axis=0).astype(np.float32)


def root_path_radius(motion: np.ndarray) -> float:
    m = validate_motion(motion)
    if len(m) == 0:
        return 0.0
    xz = m[:, [ROOT_X, ROOT_Z]] - m[:1, [ROOT_X, ROOT_Z]]
    return float(np.linalg.norm(xz, axis=-1).max())


def jerk_score(motion: np.ndarray) -> float:
    m = validate_motion(motion)
    if len(m) < 4:
        return 0.0
    x = m[:, ROT]
    j = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
    return float(np.linalg.norm(j, axis=-1).mean())


def compute_motion_curves(motion: np.ndarray, smooth: int = 7) -> Dict[str, np.ndarray]:
    m = validate_motion(motion)
    upper = moving_average(rot_energy(m, UPPER_JOINTS), smooth)
    torso = moving_average(rot_energy(m, TORSO_JOINTS), smooth)
    lower = moving_average(rot_energy(m, LOWER_JOINTS), smooth)
    full = moving_average(rot_energy(m, ALL_JOINTS), smooth)
    ry = root_y_curve(m)
    ry_vel = np.zeros_like(ry)
    if len(ry) > 1:
        ry_vel[1:] = ry[1:] - ry[:-1]
        ry_vel[0] = ry_vel[1]
    contact_sw = moving_average(contact_switch_curve(m), smooth)
    acc = np.zeros_like(full)
    if len(full) > 2:
        acc[1:-1] = full[2:] - 2 * full[1:-1] + full[:-2]
    style_tension = moving_average(0.55 * minmax(upper) + 0.35 * minmax(torso) + 0.10 * minmax(np.abs(acc)), smooth)
    return {
        "upper": upper.astype(np.float32),
        "torso": torso.astype(np.float32),
        "lower": lower.astype(np.float32),
        "full": full.astype(np.float32),
        "root_y": ry.astype(np.float32),
        "root_y_vel": ry_vel.astype(np.float32),
        "contact_switch": contact_sw.astype(np.float32),
        "acc": acc.astype(np.float32),
        "style_tension": style_tension.astype(np.float32),
    }


def zero_crossings(x: np.ndarray) -> List[int]:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return []
    s = np.sign(x)
    z = np.where((s[1:] * s[:-1]) <= 0)[0] + 1
    return [int(i) for i in z]


def local_minima(x: np.ndarray, radius: int = 3) -> List[int]:
    x = np.asarray(x, dtype=np.float32)
    out: List[int] = []
    r = int(max(1, radius))
    for i in range(r, len(x) - r):
        w = x[i - r:i + r + 1]
        if x[i] <= w.min() + 1e-8:
            out.append(int(i))
    return out


def local_maxima(x: np.ndarray, radius: int = 3) -> List[int]:
    x = np.asarray(x, dtype=np.float32)
    out: List[int] = []
    r = int(max(1, radius))
    for i in range(r, len(x) - r):
        w = x[i - r:i + r + 1]
        if x[i] >= w.max() - 1e-8:
            out.append(int(i))
    return out


def classify_motion_event(desc: Dict[str, float]) -> str:
    upper = float(desc.get("upper_activity", 0.0))
    torso = float(desc.get("torso_activity", 0.0))
    lower = float(desc.get("lower_activity", 0.0))
    tension = float(desc.get("style_tension", 0.0))
    smoothness = float(desc.get("smoothness", 0.0))
    contact_sw = float(desc.get("contact_switch", 0.0))
    root_y_range = float(desc.get("root_y_range", 0.0))
    if contact_sw > 0.18 or lower > 0.55:
        return "support_shift"
    if upper > 0.60 and tension > 0.55:
        return "arm_flourish"
    if tension > 0.65 or (upper > 0.55 and torso > 0.45):
        return "high_tension"
    if upper > 0.45 and torso > 0.35 and smoothness > 0.45:
        return "build_up"
    if smoothness > 0.65 and tension < 0.38:
        return "calm_flow"
    if root_y_range < 0.015 and upper < 0.25 and torso < 0.20:
        return "pose_hold"
    if tension < 0.35 and smoothness > 0.45:
        return "release"
    return "neutral_flow"


def describe_motion_event(motion: np.ndarray) -> Dict[str, float | str]:
    m = validate_motion(motion)
    c = compute_motion_curves(m, smooth=5)
    full = c["full"]
    upper = c["upper"]
    torso = c["torso"]
    lower = c["lower"]
    # Per-event minmax-free raw descriptors; scheduler will normalize across database too.
    desc: Dict[str, float | str] = {
        "length": int(len(m)),
        "upper_activity": float(np.mean(upper)),
        "torso_activity": float(np.mean(torso)),
        "lower_activity": float(np.mean(lower)),
        "full_activity": float(np.mean(full)),
        "entry_activity": float(np.mean(full[:min(6, len(full))])) if len(full) else 0.0,
        "exit_activity": float(np.mean(full[max(0, len(full)-6):])) if len(full) else 0.0,
        "activity_peak": float(np.max(full)) if len(full) else 0.0,
        "style_tension": float(np.mean(c["style_tension"])),
        "contact_switch": float(np.mean(c["contact_switch"])),
        "root_y_range": float(np.max(c["root_y"]) - np.min(c["root_y"])) if len(m) else 0.0,
        "root_radius": root_path_radius(m),
        "jerk": jerk_score(m),
    }
    variance = float(np.var(full)) if len(full) else 0.0
    desc["smoothness"] = float(1.0 / (1.0 + variance))
    desc["safety_score"] = float(max(0.0, 1.0 - 3.0 * desc["root_radius"] - 0.015 * desc["jerk"]))
    desc["quality_score"] = float(0.40 * desc["smoothness"] + 0.35 * desc["safety_score"] + 0.25 * min(1.0, desc["style_tension"] * 8.0))
    desc["event_type"] = classify_motion_event(desc)  # type: ignore[arg-type]
    return desc


def pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != MOTION_DIM or b.size != MOTION_DIM:
        return float(np.linalg.norm(a - b))
    rot = float(np.linalg.norm(a[ROT] - b[ROT]) / math.sqrt(144.0))
    root_y = float(abs(a[ROOT_Y] - b[ROOT_Y]))
    contact = float(np.linalg.norm(np.clip(a[CONTACT], 0, 1) - np.clip(b[CONTACT], 0, 1)))
    return rot + 0.3 * root_y + 0.1 * contact


def velocity_at(motion: np.ndarray, at_exit: bool = True, width: int = 3) -> np.ndarray:
    m = validate_motion(motion)
    if len(m) < 2:
        return np.zeros((MOTION_DIM,), dtype=np.float32)
    w = int(max(1, min(width, len(m) - 1)))
    if at_exit:
        v = (m[-1] - m[-1 - w]) / float(w)
    else:
        v = (m[w] - m[0]) / float(w)
    return v.astype(np.float32)


def linear_transition(exit_pose: np.ndarray, entry_pose: np.ndarray, k: int) -> np.ndarray:
    k = int(max(0, k))
    if k == 0:
        return np.zeros((0, MOTION_DIM), dtype=np.float32)
    a = np.asarray(exit_pose, dtype=np.float32).reshape(1, MOTION_DIM)
    b = np.asarray(entry_pose, dtype=np.float32).reshape(1, MOTION_DIM)
    weights = np.linspace(0.0, 1.0, k + 2, dtype=np.float32)[1:-1, None]
    out = (1.0 - weights) * a + weights * b
    out[:, CONTACT] = (out[:, CONTACT] > 0.5).astype(np.float32)
    # For in-place choreography, root X/Z are always local.
    out[:, ROOT_X] = 0.0
    out[:, ROOT_Z] = 0.0
    return out.astype(np.float32)


def fit_or_trim_to_length(motion: np.ndarray, num_frames: int) -> np.ndarray:
    m = validate_motion(motion)
    if len(m) == num_frames:
        return m[None].astype(np.float32)
    if len(m) > num_frames:
        return m[:num_frames][None].astype(np.float32)
    if len(m) == 0:
        return np.zeros((1, num_frames, MOTION_DIM), dtype=np.float32)
    pad = np.repeat(m[-1:], num_frames - len(m), axis=0)
    return np.concatenate([m, pad], axis=0)[None].astype(np.float32)


def save_pkl(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")
