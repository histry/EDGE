"""Soft retrieved-unit prior helpers for PACE-ChoreoRAG.

Drop-in replacement for choreorag_unit_prior.py.

Disabled by default.  Activation examples:
    export EDGE_UNIT_SOFT_PRIOR=1
    export EDGE_UNIT_PRIOR_FEATURES=upper
    export EDGE_UNIT_PRIOR_STRENGTH=0.08

Supported feature modes:
    upper       upper body + optional torso, safest for temporal smoothing
    loco_safe   weak lower-body rotations only; no contacts/root XZ
    loco_upper  upper + loco_safe; recommended weak setting 0.025
    rot_only    all rotations, not recommended except diagnostics

This module never touches contact channels or root X/Z by default.  It writes a
weak temporal constraint into the existing EDGE keyframe value/mask tensors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT_START = 7
REPR_DIM = 151

TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
LOWER_SAFE_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


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
    v = str(os.environ.get(name, "")).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _joint_rot_indices(joints: Iterable[int]) -> List[int]:
    idx: List[int] = []
    for j in joints:
        j = int(j)
        idx.extend(range(ROT_START + 6 * j, ROT_START + 6 * j + 6))
    return [i for i in idx if 0 <= i < REPR_DIM]


def make_unit_prior_mask(mode: str = "upper", include_root_y: bool = True) -> np.ndarray:
    mode = str(mode or "upper").strip().lower()
    mask = np.zeros((REPR_DIM,), dtype=np.float32)
    if include_root_y and _env_bool("EDGE_UNIT_PRIOR_INCLUDE_ROOT_Y", True):
        mask[ROOT_Y_IDX] = 1.0

    if mode in {"upper", "upper_torso", "safe_upper"}:
        joints = list(UPPER_JOINTS)
        if _env_bool("EDGE_UNIT_PRIOR_INCLUDE_TORSO", True):
            joints += TORSO_JOINTS
        mask[_joint_rot_indices(joints)] = 1.0

    elif mode == "arms":
        mask[_joint_rot_indices(UPPER_JOINTS)] = 1.0

    elif mode in {"loco_safe", "lower_safe"}:
        # Weak lower-body rotation prior for weight shift / stepping.
        # It excludes contacts and root X/Z, so it is safer than all-body prior.
        mask[_joint_rot_indices(LOWER_SAFE_JOINTS)] = 1.0

    elif mode in {"loco_upper", "upper_loco"}:
        joints = list(UPPER_JOINTS) + list(LOWER_SAFE_JOINTS)
        if _env_bool("EDGE_UNIT_PRIOR_INCLUDE_TORSO", True):
            joints += TORSO_JOINTS
        mask[_joint_rot_indices(joints)] = 1.0

    elif mode == "rot_only":
        mask[ROT_START:REPR_DIM] = 1.0

    else:
        raise ValueError(f"Unknown EDGE_UNIT_PRIOR_FEATURES={mode!r}")

    # Safety: never constrain contacts or root X/Z through this unit prior.
    mask[CONTACT_SLICE] = 0.0
    mask[ROOT_X_IDX] = 0.0
    mask[ROOT_Z_IDX] = 0.0
    return mask.astype(np.float32)


def _load_unit(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        for key in ("unit_motion", "unit", "motion", "poses"):
            if key in d:
                arr = d[key]
                break
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != REPR_DIM:
        raise ValueError(f"unit prior must be [T,151], got {arr.shape} from {path}")
    return arr.astype(np.float32)


def triangular_weight(j: int, length: int, gamma: float = 1.0, floor: float = 0.0) -> float:
    if length <= 1:
        return 1.0
    center = (length - 1) / 2.0
    w = 1.0 - abs(float(j) - center) / max(center, 1.0)
    w = max(0.0, min(1.0, w))
    w = float(w ** max(float(gamma), 1e-6))
    floor = float(np.clip(floor, 0.0, 1.0))
    return float(floor + (1.0 - floor) * w)


def _crop_unit_center(unit: np.ndarray, max_len: int) -> np.ndarray:
    max_len = min(max(1, int(max_len)), len(unit))
    if max_len >= len(unit):
        return unit.astype(np.float32)
    c = len(unit) // 2
    s = max(0, c - max_len // 2)
    e = min(len(unit), s + max_len)
    s = max(0, e - max_len)
    return unit[s:e].astype(np.float32)


def add_unit_prior(
    value: np.ndarray,
    mask: np.ndarray,
    unit_motion: np.ndarray,
    center_frame: int,
    strength: float = 0.06,
    feature_mode: str = "upper",
    max_len: int = 45,
    decay_gamma: float = 1.0,
    decay_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend a weak retrieved-unit temporal prior into keyframe constraints."""
    value = np.asarray(value, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    unit = np.asarray(unit_motion, dtype=np.float32)
    if unit.ndim != 2 or unit.shape[1] != REPR_DIM:
        raise ValueError(f"unit_motion must be [T,151], got {unit.shape}")
    if value.ndim != 2 or mask.ndim != 2 or value.shape != mask.shape or value.shape[1] != REPR_DIM:
        raise ValueError(f"value/mask must both be [T,151], got value={value.shape}, mask={mask.shape}")

    unit = _crop_unit_center(unit, max_len)
    prior_mask = make_unit_prior_mask(feature_mode)
    center_frame = int(center_frame)
    half = len(unit) // 2
    T = value.shape[0]
    strength = float(max(0.0, strength))
    if strength <= 0.0:
        return value, mask

    combine = os.environ.get("EDGE_UNIT_PRIOR_COMBINE", "max").strip().lower()
    applied_frames = 0
    for j in range(len(unit)):
        f = center_frame - half + j
        if f < 0 or f >= T:
            continue
        local = strength * triangular_weight(j, len(unit), decay_gamma, decay_floor)
        if local <= 0.0:
            continue
        new_mask = prior_mask * float(local)
        if combine == "add":
            update = new_mask > 0
            value[f, update] = unit[j, update]
            mask[f, update] = np.maximum(mask[f, update], new_mask[update])
        else:
            update = new_mask > mask[f]
            if np.any(update):
                value[f, update] = unit[j, update]
                mask[f, update] = new_mask[update]
        applied_frames += int(np.any(new_mask > 0))
    if _env_bool("EDGE_UNIT_PRIOR_VERBOSE", False):
        print(f"   unit_prior frame={center_frame} len={len(unit)} applied_frames={applied_frames}")
    return value, mask


def unit_specs_from_saved_records(saved_records: Iterable[Dict]) -> List[Dict[str, object]]:
    specs = []
    for rec in saved_records or []:
        unit_path = rec.get("unit_path") or rec.get("unit_motion_path")
        frame = rec.get("frame")
        if unit_path and frame is not None:
            specs.append({"unit_path": str(unit_path), "frame": int(frame)})
    return specs


def infer_unit_specs_from_mid_paths(mid_paths: Iterable[str], mid_frames: Iterable[int]) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for pose_path, frame in zip(mid_paths or [], mid_frames or []):
        p = Path(str(pose_path))
        candidates = []
        if p.suffix == ".npy":
            candidates.append(p.with_name(p.stem + "_unit.npy"))
        candidates.append(Path(str(pose_path) + "_unit.npy"))
        for up in candidates:
            if up.is_file():
                specs.append({"unit_path": str(up), "frame": int(frame)})
                break
    return specs


def apply_unit_priors_from_specs(value: np.ndarray, mask: np.ndarray, specs: Iterable[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    if not _env_bool("EDGE_UNIT_SOFT_PRIOR", False):
        return value, mask
    strength = _env_float("EDGE_UNIT_PRIOR_STRENGTH", 0.06)
    feature_mode = os.environ.get("EDGE_UNIT_PRIOR_FEATURES", "upper")
    max_len = _env_int("EDGE_UNIT_PRIOR_MAX_LEN", 45)
    decay_gamma = _env_float("EDGE_UNIT_PRIOR_DECAY_GAMMA", 1.0)
    decay_floor = _env_float("EDGE_UNIT_PRIOR_DECAY_FLOOR", 0.0)

    applied = 0
    missing = 0
    for spec in specs or []:
        unit_path = str(spec.get("unit_path", ""))
        frame = spec.get("frame", None)
        if not unit_path or frame is None or not Path(unit_path).is_file():
            missing += 1
            continue
        unit = _load_unit(unit_path)
        value, mask = add_unit_prior(
            value,
            mask,
            unit,
            center_frame=int(frame),
            strength=strength,
            feature_mode=feature_mode,
            max_len=max_len,
            decay_gamma=decay_gamma,
            decay_floor=decay_floor,
        )
        applied += 1
    if applied:
        print(f"✅ ChoreoRAG unit soft prior applied: count={applied}, strength={strength}, features={feature_mode}")
    elif _env_bool("EDGE_UNIT_SOFT_PRIOR", False):
        print(f"⚠️ EDGE_UNIT_SOFT_PRIOR=1 but no valid unit prior was applied; missing={missing}")
    return value, mask


def specs_to_json(specs: Iterable[Dict[str, object]]) -> str:
    return json.dumps(list(specs or []), ensure_ascii=False)


def specs_from_json(text: str) -> List[Dict[str, object]]:
    if not text:
        return []
    try:
        data = json.loads(text)
        return list(data) if isinstance(data, list) else []
    except Exception:
        return []
