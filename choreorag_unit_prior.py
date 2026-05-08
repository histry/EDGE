"""Soft retrieved-unit prior helpers for ChoreoRAG.

Drop-in replacement for choreorag_unit_prior.py.

New in this version
-------------------
1. Explicit temporal multi-frame unit prior:
   EDGE_UNIT_PRIOR_TEMPORAL=1
   EDGE_UNIT_PRIOR_WINDOW=41

2. Feature modes needed for V10/V11 experiments:
   upper
   upper+torso / upper_torso
   arms
   torso
   lower_safe
   upper+lower_safe
   all_no_root
   rot_only

3. Safety invariant:
   - never constrains contacts
   - never constrains root X/Z
   - all_no_root may constrain root_y and rotations only

4. DCT low-pass support:
   EDGE_UNIT_PRIOR_DCT=1
   EDGE_UNIT_PRIOR_LOW_FREQ_K=4

The public functions used by generate_controlled.py are preserved:
    infer_unit_specs_from_mid_paths
    apply_unit_priors_from_specs
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip().lower()


def _joint_rot_indices(joints: Iterable[int]) -> List[int]:
    idx: List[int] = []
    for j in joints:
        j = int(j)
        idx.extend(range(ROT_START + 6 * j, ROT_START + 6 * j + 6))
    return [i for i in idx if 0 <= i < REPR_DIM]


def make_unit_prior_mask(mode: str = "upper", include_root_y: bool = True) -> np.ndarray:
    """Return [151] feature mask for retrieved-unit prior.

    Safety: contacts and root X/Z are always zeroed at the end.
    """
    mode = str(mode or "upper").strip().lower().replace(" ", "")
    mode = mode.replace("-", "_")
    mask = np.zeros((REPR_DIM,), dtype=np.float32)

    include_root_y = bool(include_root_y) and _env_bool("EDGE_UNIT_PRIOR_INCLUDE_ROOT_Y", True)

    if mode in {"upper", "safe_upper"}:
        joints = list(UPPER_JOINTS)
        if _env_bool("EDGE_UNIT_PRIOR_INCLUDE_TORSO", False):
            joints += TORSO_JOINTS
        mask[_joint_rot_indices(joints)] = 1.0
        if include_root_y:
            mask[ROOT_Y_IDX] = 1.0

    elif mode in {"upper+torso", "upper_torso", "torso+upper", "upperbody", "upper_body"}:
        mask[_joint_rot_indices(list(TORSO_JOINTS) + list(UPPER_JOINTS))] = 1.0
        if include_root_y:
            mask[ROOT_Y_IDX] = 1.0

    elif mode == "arms":
        mask[_joint_rot_indices(UPPER_JOINTS)] = 1.0

    elif mode == "torso":
        mask[_joint_rot_indices(TORSO_JOINTS)] = 1.0
        if include_root_y:
            mask[ROOT_Y_IDX] = 1.0

    elif mode in {"loco_safe", "lower_safe", "lower"}:
        mask[_joint_rot_indices(LOWER_SAFE_JOINTS)] = 1.0
        if include_root_y:
            mask[ROOT_Y_IDX] = 1.0

    elif mode in {"loco_upper", "upper_loco", "upper+lower_safe", "upper_lower_safe"}:
        joints = list(UPPER_JOINTS) + list(LOWER_SAFE_JOINTS)
        if _env_bool("EDGE_UNIT_PRIOR_INCLUDE_TORSO", True):
            joints += TORSO_JOINTS
        mask[_joint_rot_indices(joints)] = 1.0
        if include_root_y:
            mask[ROOT_Y_IDX] = 1.0

    elif mode in {"all_no_root", "allnoroot", "all_no_xz", "body_no_rootxz"}:
        # all_no_root means: all rotations + optional root_y, but never root_x/root_z/contact.
        mask[ROT_START:REPR_DIM] = 1.0
        if include_root_y:
            mask[ROOT_Y_IDX] = 1.0

    elif mode == "rot_only":
        mask[ROT_START:REPR_DIM] = 1.0

    else:
        raise ValueError(
            f"Unknown EDGE_UNIT_PRIOR_FEATURES={mode!r}. "
            "Use upper, upper+torso, arms, torso, lower_safe, upper+lower_safe, all_no_root, rot_only."
        )

    # Hard safety invariant.
    mask[CONTACT_SLICE] = 0.0
    mask[ROOT_X_IDX] = 0.0
    mask[ROOT_Z_IDX] = 0.0
    return mask.astype(np.float32)


def _as_unit_t151(arr: np.ndarray, path: str = "") -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == REPR_DIM:
        return arr.reshape(1, REPR_DIM)
    if arr.ndim != 2:
        raise ValueError(f"unit prior must be [T,151] or [151,T], got {arr.shape} from {path}")
    if arr.shape[-1] == REPR_DIM:
        return arr.astype(np.float32)
    if arr.shape[0] == REPR_DIM:
        return arr.T.astype(np.float32)
    raise ValueError(f"unit prior must be [T,151] or [151,T], got {arr.shape} from {path}")


def _load_unit(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        for key in ("unit_motion", "unit", "motion", "poses", "unit_motions"):
            if key in d:
                arr = d[key]
                break
    if np.asarray(arr).ndim == 3:
        arr = np.asarray(arr)[0]
    return _as_unit_t151(arr, path=path)


def triangular_weight(j: int, length: int, gamma: float = 1.0, floor: float = 0.0) -> float:
    if length <= 1:
        return 1.0
    center = (length - 1) / 2.0
    w = 1.0 - abs(float(j) - center) / max(center, 1.0)
    w = float(np.clip(w, 0.0, 1.0))
    w = float(w ** max(float(gamma), 1e-6))
    floor = float(np.clip(floor, 0.0, 1.0))
    return float(floor + (1.0 - floor) * w)


def _crop_unit_center(unit: np.ndarray, max_len: int) -> np.ndarray:
    unit = _as_unit_t151(unit)
    max_len = min(max(1, int(max_len)), len(unit))
    if max_len >= len(unit):
        return unit.astype(np.float32)
    center = len(unit) // 2
    start = max(0, min(len(unit) - max_len, center - max_len // 2))
    end = start + max_len
    return unit[start:end].astype(np.float32)


def _center_pose_only(unit: np.ndarray) -> np.ndarray:
    unit = _as_unit_t151(unit)
    return unit[len(unit) // 2 : len(unit) // 2 + 1].astype(np.float32)


def _select_temporal_window(unit: np.ndarray) -> np.ndarray:
    """Select temporal prior window according to env.

    If EDGE_UNIT_PRIOR_TEMPORAL=0, keep only center pose for backward-compatible
    center-pose prior behavior.  If enabled, use EDGE_UNIT_PRIOR_WINDOW, falling
    back to EDGE_UNIT_PRIOR_MAX_LEN.
    """
    unit = _as_unit_t151(unit)
    temporal = _env_bool("EDGE_UNIT_PRIOR_TEMPORAL", False)
    if not temporal:
        return _center_pose_only(unit)
    window = _env_int("EDGE_UNIT_PRIOR_WINDOW", _env_int("EDGE_UNIT_PRIOR_MAX_LEN", 45))
    return _crop_unit_center(unit, max_len=window)


def _dct_lowpass_numpy(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or len(x) <= 2:
        return x.astype(np.float32)
    k = int(max(1, min(k, len(x))))
    try:
        from scipy.fftpack import dct, idct
        coeff = dct(x, type=2, norm="ortho", axis=0)
        coeff[k:, :] = 0.0
        y = idct(coeff, type=2, norm="ortho", axis=0)
        return y.astype(np.float32)
    except Exception:
        freq = np.fft.rfft(x, axis=0)
        freq[k:, :] = 0.0
        y = np.fft.irfft(freq, n=len(x), axis=0)
        return y.astype(np.float32)


def _maybe_frequency_decouple_unit(unit: np.ndarray, prior_mask: np.ndarray) -> np.ndarray:
    if not _env_bool("EDGE_UNIT_PRIOR_DCT", False):
        return unit.astype(np.float32)
    k = _env_int("EDGE_UNIT_PRIOR_LOW_FREQ_K", 4)
    out = unit.astype(np.float32).copy()
    feat_idx = np.where(prior_mask > 0)[0]
    if len(feat_idx) == 0:
        return out
    out[:, feat_idx] = _dct_lowpass_numpy(out[:, feat_idx], k=k)
    if _env_bool("EDGE_UNIT_PRIOR_VERBOSE", False):
        print(f"   DCT low-pass unit prior: K={k}, selected_features={len(feat_idx)}")
    return out.astype(np.float32)


def add_unit_prior(
    value: np.ndarray,
    mask: np.ndarray,
    unit_motion: np.ndarray,
    center_frame: int,
    strength: float = 0.012,
    feature_mode: str = "upper",
    max_len: int = 45,
    decay_gamma: float = 1.0,
    decay_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply retrieved-unit soft prior to value/mask arrays.

    value/mask are [T,151] in normalized pose space.  The unit is aligned so its
    center frame lands on center_frame.  Only features selected by feature_mode
    are written.  Root X/Z and contact channels are never constrained.
    """
    value = np.asarray(value, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    unit = _as_unit_t151(unit_motion)

    if value.ndim != 2 or mask.ndim != 2 or value.shape != mask.shape or value.shape[1] != REPR_DIM:
        raise ValueError(f"value/mask must both be [T,151], got value={value.shape}, mask={mask.shape}")

    # EDGE_UNIT_PRIOR_TEMPORAL controls center-pose vs temporal prior.
    if _env_bool("EDGE_UNIT_PRIOR_TEMPORAL", False):
        unit = _select_temporal_window(unit)
    else:
        # Backward compatibility: max_len is ignored when temporal mode is off.
        unit = _center_pose_only(unit)

    prior_mask = make_unit_prior_mask(feature_mode)
    unit = _maybe_frequency_decouple_unit(unit, prior_mask)

    center_frame = int(center_frame)
    half = len(unit) // 2
    total_frames = value.shape[0]
    strength = float(max(0.0, strength))
    if strength <= 0.0:
        return value, mask

    combine = _env_str("EDGE_UNIT_PRIOR_COMBINE", "max")
    applied_frames = 0
    applied_features = int(np.sum(prior_mask > 0))

    for j in range(len(unit)):
        frame = center_frame - half + j
        if frame < 0 or frame >= total_frames:
            continue
        local_strength = strength * triangular_weight(j, len(unit), decay_gamma, decay_floor)
        if local_strength <= 0.0:
            continue
        new_mask = prior_mask * float(local_strength)
        if combine == "add":
            update = new_mask > 0
            value[frame, update] = unit[j, update]
            mask[frame, update] = np.maximum(mask[frame, update], new_mask[update])
        else:
            update = new_mask > mask[frame]
            if np.any(update):
                value[frame, update] = unit[j, update]
                mask[frame, update] = new_mask[update]
        applied_frames += int(np.any(new_mask > 0))

    if _env_bool("EDGE_UNIT_PRIOR_VERBOSE", False):
        print(
            f"   unit_prior center={center_frame} len={len(unit)} temporal={_env_bool('EDGE_UNIT_PRIOR_TEMPORAL', False)} "
            f"features={feature_mode}/{applied_features} applied_frames={applied_frames} strength={strength}"
        )
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
            # V10 planner writes <prefix>_mid01_f40.npy and <prefix>_mid01_f40_unit.npy.
            candidates.append(p.with_name(p.stem.replace("_pose", "") + "_unit.npy"))
        candidates.append(Path(str(pose_path) + "_unit.npy"))
        for unit_path in candidates:
            if unit_path.is_file():
                specs.append({"unit_path": str(unit_path), "frame": int(frame)})
                break
    return specs


def apply_unit_priors_from_specs(value: np.ndarray, mask: np.ndarray, specs: Iterable[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    if not _env_bool("EDGE_UNIT_SOFT_PRIOR", False):
        return value, mask

    strength = _env_float("EDGE_UNIT_PRIOR_STRENGTH", 0.012)
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
            value=value,
            mask=mask,
            unit_motion=unit,
            center_frame=int(frame),
            strength=strength,
            feature_mode=feature_mode,
            max_len=max_len,
            decay_gamma=decay_gamma,
            decay_floor=decay_floor,
        )
        applied += 1

    if applied:
        dct_msg = ""
        if _env_bool("EDGE_UNIT_PRIOR_DCT", False):
            dct_msg = f", dct_lowpass_k={_env_int('EDGE_UNIT_PRIOR_LOW_FREQ_K', 4)}"
        print(
            "✅ ChoreoRAG unit soft prior applied: "
            f"count={applied}, strength={strength}, features={feature_mode}, "
            f"temporal={_env_bool('EDGE_UNIT_PRIOR_TEMPORAL', False)}, "
            f"window={_env_int('EDGE_UNIT_PRIOR_WINDOW', _env_int('EDGE_UNIT_PRIOR_MAX_LEN', 45))}{dct_msg}"
        )
    else:
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
