"""Formal-safe soft retrieved-unit prior helpers for ChoreoRAG.

Drop-in replacement for ``choreorag_unit_prior.py``.

Why this version exists
-----------------------
``generate_controlled.py`` historically catches ``Exception`` around unit-prior
application and only prints a warning.  That is unsafe for formal experiments:
``EDGE_UNIT_PRIOR_REQUIRED=1`` must stop generation if no unit prior is actually
applied.  This module therefore raises ``FormalUnitPriorContractError`` (a
``BaseException`` subclass) for required/formal contract failures so legacy
``except Exception`` blocks cannot silently downgrade the failure to a warning.

Public API preserved
--------------------
    infer_unit_specs_from_mid_paths
    apply_unit_priors_from_specs
    get_last_unit_prior_report
    specs_to_json
    specs_from_json

Formal settings
---------------
    EDGE_RUN_MODE=formal
    EDGE_UNIT_SOFT_PRIOR=1
    EDGE_UNIT_PRIOR_REQUIRED=1
    EDGE_UNIT_PRIOR_TEMPORAL=1
    EDGE_UNIT_PRIOR_DCT=1
    EDGE_UNIT_PRIOR_LOW_FREQ_K=4
    EDGE_UNIT_PRIOR_FEATURES=upper+torso
    EDGE_UNIT_PRIOR_STRENGTH=0.006
    EDGE_UNIT_PRIOR_REPORT_JSON=output/.../unit_prior_report.json
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

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}
_LAST_UNIT_PRIOR_REPORT: Dict[str, object] = {}


class FormalUnitPriorContractError(BaseException):
    """Hard failure for formal/required unit-prior contract violations.

    This intentionally derives from BaseException, not Exception, so older
    wrappers that contain broad ``except Exception`` blocks cannot swallow the
    failure and continue a formally invalid run.
    """


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return bool(default)


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


def _formal() -> bool:
    return _env_str("EDGE_RUN_MODE", "").lower() == "formal"


def _required() -> bool:
    return _env_bool("EDGE_UNIT_PRIOR_REQUIRED", _formal())


def _split_csv(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def _write_report_if_requested(report: Dict[str, object]) -> None:
    path = _env_str("EDGE_UNIT_PRIOR_REPORT_JSON", "")
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ ChoreoRAG unit prior report saved: {p}")
    except Exception as exc:
        print(f"⚠️ failed to write ChoreoRAG unit prior report: {exc}")


def _contract_fail(message: str, report: Dict[str, object] | None = None) -> None:
    if report is not None:
        report["error"] = message
        global _LAST_UNIT_PRIOR_REPORT
        _LAST_UNIT_PRIOR_REPORT = report
        _write_report_if_requested(report)
    if _required() or _formal():
        raise FormalUnitPriorContractError("CRITICAL: " + message)
    raise RuntimeError(message)


def _joint_rot_indices(joints: Iterable[int]) -> List[int]:
    idx: List[int] = []
    for j in joints:
        j = int(j)
        idx.extend(range(ROT_START + 6 * j, ROT_START + 6 * j + 6))
    return [i for i in idx if 0 <= i < REPR_DIM]


def make_unit_prior_mask(mode: str = "upper", include_root_y: bool = True) -> np.ndarray:
    """Return a [151] mask for retrieved-unit prior.

    Contacts and root X/Z are hard-disabled.  This is not configurable because
    the unit prior must not fake contact labels or trajectory following.
    """
    mode = str(mode or "upper").strip().lower().replace(" ", "").replace("-", "_")
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
        mask[ROT_START:REPR_DIM] = 1.0
        if include_root_y:
            mask[ROOT_Y_IDX] = 1.0
    elif mode == "rot_only":
        mask[ROT_START:REPR_DIM] = 1.0
    else:
        raise ValueError(
            f"Unknown EDGE_UNIT_PRIOR_FEATURES={mode!r}. Use upper, upper+torso, "
            "arms, torso, lower_safe, upper+lower_safe, all_no_root, rot_only."
        )

    mask[CONTACT_SLICE] = 0.0
    mask[ROOT_X_IDX] = 0.0
    mask[ROOT_Z_IDX] = 0.0
    return mask.astype(np.float32)


def _mask_safety_report(prior_mask: np.ndarray) -> Dict[str, object]:
    nonzero = np.where(prior_mask > 0)[0].astype(int).tolist()
    return {
        "selected_feature_count": int(len(nonzero)),
        "selected_feature_indices_head": nonzero[:32],
        "mask_contact_sum": float(prior_mask[CONTACT_SLICE].sum()),
        "mask_root_x_sum": float(prior_mask[ROOT_X_IDX]),
        "mask_root_y_sum": float(prior_mask[ROOT_Y_IDX]),
        "mask_root_z_sum": float(prior_mask[ROOT_Z_IDX]),
        "safe_no_contact": bool(float(prior_mask[CONTACT_SLICE].sum()) == 0.0),
        "safe_no_root_xz": bool(float(prior_mask[ROOT_X_IDX] + prior_mask[ROOT_Z_IDX]) == 0.0),
    }


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
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
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
    return unit[start : start + max_len].astype(np.float32)


def _center_pose_only(unit: np.ndarray) -> np.ndarray:
    unit = _as_unit_t151(unit)
    return unit[len(unit) // 2 : len(unit) // 2 + 1].astype(np.float32)


def _select_temporal_window(unit: np.ndarray) -> np.ndarray:
    unit = _as_unit_t151(unit)
    if not _env_bool("EDGE_UNIT_PRIOR_TEMPORAL", False):
        return _center_pose_only(unit)
    window = _env_int("EDGE_UNIT_PRIOR_WINDOW", _env_int("EDGE_UNIT_PRIOR_MAX_LEN", 45))
    return _crop_unit_center(unit, max_len=window)


def _dct_lowpass_numpy(x: np.ndarray, k: int) -> np.ndarray:
    """Low-pass temporal unit prior with optional soft spectral decay.

    Historical behavior used a hard cutoff (all coefficients after k set to
    zero).  That is still available with EDGE_UNIT_PRIOR_DCT_DECAY=hard, but
    formal V10 runs should usually prefer soft_exp so medium/high-frequency
    expressive details are attenuated instead of fully removed.

    Environment:
        EDGE_UNIT_PRIOR_DCT_DECAY=soft_exp|hard
        EDGE_UNIT_PRIOR_DCT_DECAY_STRENGTH=3.0
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or len(x) <= 2:
        return x.astype(np.float32)

    k = int(max(1, min(k, len(x))))
    decay_mode = _env_str("EDGE_UNIT_PRIOR_DCT_DECAY", "hard").strip().lower()
    decay_strength = _env_float("EDGE_UNIT_PRIOR_DCT_DECAY_STRENGTH", 3.0)

    def _apply_tail_decay(coeff: np.ndarray) -> np.ndarray:
        if k >= coeff.shape[0]:
            return coeff
        if decay_mode in {"soft", "soft_exp", "exp", "exponential"}:
            n_tail = coeff.shape[0] - k
            # Starts at 1.0 at the cutoff and decays smoothly, preserving some
            # mid/high-frequency expressiveness while suppressing abrupt jumps.
            decay = np.exp(-np.linspace(0.0, max(decay_strength, 0.0), n_tail)).astype(np.float32)
            coeff[k:, :] *= decay[:, None]
        else:
            coeff[k:, :] = 0.0
        return coeff

    try:
        from scipy.fftpack import dct, idct

        coeff = dct(x, type=2, norm="ortho", axis=0)
        coeff = _apply_tail_decay(coeff)
        return idct(coeff, type=2, norm="ortho", axis=0).astype(np.float32)
    except Exception:
        freq = np.fft.rfft(x, axis=0)
        # For rFFT, k is still interpreted as the number of low-frequency bins
        # to preserve before decay/cutoff.
        kk = int(max(1, min(k, freq.shape[0])))
        if kk < freq.shape[0]:
            if decay_mode in {"soft", "soft_exp", "exp", "exponential"}:
                n_tail = freq.shape[0] - kk
                decay = np.exp(-np.linspace(0.0, max(decay_strength, 0.0), n_tail)).astype(np.float32)
                freq[kk:, :] *= decay[:, None]
            else:
                freq[kk:, :] = 0.0
        return np.fft.irfft(freq, n=len(x), axis=0).astype(np.float32)


def _maybe_frequency_decouple_unit(unit: np.ndarray, prior_mask: np.ndarray) -> np.ndarray:
    if not _env_bool("EDGE_UNIT_PRIOR_DCT", False):
        return unit.astype(np.float32)
    k = _env_int("EDGE_UNIT_PRIOR_LOW_FREQ_K", 4)
    out = unit.astype(np.float32).copy()
    feat_idx = np.where(prior_mask > 0)[0]
    if len(feat_idx):
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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Apply one retrieved-unit soft prior and return a per-unit report."""
    value = np.asarray(value, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    unit = _as_unit_t151(unit_motion)

    if value.ndim != 2 or mask.ndim != 2 or value.shape != mask.shape or value.shape[1] != REPR_DIM:
        raise ValueError(f"value/mask must both be [T,151], got value={value.shape}, mask={mask.shape}")

    unit = _select_temporal_window(unit)
    prior_mask = make_unit_prior_mask(feature_mode)
    unit = _maybe_frequency_decouple_unit(unit, prior_mask)

    center_frame = int(center_frame)
    half = len(unit) // 2
    total_frames = value.shape[0]
    strength = float(max(0.0, strength))
    combine = _env_str("EDGE_UNIT_PRIOR_COMBINE", "max").lower()

    report: Dict[str, object] = {
        "center_frame": center_frame,
        "unit_len_after_window": int(len(unit)),
        "strength": strength,
        "feature_mode": feature_mode,
        "temporal": _env_bool("EDGE_UNIT_PRIOR_TEMPORAL", False),
        "window": _env_int("EDGE_UNIT_PRIOR_WINDOW", _env_int("EDGE_UNIT_PRIOR_MAX_LEN", 45)),
        "dct_lowpass": _env_bool("EDGE_UNIT_PRIOR_DCT", False),
        "dct_lowpass_k": _env_int("EDGE_UNIT_PRIOR_LOW_FREQ_K", 4),
        "dct_decay": _env_str("EDGE_UNIT_PRIOR_DCT_DECAY", "hard"),
        "dct_decay_strength": _env_float("EDGE_UNIT_PRIOR_DCT_DECAY_STRENGTH", 3.0),
        "combine": combine,
        **_mask_safety_report(prior_mask),
        "actual_frames": [],
        "applied_frames": 0,
    }

    if strength <= 0.0:
        return value, mask, report

    actual_frames: List[int] = []
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
        if np.any(new_mask > 0):
            actual_frames.append(int(frame))

    report["actual_frames"] = sorted(set(actual_frames))
    report["applied_frames"] = int(len(set(actual_frames)))
    return value, mask, report


def _explicit_unit_specs_from_env() -> List[Dict[str, object]]:
    paths = _split_csv(os.environ.get("EDGE_UNIT_PRIOR_UNIT_PATHS", ""))
    if not paths:
        return []
    raw_frames = _split_csv(os.environ.get("EDGE_UNIT_PRIOR_MID_FRAMES", ""))
    if raw_frames and len(raw_frames) != len(paths):
        msg = (
            "EDGE_UNIT_PRIOR_UNIT_PATHS and EDGE_UNIT_PRIOR_MID_FRAMES length mismatch: "
            f"{len(paths)} paths vs {len(raw_frames)} frames"
        )
        if _required() or _formal():
            raise FormalUnitPriorContractError("CRITICAL: " + msg)
        raise RuntimeError(msg)
    specs = []
    for i, path in enumerate(paths):
        frame = int(float(raw_frames[i])) if raw_frames else -1
        specs.append({"unit_path": path, "frame": frame, "source": "env"})
    return specs


def unit_specs_from_saved_records(saved_records: Iterable[Dict]) -> List[Dict[str, object]]:
    specs = []
    for rec in saved_records or []:
        unit_path = rec.get("unit_path") or rec.get("unit_motion_path")
        frame = rec.get("frame")
        if unit_path and frame is not None:
            specs.append({"unit_path": str(unit_path), "frame": int(frame), "source": "saved_record"})
    return specs


def infer_unit_specs_from_mid_paths(mid_paths: Iterable[str], mid_frames: Iterable[int]) -> List[Dict[str, object]]:
    """Infer unit specs, preferring explicit env paths over sibling inference."""
    explicit = _explicit_unit_specs_from_env()
    if explicit:
        frames = list(mid_frames or [])
        for i, spec in enumerate(explicit):
            if int(spec.get("frame", -1)) < 0 and i < len(frames):
                spec["frame"] = int(frames[i])
        return explicit

    specs: List[Dict[str, object]] = []
    for pose_path, frame in zip(mid_paths or [], mid_frames or []):
        p = Path(str(pose_path))
        candidates = []
        if p.suffix == ".npy":
            candidates.append(p.with_name(p.stem + "_unit.npy"))
            candidates.append(p.with_name(p.stem.replace("_pose", "") + "_unit.npy"))
        candidates.append(Path(str(pose_path) + "_unit.npy"))
        for unit_path in candidates:
            if unit_path.is_file():
                specs.append({"unit_path": str(unit_path), "frame": int(frame), "source": "sibling_inference"})
                break
    return specs


def get_last_unit_prior_report() -> Dict[str, object]:
    return dict(_LAST_UNIT_PRIOR_REPORT)


def _new_report(enabled: bool, specs_list: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "enabled": bool(enabled),
        "required": bool(_required()),
        "formal": bool(_formal()),
        "num_specs": int(len(specs_list)),
        "num_applied": 0,
        "num_missing": 0,
        "missing_specs": [],
        "applied_unit_paths": [],
        "applied_mid_frames": [],
        "per_unit": [],
        "feature_mode": os.environ.get("EDGE_UNIT_PRIOR_FEATURES", "upper"),
        "strength": _env_float("EDGE_UNIT_PRIOR_STRENGTH", 0.012),
        "temporal": _env_bool("EDGE_UNIT_PRIOR_TEMPORAL", False),
        "window": _env_int("EDGE_UNIT_PRIOR_WINDOW", _env_int("EDGE_UNIT_PRIOR_MAX_LEN", 45)),
        "dct_lowpass": _env_bool("EDGE_UNIT_PRIOR_DCT", False),
        "dct_lowpass_k": _env_int("EDGE_UNIT_PRIOR_LOW_FREQ_K", 4),
        "dct_decay": _env_str("EDGE_UNIT_PRIOR_DCT_DECAY", "hard"),
        "dct_decay_strength": _env_float("EDGE_UNIT_PRIOR_DCT_DECAY_STRENGTH", 3.0),
    }


def apply_unit_priors_from_specs(value: np.ndarray, mask: np.ndarray, specs: Iterable[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply unit priors and enforce formal required contract."""
    global _LAST_UNIT_PRIOR_REPORT

    enabled = _env_bool("EDGE_UNIT_SOFT_PRIOR", False)
    specs_list = list(specs or [])
    report = _new_report(enabled=enabled, specs_list=specs_list)

    if not enabled:
        _LAST_UNIT_PRIOR_REPORT = report
        _write_report_if_requested(report)
        return value, mask

    strength = float(report["strength"])
    feature_mode = str(report["feature_mode"])
    max_len = _env_int("EDGE_UNIT_PRIOR_MAX_LEN", 45)
    decay_gamma = _env_float("EDGE_UNIT_PRIOR_DECAY_GAMMA", 1.0)
    decay_floor = _env_float("EDGE_UNIT_PRIOR_DECAY_FLOOR", 0.0)

    applied = 0
    missing = 0
    for spec in specs_list:
        unit_path = str(spec.get("unit_path", ""))
        frame = spec.get("frame", None)
        if not unit_path or frame is None or not Path(unit_path).is_file():
            missing += 1
            report["missing_specs"].append({"unit_path": unit_path, "frame": frame, "reason": "missing_file_or_frame"})
            continue
        try:
            unit = _load_unit(unit_path)
            value, mask, unit_report = add_unit_prior(
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
        except BaseException:
            raise
        except Exception as exc:
            # In formal/required mode, malformed unit files are contract failures.
            report["missing_specs"].append({
                "unit_path": unit_path,
                "frame": frame,
                "reason": f"load_or_apply_error:{type(exc).__name__}:{exc}",
            })
            if _required() or _formal():
                _contract_fail(
                    f"Failed to load/apply required unit prior from {unit_path}: {exc}",
                    report=report,
                )
            missing += 1
            continue

        unit_report["unit_path"] = unit_path
        unit_report["frame"] = int(frame)
        unit_report["spec_source"] = spec.get("source", "unknown")
        report["per_unit"].append(unit_report)
        report["applied_unit_paths"].append(unit_path)
        report["applied_mid_frames"].append(int(frame))
        applied += 1

    report["num_applied"] = int(applied)
    report["num_missing"] = int(missing)
    report["temporal_prior_actual_frames"] = sorted(
        set(int(f) for u in report["per_unit"] for f in u.get("actual_frames", []))
    )

    prior_mask = make_unit_prior_mask(feature_mode)
    report.update({f"aggregate_{k}": v for k, v in _mask_safety_report(prior_mask).items()})

    if applied:
        dct_msg = f", dct_lowpass_k={report['dct_lowpass_k']}" if report["dct_lowpass"] else ""
        print(
            "✅ ChoreoRAG unit soft prior applied: "
            f"count={applied}, strength={strength}, features={feature_mode}, "
            f"temporal={report['temporal']}, window={report['window']}{dct_msg}"
        )
    else:
        msg = f"EDGE_UNIT_SOFT_PRIOR=1 but no valid unit prior was applied; specs={len(specs_list)}, missing={missing}"
        if _required() or _formal():
            _contract_fail(msg, report=report)
        print("⚠️ " + msg)

    if not bool(report.get("aggregate_safe_no_contact", False)) or not bool(report.get("aggregate_safe_no_root_xz", False)):
        _contract_fail(f"Unit prior mask safety invariant violated: {report}", report=report)

    _LAST_UNIT_PRIOR_REPORT = report
    _write_report_if_requested(report)
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
