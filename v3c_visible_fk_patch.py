from __future__ import annotations

import os
import math
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F


_TRUE = {"1", "true", "yes", "y", "on"}
_DCT_CACHE: Dict[tuple, torch.Tensor] = {}
_SMPL_CACHE: Dict[str, Any] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


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


def _v3_enabled() -> bool:
    return _env_bool("EDGE_V3_UNIT_RECON", False) or (
        os.environ.get("EDGE_TRAIN_PROFILE", "").strip().lower() == "v3_unit_recon"
    )


def _visible_fk_enabled() -> bool:
    return _v3_enabled() and _env_bool("EDGE_V3C_VISIBLE_FK", False)


def _v3h_enabled() -> bool:
    return _v3_enabled() and _env_bool("EDGE_V3H_SUPPORT_CHAIN_LOSS", False)


def _stable_enabled() -> bool:
    # Default ON for V3 because old MSE/FK support loss can create rare e20/e25 spikes.
    return _v3_enabled() and _env_bool("EDGE_V3_LOSS_STABILITY", True)


def _maybe_unnormalize_tensor(normalizer: Any, x: torch.Tensor) -> torch.Tensor:
    if normalizer is None or not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
        return x
    mean = torch.as_tensor(normalizer.mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(normalizer.std, device=x.device, dtype=x.dtype)
    while mean.ndim < x.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return x * std + mean


def _sanitize_tensor(x: torch.Tensor, clip: float | None = None) -> torch.Tensor:
    if clip is None:
        clip = _env_float("EDGE_V3H_PHYS_CLIP", 6.0)
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=float(clip), neginf=-float(clip))
    if _env_bool("EDGE_V3H_CLAMP_PHYS", True):
        x = x.clamp(-float(clip), float(clip))
    return x


def _get_smpl(device: torch.device):
    key = str(device)
    smpl = _SMPL_CACHE.get(key)
    if smpl is None:
        from vis import SMPLSkeleton
        smpl = SMPLSkeleton(device=device)
        _SMPL_CACHE[key] = smpl
    return smpl


def _fk_positions_151(motion_151: torch.Tensor) -> torch.Tensor:
    if motion_151.ndim != 3 or motion_151.shape[-1] != 151:
        raise ValueError(f"expected [B,T,151], got {tuple(motion_151.shape)}")
    from dataset.quaternion import ax_from_6v
    b, t, _ = motion_151.shape
    pos = motion_151[:, :, 4:7]
    q6 = motion_151[:, :, 7:].reshape(b, t, 24, 6)
    qax = ax_from_6v(q6)
    smpl = _get_smpl(motion_151.device)
    return smpl.forward(qax, pos)


def _safe_fk_positions_151(motion_151: torch.Tensor) -> torch.Tensor | None:
    try:
        motion_151 = _sanitize_tensor(motion_151)
        pos = _fk_positions_151(motion_151)
        return _sanitize_tensor(pos, clip=_env_float("EDGE_V3H_FK_POS_CLIP", 8.0))
    except Exception:
        return None


def _dct_basis(length: int, keep: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    keep = max(1, min(int(keep), int(length)))
    key = (int(length), int(keep), str(device), str(dtype))
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached

    n = torch.arange(length, device=device, dtype=dtype).view(length, 1)
    k = torch.arange(keep, device=device, dtype=dtype).view(1, keep)
    basis = torch.cos(math.pi / float(length) * (n + 0.5) * k)
    basis[:, 0] *= math.sqrt(1.0 / float(length))
    if keep > 1:
        basis[:, 1:] *= math.sqrt(2.0 / float(length))
    _DCT_CACHE[key] = basis
    return basis


def _dct_lowpass(x: torch.Tensor, keep: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] <= 1:
        return x
    basis = _dct_basis(x.shape[1], keep, x.device, x.dtype)
    coeff = torch.einsum("btc,tk->bkc", x, basis)
    return torch.einsum("bkc,tk->btc", coeff, basis)


def _rot_dims(joints: Iterable[int]) -> List[int]:
    dims = []
    for j in joints:
        start = 7 + 6 * int(j)
        dims.extend(range(start, min(start + 6, 151)))
    return dims


# EDGE 151D convention:
# 0:4 contacts
# 4:7 root XYZ
# 7:151 24 joints × 6D rotations
CONTACTS = [0, 1, 2, 3]

PELVIS_ROT = _rot_dims([0])
HIPS_ROT = _rot_dims([1, 2])
SPINE_TORSO_ROT = _rot_dims([3, 6, 9])
KNEES_ROT = _rot_dims([4, 5])
ANKLES_FEET_ROT = _rot_dims([7, 8, 10, 11])
NECK_HEAD_ROT = _rot_dims([12, 15])
ARMS_HANDS_ROT = _rot_dims([13, 14, 16, 17, 18, 19, 20, 21, 22, 23])

UPPER_SAFE_ROT = SPINE_TORSO_ROT + NECK_HEAD_ROT + ARMS_HANDS_ROT
SUPPORT_ROT = PELVIS_ROT + HIPS_ROT + KNEES_ROT + ANKLES_FEET_ROT

LOWER_SUPPORT_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
FEET_JOINTS = [10, 11]


def _vel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 2:
        return x[:, :0]
    return x[:, 1:] - x[:, :-1]


def _per_sample_mean(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0:
        return x.view(1)
    return x.reshape(x.shape[0], -1).mean(dim=1)


def _huber_from_diff(diff: torch.Tensor) -> torch.Tensor:
    # Manual SmoothL1 with beta=1.0 to avoid version-specific F.smooth_l1_loss beta issues.
    abs_d = diff.abs()
    return torch.where(abs_d < 1.0, 0.5 * diff.pow(2), abs_d - 0.5)


def _target_scale(target: torch.Tensor, floor: float) -> torch.Tensor:
    scale = torch.sqrt(target.detach().pow(2).reshape(target.shape[0], -1).mean(dim=1) + 1e-8)
    scale = scale.clamp_min(float(floor))
    while scale.ndim < target.ndim:
        scale = scale.unsqueeze(-1)
    return scale


def _robust_feature_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    floor: float = 0.01,
    diff_clip: float | None = None,
    sample_cap: float | None = None,
) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)

    diff_clip = _env_float("EDGE_V3H_DIFF_CLAMP", 8.0) if diff_clip is None else float(diff_clip)
    sample_cap = _env_float("EDGE_V3H_TERM_CAP", 25.0) if sample_cap is None else float(sample_cap)

    pred = _sanitize_tensor(pred)
    target = _sanitize_tensor(target)
    scale = _target_scale(target, floor=floor)
    diff = ((pred - target) / scale).clamp(-diff_clip, diff_clip)
    ps = _per_sample_mean(_huber_from_diff(diff))
    ps = torch.nan_to_num(ps, nan=0.0, posinf=sample_cap, neginf=0.0).clamp(0.0, sample_cap)
    return ps.mean()


def _range_floor_loss(pred: torch.Tensor, target: torch.Tensor, floor_ratio: float, scale_floor: float = 0.02) -> torch.Tensor:
    if pred is None or target is None or pred.shape[1] < 2:
        return pred.new_tensor(0.0) if pred is not None else torch.tensor(0.0)

    pred = _sanitize_tensor(pred)
    target = _sanitize_tensor(target)
    pred_range = torch.linalg.norm(pred.amax(dim=1) - pred.amin(dim=1), dim=-1).mean(dim=-1)
    target_range = torch.linalg.norm(target.amax(dim=1) - target.amin(dim=1), dim=-1).mean(dim=-1)
    scale = target_range.detach().clamp_min(float(scale_floor))
    diff = torch.relu(float(floor_ratio) * target_range.detach() - pred_range) / scale
    diff = diff.clamp(0.0, _env_float("EDGE_V3H_DIFF_CLAMP", 8.0))
    ps = _huber_from_diff(diff).clamp(0.0, _env_float("EDGE_V3H_TERM_CAP", 25.0))
    return ps.mean()


def _activity_floor_loss(pred_v: torch.Tensor, target_v: torch.Tensor, floor_ratio: float, scale_floor: float = 0.01) -> torch.Tensor:
    if pred_v.numel() == 0 or target_v.numel() == 0:
        return pred_v.new_tensor(0.0)
    pred_activity = pred_v.reshape(pred_v.shape[0], -1).mean(dim=1)
    target_activity = target_v.reshape(target_v.shape[0], -1).mean(dim=1)
    scale = target_activity.detach().clamp_min(float(scale_floor))
    diff = torch.relu(float(floor_ratio) * target_activity.detach() - pred_activity) / scale
    diff = diff.clamp(0.0, _env_float("EDGE_V3H_DIFF_CLAMP", 8.0))
    ps = _huber_from_diff(diff).clamp(0.0, _env_float("EDGE_V3H_TERM_CAP", 25.0))
    return ps.mean()


def _upper_safe_rot_features(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 151:
        return x
    return x[..., UPPER_SAFE_ROT]


def _optional_upper_rot_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    if not _env_bool("EDGE_V3C_UPPER_ROT_LOSS", False):
        return pred_phys.new_tensor(0.0)
    pred = _upper_safe_rot_features(pred_phys.float())
    target = _upper_safe_rot_features(target_phys.float())
    loss = _env_float("EDGE_V3C_UPPER_ROT_WEIGHT", 0.05) * _robust_feature_loss(
        pred, target, floor=0.02
    )
    if pred.shape[1] > 1:
        loss = loss + _env_float("EDGE_V3C_UPPER_ROT_VEL_WEIGHT", 0.10) * _robust_feature_loss(
            _vel(pred), _vel(target), floor=0.01
        )
    dct_w = _env_float("EDGE_V3C_UPPER_ROT_DCT_WEIGHT", 0.0)
    if dct_w > 0:
        keep = _env_int("EDGE_V3C_UPPER_ROT_DCT_KEEP", 8)
        loss = loss + dct_w * _robust_feature_loss(
            _dct_lowpass(pred, keep), _dct_lowpass(target, keep), floor=0.02
        )
    return loss


def _visible_upper_fk_loss(
    self,
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    pred_phys: torch.Tensor | None = None,
    target_phys: torch.Tensor | None = None,
    pred_pos: torch.Tensor | None = None,
    target_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred_x0.ndim != 3 or pred_x0.shape[-1] != 151 or pred_x0.shape[1] < 3:
        return pred_x0.new_tensor(0.0)

    if pred_phys is None:
        pred_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), pred_x0.float())
    if target_phys is None:
        target_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), target_x0.float())

    pred_phys = _sanitize_tensor(pred_phys)
    target_phys = _sanitize_tensor(target_phys)

    if pred_pos is None:
        pred_pos = _safe_fk_positions_151(pred_phys)
    if target_pos is None:
        target_pos = _safe_fk_positions_151(target_phys)

    total = pred_x0.new_tensor(0.0)

    if pred_pos is not None and target_pos is not None:
        joint_mode = os.environ.get("EDGE_V3C_VISIBLE_JOINTS", "upper_safe_plus").strip().lower()
        if joint_mode in {"arms", "hands"}:
            upper_joints = [16, 17, 18, 19, 20, 21, 22, 23]
        else:
            upper_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        pred_u = pred_pos[:, :, upper_joints, :]
        target_u = target_pos[:, :, upper_joints, :]

        pred_v = torch.linalg.norm(_vel(pred_u), dim=-1)
        target_v = torch.linalg.norm(_vel(target_u), dim=-1)

        speed_loss = _robust_feature_loss(pred_v, target_v, floor=0.02)
        activity_loss = _activity_floor_loss(
            pred_v, target_v, _env_float("EDGE_V3C_ACTIVITY_FLOOR", 0.70), scale_floor=0.02
        )
        range_loss = _range_floor_loss(
            pred_u, target_u, _env_float("EDGE_V3C_RANGE_FLOOR", 0.65), scale_floor=0.05
        )

        hand_joints = [20, 21, 22, 23]
        pred_h = pred_pos[:, :, hand_joints, :]
        target_h = target_pos[:, :, hand_joints, :]
        hand_speed = _robust_feature_loss(
            torch.linalg.norm(_vel(pred_h), dim=-1),
            torch.linalg.norm(_vel(target_h), dim=-1),
            floor=0.02,
        )
        hand_range = _range_floor_loss(
            pred_h, target_h, _env_float("EDGE_V3C_RANGE_FLOOR", 0.65), scale_floor=0.05
        )

        total = total + (
            _env_float("EDGE_V3C_FK_SPEED_WEIGHT", 1.00) * speed_loss
            + _env_float("EDGE_V3C_FK_ACTIVITY_WEIGHT", 3.00) * activity_loss
            + _env_float("EDGE_V3C_FK_RANGE_WEIGHT", 2.00) * range_loss
            + _env_float("EDGE_V3C_FK_HAND_WEIGHT", 1.00) * hand_speed
            + _env_float("EDGE_V3C_FK_HAND_RANGE_WEIGHT", 2.00) * hand_range
        )

    total = total + _optional_upper_rot_loss(pred_phys, target_phys)
    total = torch.nan_to_num(total, nan=0.0, posinf=_env_float("EDGE_V3C_SAMPLE_LOSS_CAP", 25.0), neginf=0.0)
    return total.clamp(0.0, _env_float("EDGE_V3C_SAMPLE_LOSS_CAP", 25.0))


def _lower_rot_velocity_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    if pred_phys.shape[-1] != 151 or pred_phys.shape[1] < 2:
        return pred_phys.new_tensor(0.0)

    loss = pred_phys.new_tensor(0.0)

    def add_rot_vel(name: str, dims: List[int], default_w: float, floor: float = 0.01) -> None:
        nonlocal loss
        w = _env_float(name, default_w)
        if w <= 0:
            return
        loss = loss + w * _robust_feature_loss(
            _vel(pred_phys[:, :, dims]),
            _vel(target_phys[:, :, dims]),
            floor=floor,
        )

    def add_rot_abs(name: str, dims: List[int], default_w: float, floor: float = 0.05) -> None:
        nonlocal loss
        w = _env_float(name, default_w)
        if w <= 0:
            return
        loss = loss + w * _robust_feature_loss(
            pred_phys[:, :, dims],
            target_phys[:, :, dims],
            floor=floor,
        )

    add_rot_abs("EDGE_V3H_PELVIS_ABS_WEIGHT", PELVIS_ROT, 0.02)
    add_rot_abs("EDGE_V3H_HIPS_ABS_WEIGHT", HIPS_ROT, 0.015)

    add_rot_vel("EDGE_V3H_PELVIS_ROT_VEL_WEIGHT", PELVIS_ROT, 0.12)
    add_rot_vel("EDGE_V3H_HIPS_ROT_VEL_WEIGHT", HIPS_ROT, 0.10)
    add_rot_vel("EDGE_V3H_KNEES_ROT_VEL_WEIGHT", KNEES_ROT, 0.08)
    add_rot_vel("EDGE_V3H_ANKLES_FEET_ROT_VEL_WEIGHT", ANKLES_FEET_ROT, 0.06)

    dct_w = _env_float("EDGE_V3H_SUPPORT_ROT_DCT_WEIGHT", 0.0)
    if dct_w > 0:
        keep = _env_int("EDGE_V3H_SUPPORT_ROT_DCT_KEEP", 8)
        pred = pred_phys[:, :, SUPPORT_ROT]
        target = target_phys[:, :, SUPPORT_ROT]
        loss = loss + dct_w * _robust_feature_loss(
            _dct_lowpass(pred, keep), _dct_lowpass(target, keep), floor=0.02
        )

    return loss


def _root_support_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    if pred_phys.shape[-1] != 151 or pred_phys.shape[1] < 2:
        return pred_phys.new_tensor(0.0)

    pred_root = pred_phys[:, :, 4:7]
    target_root = target_phys[:, :, 4:7]

    pred_root_v = _vel(pred_root)
    target_root_v = _vel(target_root)

    root_y_v = _robust_feature_loss(
        pred_root_v[:, :, 1:2],
        target_root_v[:, :, 1:2],
        floor=_env_float("EDGE_V3H_ROOT_Y_SCALE_FLOOR", 0.01),
    )
    root_xz_v = _robust_feature_loss(
        pred_root_v[:, :, [0, 2]],
        target_root_v[:, :, [0, 2]],
        floor=_env_float("EDGE_V3H_ROOT_XZ_SCALE_FLOOR", 0.01),
    )
    root_y_abs = _robust_feature_loss(
        pred_root[:, :, 1:2],
        target_root[:, :, 1:2],
        floor=_env_float("EDGE_V3H_ROOT_Y_ABS_SCALE_FLOOR", 0.05),
    )

    return (
        _env_float("EDGE_V3H_ROOT_Y_VEL_WEIGHT", 0.35) * root_y_v
        + _env_float("EDGE_V3H_ROOT_XZ_VEL_WEIGHT", 0.08) * root_xz_v
        + _env_float("EDGE_V3H_ROOT_Y_ABS_WEIGHT", 0.02) * root_y_abs
    )


def _lower_fk_support_loss(pred_pos: torch.Tensor | None, target_pos: torch.Tensor | None, device_tensor: torch.Tensor) -> torch.Tensor:
    if pred_pos is None or target_pos is None:
        return device_tensor.new_tensor(0.0)
    if pred_pos.shape[1] < 2:
        return device_tensor.new_tensor(0.0)

    pred_lower = pred_pos[:, :, LOWER_SUPPORT_JOINTS, :]
    target_lower = target_pos[:, :, LOWER_SUPPORT_JOINTS, :]

    pred_lower_v = torch.linalg.norm(_vel(pred_lower), dim=-1)
    target_lower_v = torch.linalg.norm(_vel(target_lower), dim=-1)

    speed = _robust_feature_loss(pred_lower_v, target_lower_v, floor=0.02)
    activity = _activity_floor_loss(
        pred_lower_v,
        target_lower_v,
        _env_float("EDGE_V3H_LOWER_ACTIVITY_FLOOR", 0.60),
        scale_floor=0.02,
    )
    range_loss = _range_floor_loss(
        pred_lower,
        target_lower,
        _env_float("EDGE_V3H_LOWER_RANGE_FLOOR", 0.55),
        scale_floor=0.05,
    )

    pred_feet = pred_pos[:, :, FEET_JOINTS, :]
    target_feet = target_pos[:, :, FEET_JOINTS, :]

    pred_feet_v = torch.linalg.norm(_vel(pred_feet), dim=-1)
    target_feet_v = torch.linalg.norm(_vel(target_feet), dim=-1)

    foot_speed = _robust_feature_loss(pred_feet_v, target_feet_v, floor=0.02)
    foot_height = _robust_feature_loss(pred_feet[:, :, :, 1], target_feet[:, :, :, 1], floor=0.03)
    foot_range = _range_floor_loss(
        pred_feet,
        target_feet,
        _env_float("EDGE_V3H_FOOT_RANGE_FLOOR", 0.50),
        scale_floor=0.05,
    )

    dct_fk_w = _env_float("EDGE_V3H_LOWER_FK_DCT_WEIGHT", 0.0)
    dct_loss = device_tensor.new_tensor(0.0)
    if dct_fk_w > 0:
        keep = _env_int("EDGE_V3H_LOWER_FK_DCT_KEEP", 8)
        pred_flat = pred_lower.reshape(pred_lower.shape[0], pred_lower.shape[1], -1)
        target_flat = target_lower.reshape(target_lower.shape[0], target_lower.shape[1], -1)
        dct_loss = dct_fk_w * _robust_feature_loss(
            _dct_lowpass(pred_flat, keep), _dct_lowpass(target_flat, keep), floor=0.05
        )

    return (
        _env_float("EDGE_V3H_LOWER_FK_SPEED_WEIGHT", 0.60) * speed
        + _env_float("EDGE_V3H_LOWER_FK_ACTIVITY_WEIGHT", 0.60) * activity
        + _env_float("EDGE_V3H_LOWER_FK_RANGE_WEIGHT", 0.45) * range_loss
        + _env_float("EDGE_V3H_FOOT_FK_SPEED_WEIGHT", 0.25) * foot_speed
        + _env_float("EDGE_V3H_FOOT_HEIGHT_WEIGHT", 0.15) * foot_height
        + _env_float("EDGE_V3H_FOOT_RANGE_WEIGHT", 0.25) * foot_range
        + dct_loss
    )


def _contact_support_loss(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    pred_pos: torch.Tensor | None,
    target_pos: torch.Tensor | None,
) -> torch.Tensor:
    if pred_phys.shape[-1] != 151 or pred_phys.shape[1] < 2:
        return pred_phys.new_tensor(0.0)

    pred_c = pred_phys[:, :, CONTACTS].clamp(0.0, 1.0)
    target_c = target_phys[:, :, CONTACTS].clamp(0.0, 1.0)

    contact_abs = _robust_feature_loss(pred_c, target_c, floor=0.20)
    contact_switch = _robust_feature_loss(_vel(pred_c), _vel(target_c), floor=0.10)

    loss = (
        _env_float("EDGE_V3H_CONTACT_RECON_WEIGHT", 0.02) * contact_abs
        + _env_float("EDGE_V3H_CONTACT_SWITCH_WEIGHT", 0.04) * contact_switch
    )

    if pred_pos is not None and target_pos is not None and pred_pos.shape[1] > 1:
        pred_feet = pred_pos[:, :, FEET_JOINTS, :]
        pred_feet_v = torch.linalg.norm(_vel(pred_feet), dim=-1)

        left_contact = torch.maximum(target_c[:, :, 0], target_c[:, :, 2])
        right_contact = torch.maximum(target_c[:, :, 1], target_c[:, :, 3])
        target_contact_lr = torch.stack([left_contact, right_contact], dim=-1)
        target_contact_lr_v = 0.5 * (target_contact_lr[:, 1:] + target_contact_lr[:, :-1]).detach()

        gated_foot_v = target_contact_lr_v * pred_feet_v
        zero = torch.zeros_like(gated_foot_v)
        contact_foot_v = _robust_feature_loss(gated_foot_v, zero, floor=0.03)

        pred_foot_y = pred_feet[:, :, :, 1]
        target_foot_y = target_pos[:, :, FEET_JOINTS, 1]
        contact_y = _robust_feature_loss(
            target_contact_lr.detach() * pred_foot_y,
            target_contact_lr.detach() * target_foot_y,
            floor=0.03,
        )

        loss = loss + (
            _env_float("EDGE_V3H_CONTACT_FOOT_VEL_WEIGHT", 0.02) * contact_foot_v
            + _env_float("EDGE_V3H_CONTACT_FOOT_HEIGHT_WEIGHT", 0.02) * contact_y
        )

    return loss


def _support_chain_fk_loss(
    self,
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    pred_phys: torch.Tensor | None = None,
    target_phys: torch.Tensor | None = None,
    pred_pos: torch.Tensor | None = None,
    target_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _v3h_enabled():
        return pred_x0.new_tensor(0.0)

    if pred_x0.ndim != 3 or pred_x0.shape[-1] != 151 or pred_x0.shape[1] < 3:
        return pred_x0.new_tensor(0.0)

    if pred_phys is None:
        pred_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), pred_x0.float())
    if target_phys is None:
        target_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), target_x0.float())

    pred_phys = _sanitize_tensor(pred_phys)
    target_phys = _sanitize_tensor(target_phys)

    if pred_pos is None:
        pred_pos = _safe_fk_positions_151(pred_phys)
    if target_pos is None:
        target_pos = _safe_fk_positions_151(target_phys)

    root_loss = _root_support_loss(pred_phys, target_phys)
    rot_loss = _lower_rot_velocity_loss(pred_phys, target_phys)
    fk_loss = _lower_fk_support_loss(pred_pos, target_pos, pred_x0)
    contact_loss = _contact_support_loss(pred_phys, target_phys, pred_pos, target_pos)

    total = root_loss + rot_loss + fk_loss + contact_loss
    cap = _env_float("EDGE_V3H_SAMPLE_LOSS_CAP", 25.0)
    total = torch.nan_to_num(total, nan=0.0, posinf=cap, neginf=0.0).clamp(0.0, cap)

    if _env_bool("EDGE_V3H_DEBUG", False):
        if not hasattr(self, "_edge_v3h_debug_counter"):
            self._edge_v3h_debug_counter = 0
        self._edge_v3h_debug_counter += 1
        if self._edge_v3h_debug_counter <= 20 or self._edge_v3h_debug_counter % 100 == 0:
            with torch.no_grad():
                pred_root_v = _vel(pred_phys[:, :, 4:7])
                target_root_v = _vel(target_phys[:, :, 4:7])
                pred_pelvis_v = _vel(pred_phys[:, :, PELVIS_ROT]).abs().mean()
                target_pelvis_v = _vel(target_phys[:, :, PELVIS_ROT]).abs().mean()
                pred_hips_v = _vel(pred_phys[:, :, HIPS_ROT]).abs().mean()
                target_hips_v = _vel(target_phys[:, :, HIPS_ROT]).abs().mean()
                print(
                    "🧪 V3H stable support-chain | "
                    f"root={float(root_loss.detach().cpu()):.6f} "
                    f"rot={float(rot_loss.detach().cpu()):.6f} "
                    f"fk={float(fk_loss.detach().cpu()):.6f} "
                    f"contact={float(contact_loss.detach().cpu()):.6f} "
                    f"total_capped={float(total.detach().cpu()):.6f} "
                    f"pred_rootYv={float(pred_root_v[:, :, 1].abs().mean().detach().cpu()):.6f} "
                    f"target_rootYv={float(target_root_v[:, :, 1].abs().mean().detach().cpu()):.6f} "
                    f"pred_rootXZv={float(pred_root_v[:, :, [0,2]].abs().mean().detach().cpu()):.6f} "
                    f"target_rootXZv={float(target_root_v[:, :, [0,2]].abs().mean().detach().cpu()):.6f} "
                    f"pred_pelvis_v={float(pred_pelvis_v.detach().cpu()):.6f} "
                    f"target_pelvis_v={float(target_pelvis_v.detach().cpu()):.6f} "
                    f"pred_hips_v={float(pred_hips_v.detach().cpu()):.6f} "
                    f"target_hips_v={float(target_hips_v.detach().cpu()):.6f}",
                    flush=True,
                )

    return total


def install_v3c_visible_fk_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3C/V3H visible-FK patch skipped: cannot import GaussianDiffusion: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3c_visible_fk_patched", False):
        if verbose:
            print("✅ V3C/V3H visible-FK patch already installed")
        return True

    orig_motion_energy_loss = GaussianDiffusion._motion_energy_loss
    orig_p_losses = GaussianDiffusion.p_losses

    def patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        # Preserve whatever was installed before us, especially V3 temporal unit reconstruction.
        base = orig_motion_energy_loss(self, model_motion_x0, target_motion_x0)

        if not _visible_fk_enabled() and not _v3h_enabled():
            return base

        if model_motion_x0.ndim != 3 or model_motion_x0.shape[-1] != 151:
            return base

        pred_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), model_motion_x0.float())
        target_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), target_motion_x0.float())

        pred_phys = _sanitize_tensor(pred_phys)
        target_phys = _sanitize_tensor(target_phys)

        pred_pos = None
        target_pos = None
        if _visible_fk_enabled() or _v3h_enabled():
            pred_pos = _safe_fk_positions_151(pred_phys)
            target_pos = _safe_fk_positions_151(target_phys)

        denom = max(1e-6, _env_float("EDGE_V3C_LOSS_SCALE_DENOM", 0.05))
        total = base

        if _visible_fk_enabled():
            upper = _visible_upper_fk_loss(
                self,
                model_motion_x0,
                target_motion_x0,
                pred_phys=pred_phys,
                target_phys=target_phys,
                pred_pos=pred_pos,
                target_pos=target_pos,
            )
            total = total + (_env_float("EDGE_V3C_VISIBLE_FK_WEIGHT", 10.0) * upper) / denom

        if _v3h_enabled():
            support = _support_chain_fk_loss(
                self,
                model_motion_x0,
                target_motion_x0,
                pred_phys=pred_phys,
                target_phys=target_phys,
                pred_pos=pred_pos,
                target_pos=target_pos,
            )
            total = total + (_env_float("EDGE_V3H_SUPPORT_CHAIN_WEIGHT", 0.6) * support) / denom

        cap = _env_float("EDGE_V3_MOTION_ENERGY_LOSS_CAP", 2000.0)
        return torch.nan_to_num(total, nan=0.0, posinf=cap, neginf=0.0).clamp(0.0, cap)

    def patched_p_losses(self, *args, **kwargs):
        total_loss, losses = orig_p_losses(self, *args, **kwargs)

        if _stable_enabled() and _env_bool("EDGE_V3_CAP_TOTAL_LOSS", False):
            cap = _env_float("EDGE_V3_TOTAL_LOSS_CAP", 5000.0)
            total_loss = torch.nan_to_num(
                total_loss,
                nan=0.0,
                posinf=cap,
                neginf=0.0,
            ).clamp(0.0, cap)

            clean_losses = []
            loss_cap = _env_float("EDGE_V3_COMPONENT_LOSS_CAP", 5000.0)
            for item in losses:
                if torch.is_tensor(item):
                    clean_losses.append(
                        torch.nan_to_num(item, nan=0.0, posinf=loss_cap, neginf=0.0).clamp(0.0, loss_cap)
                    )
                else:
                    clean_losses.append(item)
            losses = tuple(clean_losses)

        return total_loss, losses

    GaussianDiffusion._motion_energy_loss = patched_motion_energy_loss
    GaussianDiffusion.p_losses = patched_p_losses
    GaussianDiffusion._edge_v3c_visible_fk_patched = True

    if verbose:
        print("✅ Installed STABLE V3C/V3H FK-visible support-chain patch: robust Huber + finite clamp + per-sample cap")
    return True


install_v3_visible_motion_patch = install_v3c_visible_fk_patch
install_visible_fk_patch = install_v3c_visible_fk_patch
install_v3h_support_chain_patch = install_v3c_visible_fk_patch
install_v3c_visible_fk_stable_patch = install_v3c_visible_fk_patch
