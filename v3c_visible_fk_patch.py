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


def _maybe_unnormalize_tensor(normalizer: Any, x: torch.Tensor) -> torch.Tensor:
    if normalizer is None or not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
        return x
    mean = torch.as_tensor(normalizer.mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(normalizer.std, device=x.device, dtype=x.dtype)
    while mean.ndim < x.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return x * std + mean


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
ROOT_XZ = [4, 6]
ROOT_Y = [5]

PELVIS_ROT = _rot_dims([0])
HIPS_ROT = _rot_dims([1, 2])
SPINE_TORSO_ROT = _rot_dims([3, 6, 9])
KNEES_ROT = _rot_dims([4, 5])
ANKLES_FEET_ROT = _rot_dims([7, 8, 10, 11])
NECK_HEAD_ROT = _rot_dims([12, 15])
ARMS_HANDS_ROT = _rot_dims([13, 14, 16, 17, 18, 19, 20, 21, 22, 23])
UPPER_SAFE_ROT = SPINE_TORSO_ROT + NECK_HEAD_ROT + ARMS_HANDS_ROT
SUPPORT_ROT = PELVIS_ROT + HIPS_ROT + KNEES_ROT + ANKLES_FEET_ROT

# SMPL joint ids used in FK-space losses.
PELVIS_JOINTS = [0]
HIP_JOINTS = [1, 2]
KNEE_JOINTS = [4, 5]
ANKLE_FOOT_JOINTS = [7, 8, 10, 11]
LOWER_SUPPORT_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
FEET_JOINTS = [10, 11]


def _safe_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_tensor(0.0)
    return F.mse_loss(a, b)


def _vel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 2:
        return x[:, :0]
    return x[:, 1:] - x[:, :-1]


def _acc(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return x[:, :0]
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def _range_floor_loss(pred: torch.Tensor, target: torch.Tensor, floor: float) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_range = torch.linalg.norm(pred.amax(dim=1) - pred.amin(dim=1), dim=-1).mean(dim=-1)
    target_range = torch.linalg.norm(target.amax(dim=1) - target.amin(dim=1), dim=-1).mean(dim=-1)
    return torch.relu(float(floor) * target_range.detach() - pred_range).pow(2).mean()


def _activity_floor_loss(pred_v_norm: torch.Tensor, target_v_norm: torch.Tensor, floor: float) -> torch.Tensor:
    pred_activity = pred_v_norm.mean(dim=tuple(range(1, pred_v_norm.ndim)))
    target_activity = target_v_norm.mean(dim=tuple(range(1, target_v_norm.ndim)))
    return torch.relu(float(floor) * target_activity.detach() - pred_activity).pow(2).mean()


def _upper_safe_rot_features(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 151:
        return x
    return x[..., UPPER_SAFE_ROT]


def _optional_upper_rot_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    if not _env_bool("EDGE_V3C_UPPER_ROT_LOSS", False):
        return pred_phys.new_tensor(0.0)

    pred = _upper_safe_rot_features(pred_phys.float())
    target = _upper_safe_rot_features(target_phys.float())

    loss = _env_float("EDGE_V3C_UPPER_ROT_WEIGHT", 0.05) * _safe_mse(pred, target)

    if pred.shape[1] > 1:
        loss = loss + _env_float("EDGE_V3C_UPPER_ROT_VEL_WEIGHT", 0.10) * _safe_mse(
            pred[:, 1:] - pred[:, :-1],
            target[:, 1:] - target[:, :-1],
        )

    dct_w = _env_float("EDGE_V3C_UPPER_ROT_DCT_WEIGHT", 0.0)
    if dct_w > 0:
        keep = _env_int("EDGE_V3C_UPPER_ROT_DCT_KEEP", 8)
        loss = loss + dct_w * _safe_mse(_dct_lowpass(pred, keep), _dct_lowpass(target, keep))

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
    if pred_pos is None:
        pred_pos = _fk_positions_151(pred_phys)
    if target_pos is None:
        target_pos = _fk_positions_151(target_phys)

    joint_mode = os.environ.get("EDGE_V3C_VISIBLE_JOINTS", "upper_safe_plus").strip().lower()
    if joint_mode in {"arms", "hands"}:
        upper_joints = [16, 17, 18, 19, 20, 21, 22, 23]
    else:
        upper_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    pred_u = pred_pos[:, :, upper_joints, :]
    target_u = target_pos[:, :, upper_joints, :]

    pred_v = torch.linalg.norm(pred_u[:, 1:] - pred_u[:, :-1], dim=-1)
    target_v = torch.linalg.norm(target_u[:, 1:] - target_u[:, :-1], dim=-1)

    speed_loss = _safe_mse(pred_v, target_v)

    pred_env = pred_v.mean(dim=-1)
    target_env = target_v.mean(dim=-1)
    pred_env_n = pred_env / pred_env.amax(dim=1, keepdim=True).clamp_min(1e-6)
    target_env_n = target_env / target_env.amax(dim=1, keepdim=True).clamp_min(1e-6)
    envelope_loss = _safe_mse(pred_env_n, target_env_n)

    activity_floor_loss = _activity_floor_loss(
        pred_v,
        target_v,
        _env_float("EDGE_V3C_ACTIVITY_FLOOR", 0.70),
    )

    pred_range = torch.linalg.norm(pred_u.amax(dim=1) - pred_u.amin(dim=1), dim=-1).mean(dim=1)
    target_range = torch.linalg.norm(target_u.amax(dim=1) - target_u.amin(dim=1), dim=-1).mean(dim=1)
    range_floor = _env_float("EDGE_V3C_RANGE_FLOOR", 0.65)
    range_floor_loss = torch.relu(range_floor * target_range.detach() - pred_range).pow(2).mean()

    hand_joints = [20, 21, 22, 23]
    pred_h = pred_pos[:, :, hand_joints, :]
    target_h = target_pos[:, :, hand_joints, :]
    pred_hv = torch.linalg.norm(pred_h[:, 1:] - pred_h[:, :-1], dim=-1)
    target_hv = torch.linalg.norm(target_h[:, 1:] - target_h[:, :-1], dim=-1)
    hand_speed_loss = _safe_mse(pred_hv, target_hv)

    pred_hrange = torch.linalg.norm(pred_h.amax(dim=1) - pred_h.amin(dim=1), dim=-1).mean(dim=1)
    target_hrange = torch.linalg.norm(target_h.amax(dim=1) - target_h.amin(dim=1), dim=-1).mean(dim=1)
    hand_range_floor_loss = torch.relu(range_floor * target_hrange.detach() - pred_hrange).pow(2).mean()

    loss = (
        _env_float("EDGE_V3C_FK_SPEED_WEIGHT", 1.00) * speed_loss
        + _env_float("EDGE_V3C_FK_ENVELOPE_WEIGHT", 0.50) * envelope_loss
        + _env_float("EDGE_V3C_FK_ACTIVITY_WEIGHT", 3.00) * activity_floor_loss
        + _env_float("EDGE_V3C_FK_RANGE_WEIGHT", 2.00) * range_floor_loss
        + _env_float("EDGE_V3C_FK_HAND_WEIGHT", 1.00) * hand_speed_loss
        + _env_float("EDGE_V3C_FK_HAND_RANGE_WEIGHT", 2.00) * hand_range_floor_loss
    )

    loss = loss + _optional_upper_rot_loss(pred_phys, target_phys)

    if _env_bool("EDGE_V3C_VISIBLE_FK_DEBUG", False):
        if not hasattr(self, "_edge_v3c_visible_debug_counter"):
            self._edge_v3c_visible_debug_counter = 0
        self._edge_v3c_visible_debug_counter += 1
        if self._edge_v3c_visible_debug_counter <= 20 or self._edge_v3c_visible_debug_counter % 100 == 0:
            print(
                "🧪 V3C visible-FK | "
                f"upper_speed={float(speed_loss.detach().cpu()):.6f} "
                f"upper_env={float(envelope_loss.detach().cpu()):.6f} "
                f"upper_act_floor={float(activity_floor_loss.detach().cpu()):.6f} "
                f"upper_range_floor={float(range_floor_loss.detach().cpu()):.6f} "
                f"hand_speed={float(hand_speed_loss.detach().cpu()):.6f} "
                f"hand_range_floor={float(hand_range_floor_loss.detach().cpu()):.6f}",
                flush=True,
            )

    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)


def _lower_rot_velocity_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    if pred_phys.shape[-1] != 151 or pred_phys.shape[1] < 2:
        return pred_phys.new_tensor(0.0)

    loss = pred_phys.new_tensor(0.0)

    def add_rot_vel(name: str, dims: List[int], default_w: float) -> None:
        nonlocal loss
        w = _env_float(name, default_w)
        if w <= 0:
            return
        pred = pred_phys[:, :, dims]
        target = target_phys[:, :, dims]
        loss = loss + w * _safe_mse(_vel(pred), _vel(target))

    def add_rot_abs(name: str, dims: List[int], default_w: float) -> None:
        nonlocal loss
        w = _env_float(name, default_w)
        if w <= 0:
            return
        pred = pred_phys[:, :, dims]
        target = target_phys[:, :, dims]
        loss = loss + w * _safe_mse(pred, target)

    # Absolute pelvis/hips are style-bearing but should be moderate.
    add_rot_abs("EDGE_V3H_PELVIS_ABS_WEIGHT", PELVIS_ROT, 0.04)
    add_rot_abs("EDGE_V3H_HIPS_ABS_WEIGHT", HIPS_ROT, 0.03)

    # Velocity terms carry support-chain dynamics.
    add_rot_vel("EDGE_V3H_PELVIS_ROT_VEL_WEIGHT", PELVIS_ROT, 0.30)
    add_rot_vel("EDGE_V3H_HIPS_ROT_VEL_WEIGHT", HIPS_ROT, 0.28)
    add_rot_vel("EDGE_V3H_KNEES_ROT_VEL_WEIGHT", KNEES_ROT, 0.20)
    add_rot_vel("EDGE_V3H_ANKLES_FEET_ROT_VEL_WEIGHT", ANKLES_FEET_ROT, 0.16)

    dct_w = _env_float("EDGE_V3H_SUPPORT_ROT_DCT_WEIGHT", 0.0)
    if dct_w > 0:
        keep = _env_int("EDGE_V3H_SUPPORT_ROT_DCT_KEEP", 8)
        pred = pred_phys[:, :, SUPPORT_ROT]
        target = target_phys[:, :, SUPPORT_ROT]
        loss = loss + dct_w * _safe_mse(_dct_lowpass(pred, keep), _dct_lowpass(target, keep))

    return loss


def _root_support_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    if pred_phys.shape[-1] != 151 or pred_phys.shape[1] < 2:
        return pred_phys.new_tensor(0.0)

    pred_root = pred_phys[:, :, 4:7]
    target_root = target_phys[:, :, 4:7]

    pred_root_v = _vel(pred_root)
    target_root_v = _vel(target_root)

    root_y_v_loss = _safe_mse(pred_root_v[:, :, 1:2], target_root_v[:, :, 1:2])
    root_xz_v_loss = _safe_mse(
        pred_root_v[:, :, [0, 2]],
        target_root_v[:, :, [0, 2]],
    )

    # Absolute root-Y helps preserve Dunhuang sinking / rising center;
    # root-XZ absolute locking is intentionally absent.
    root_y_abs_loss = _safe_mse(pred_root[:, :, 1:2], target_root[:, :, 1:2])

    loss = (
        _env_float("EDGE_V3H_ROOT_Y_VEL_WEIGHT", 0.80) * root_y_v_loss
        + _env_float("EDGE_V3H_ROOT_XZ_VEL_WEIGHT", 0.50) * root_xz_v_loss
        + _env_float("EDGE_V3H_ROOT_Y_ABS_WEIGHT", 0.05) * root_y_abs_loss
    )

    return loss


def _lower_fk_support_loss(pred_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    if pred_pos.shape[1] < 2:
        return pred_pos.new_tensor(0.0)

    pred_lower = pred_pos[:, :, LOWER_SUPPORT_JOINTS, :]
    target_lower = target_pos[:, :, LOWER_SUPPORT_JOINTS, :]

    pred_lower_v = torch.linalg.norm(pred_lower[:, 1:] - pred_lower[:, :-1], dim=-1)
    target_lower_v = torch.linalg.norm(target_lower[:, 1:] - target_lower[:, :-1], dim=-1)

    speed_loss = _safe_mse(pred_lower_v, target_lower_v)
    activity_loss = _activity_floor_loss(
        pred_lower_v,
        target_lower_v,
        _env_float("EDGE_V3H_LOWER_ACTIVITY_FLOOR", 0.60),
    )
    range_loss = _range_floor_loss(
        pred_lower,
        target_lower,
        _env_float("EDGE_V3H_LOWER_RANGE_FLOOR", 0.55),
    )

    pred_feet = pred_pos[:, :, FEET_JOINTS, :]
    target_feet = target_pos[:, :, FEET_JOINTS, :]

    pred_feet_v = torch.linalg.norm(pred_feet[:, 1:] - pred_feet[:, :-1], dim=-1)
    target_feet_v = torch.linalg.norm(target_feet[:, 1:] - target_feet[:, :-1], dim=-1)
    foot_speed_loss = _safe_mse(pred_feet_v, target_feet_v)

    # Foot height consistency: y-axis of FK positions.
    foot_height_loss = _safe_mse(pred_feet[:, :, :, 1], target_feet[:, :, :, 1])

    foot_range_loss = _range_floor_loss(
        pred_feet,
        target_feet,
        _env_float("EDGE_V3H_FOOT_RANGE_FLOOR", 0.50),
    )

    loss = (
        _env_float("EDGE_V3H_LOWER_FK_SPEED_WEIGHT", 1.00) * speed_loss
        + _env_float("EDGE_V3H_LOWER_FK_ACTIVITY_WEIGHT", 2.00) * activity_loss
        + _env_float("EDGE_V3H_LOWER_FK_RANGE_WEIGHT", 1.50) * range_loss
        + _env_float("EDGE_V3H_FOOT_FK_SPEED_WEIGHT", 1.00) * foot_speed_loss
        + _env_float("EDGE_V3H_FOOT_HEIGHT_WEIGHT", 0.60) * foot_height_loss
        + _env_float("EDGE_V3H_FOOT_RANGE_WEIGHT", 1.00) * foot_range_loss
    )

    return loss


def _contact_support_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor, pred_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    if pred_phys.shape[-1] != 151 or pred_phys.shape[1] < 2:
        return pred_phys.new_tensor(0.0)

    pred_c = pred_phys[:, :, CONTACTS]
    target_c = target_phys[:, :, CONTACTS]

    # Contact reconstruction and support phase switching.
    contact_abs = _safe_mse(pred_c, target_c)
    contact_switch = _safe_mse(_vel(pred_c), _vel(target_c))

    # Contact channels should correlate with lower foot velocity.
    # When target contact is high, generated foot velocity should not explode.
    pred_feet = pred_pos[:, :, FEET_JOINTS, :]
    pred_feet_v = torch.linalg.norm(pred_feet[:, 1:] - pred_feet[:, :-1], dim=-1)

    target_contact = target_c
    # Approximate left/right contact from 4 contact channels.
    left_contact = torch.maximum(target_contact[:, :, 0], target_contact[:, :, 2])
    right_contact = torch.maximum(target_contact[:, :, 1], target_contact[:, :, 3])
    target_contact_lr = torch.stack([left_contact, right_contact], dim=-1)
    target_contact_lr_v = 0.5 * (target_contact_lr[:, 1:] + target_contact_lr[:, :-1]).detach()

    contact_foot_vel = (target_contact_lr_v * pred_feet_v).mean()

    # Optional foot height consistency under target contact.
    pred_foot_y = pred_feet[:, :, :, 1]
    target_foot_y = target_pos[:, :, FEET_JOINTS, 1]
    contact_y = (target_contact_lr.detach() * (pred_foot_y - target_foot_y).pow(2)).mean()

    loss = (
        _env_float("EDGE_V3H_CONTACT_RECON_WEIGHT", 0.10) * contact_abs
        + _env_float("EDGE_V3H_CONTACT_SWITCH_WEIGHT", 0.30) * contact_switch
        + _env_float("EDGE_V3H_CONTACT_FOOT_VEL_WEIGHT", 0.10) * contact_foot_vel
        + _env_float("EDGE_V3H_CONTACT_FOOT_HEIGHT_WEIGHT", 0.10) * contact_y
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
    if pred_pos is None:
        pred_pos = _fk_positions_151(pred_phys)
    if target_pos is None:
        target_pos = _fk_positions_151(target_phys)

    root_loss = _root_support_loss(pred_phys, target_phys)
    rot_loss = _lower_rot_velocity_loss(pred_phys, target_phys)
    fk_loss = _lower_fk_support_loss(pred_pos, target_pos)
    contact_loss = _contact_support_loss(pred_phys, target_phys, pred_pos, target_pos)

    # Optional low-pass support-chain reconstruction for smooth lower-body transfer.
    dct_fk_w = _env_float("EDGE_V3H_LOWER_FK_DCT_WEIGHT", 0.0)
    dct_fk_loss = pred_x0.new_tensor(0.0)
    if dct_fk_w > 0:
        keep = _env_int("EDGE_V3H_LOWER_FK_DCT_KEEP", 8)
        pred_lower = pred_pos[:, :, LOWER_SUPPORT_JOINTS, :].reshape(pred_pos.shape[0], pred_pos.shape[1], -1)
        target_lower = target_pos[:, :, LOWER_SUPPORT_JOINTS, :].reshape(target_pos.shape[0], target_pos.shape[1], -1)
        dct_fk_loss = dct_fk_w * _safe_mse(_dct_lowpass(pred_lower, keep), _dct_lowpass(target_lower, keep))

    loss = root_loss + rot_loss + fk_loss + contact_loss + dct_fk_loss

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
                    "🧪 V3H support-chain | "
                    f"root={float(root_loss.detach().cpu()):.6f} "
                    f"rot={float(rot_loss.detach().cpu()):.6f} "
                    f"fk={float(fk_loss.detach().cpu()):.6f} "
                    f"contact={float(contact_loss.detach().cpu()):.6f} "
                    f"dct={float(dct_fk_loss.detach().cpu()):.6f} "
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

    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)


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

    def patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        if not _visible_fk_enabled() and not _v3h_enabled():
            return orig_motion_energy_loss(self, model_motion_x0, target_motion_x0)

        if model_motion_x0.ndim != 3 or model_motion_x0.shape[-1] != 151:
            return orig_motion_energy_loss(self, model_motion_x0, target_motion_x0)

        pred_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), model_motion_x0.float())
        target_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), target_motion_x0.float())

        # FK is relatively expensive, so compute once and reuse.
        pred_pos = _fk_positions_151(pred_phys)
        target_pos = _fk_positions_151(target_phys)

        total = model_motion_x0.new_tensor(0.0)

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
            total = total + _env_float("EDGE_V3C_VISIBLE_FK_WEIGHT", 10.0) * upper

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
            total = total + _env_float("EDGE_V3H_SUPPORT_CHAIN_WEIGHT", 4.0) * support

        # Match existing V3C scaling so current training scripts remain comparable.
        denom = _env_float("EDGE_V3C_LOSS_SCALE_DENOM", 0.05)
        return total / max(1e-6, denom)

    GaussianDiffusion._motion_energy_loss = patched_motion_energy_loss
    GaussianDiffusion._edge_v3c_visible_fk_patched = True

    if verbose:
        print("✅ Installed V3C/V3H FK-visible support-chain motion patch")
    return True


install_v3_visible_motion_patch = install_v3c_visible_fk_patch
install_visible_fk_patch = install_v3c_visible_fk_patch
install_v3h_support_chain_patch = install_v3c_visible_fk_patch
