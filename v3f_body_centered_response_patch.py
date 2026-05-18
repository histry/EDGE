from __future__ import annotations

import os
import torch
import torch.nn.functional as F

_TRUE = {"1", "true", "yes", "y", "on"}
_SMPL_CACHE = {}


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


def _enabled() -> bool:
    return _env_bool("EDGE_V3F_BODY_CENTERED", False)


def _maybe_unnormalize(normalizer, x: torch.Tensor) -> torch.Tensor:
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
    if key not in _SMPL_CACHE:
        from vis import SMPLSkeleton
        _SMPL_CACHE[key] = SMPLSkeleton(device=device)
    return _SMPL_CACHE[key]


def _fk_positions_151(motion_151: torch.Tensor) -> torch.Tensor:
    if motion_151.ndim != 3 or motion_151.shape[-1] != 151:
        raise ValueError(f"expected [B,T,151], got {tuple(motion_151.shape)}")
    from dataset.quaternion import ax_from_6v
    b, t, _ = motion_151.shape
    pos = motion_151[:, :, 4:7]
    q6 = motion_151[:, :, 7:].reshape(b, t, 24, 6)
    qax = ax_from_6v(q6)
    return _get_smpl(motion_151.device).forward(qax, pos)


def _norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.amin(dim=1, keepdim=True)
    return x / x.amax(dim=1, keepdim=True).clamp_min(eps)


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [B,T]
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return (a * b).sum(dim=1)


def _safe_corr_loss(pred_a: torch.Tensor, pred_b: torch.Tensor, target_a: torch.Tensor, target_b: torch.Tensor) -> torch.Tensor:
    pred_corr = _safe_corr(pred_a, pred_b)
    target_corr = _safe_corr(target_a.detach(), target_b.detach())
    return F.mse_loss(pred_corr, target_corr)


def _range_mean(x: torch.Tensor) -> torch.Tensor:
    # x: [B,T,J,3] -> [B]
    return torch.linalg.norm(x.amax(dim=1) - x.amin(dim=1), dim=-1).mean(dim=1)


def _speed_env(x: torch.Tensor) -> torch.Tensor:
    # x: [B,T,J,3] -> [B,T-1]
    return torch.linalg.norm(x[:, 1:] - x[:, :-1], dim=-1).mean(dim=-1)


def _jerk_env(x: torch.Tensor) -> torch.Tensor:
    # x: [B,T,J,3] -> [B,T-3]
    if x.shape[1] < 4:
        return x.new_zeros((x.shape[0], 1))
    v = x[:, 1:] - x[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    j = a[:, 1:] - a[:, :-1]
    return torch.linalg.norm(j, dim=-1).mean(dim=-1)


def _unit_vec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    v = b - a
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)


def _spine_direction_features(pos_bc: torch.Tensor) -> torch.Tensor:
    # pos_bc: [B,T,24,3]
    pairs = [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
        (12, 15),
    ]
    vecs = []
    for i, j in pairs:
        vecs.append(_unit_vec(pos_bc[:, :, i], pos_bc[:, :, j]))
    return torch.stack(vecs, dim=2)  # [B,T,P,3]


def _body_centered_response_loss(self, pred_x0: torch.Tensor, target_x0: torch.Tensor) -> torch.Tensor:
    if pred_x0.ndim != 3 or pred_x0.shape[-1] != 151 or pred_x0.shape[1] < 4:
        return pred_x0.new_tensor(0.0)

    pred_phys = _maybe_unnormalize(getattr(self, "normalizer", None), pred_x0.float())
    target_phys = _maybe_unnormalize(getattr(self, "normalizer", None), target_x0.float())

    pred_pos = _fk_positions_151(pred_phys)
    target_pos = _fk_positions_151(target_phys)

    # Body-centered FK: remove global root joint position.
    pred_bc = pred_pos - pred_pos[:, :, 0:1, :]
    target_bc = target_pos - target_pos[:, :, 0:1, :]

    torso_joints = [3, 6, 9, 12, 15]
    arms_joints = [16, 17, 18, 19, 20, 21, 22, 23]
    hands_joints = [20, 21, 22, 23]
    upper_joints = torso_joints + arms_joints

    pred_torso = pred_bc[:, :, torso_joints, :]
    target_torso = target_bc[:, :, torso_joints, :]
    pred_arms = pred_bc[:, :, arms_joints, :]
    target_arms = target_bc[:, :, arms_joints, :]
    pred_hands = pred_bc[:, :, hands_joints, :]
    target_hands = target_bc[:, :, hands_joints, :]
    pred_upper = pred_bc[:, :, upper_joints, :]
    target_upper = target_bc[:, :, upper_joints, :]

    # 1. Root-relative torso range floor.
    pred_torso_range = _range_mean(pred_torso)
    target_torso_range = _range_mean(target_torso)
    torso_floor = _env_float("EDGE_V3F_TORSO_RANGE_FLOOR", 0.55)
    torso_range_loss = torch.relu(
        torso_floor * target_torso_range.detach() - pred_torso_range
    ).pow(2).mean()

    # 2. Torso / arms range ratio coupling.
    pred_arm_range = _range_mean(pred_arms).clamp_min(1e-6)
    target_arm_range = _range_mean(target_arms).detach().clamp_min(1e-6)
    pred_ratio = pred_torso_range / pred_arm_range
    target_ratio = (target_torso_range.detach() / target_arm_range).clamp(0.05, 2.0)
    ratio_loss = F.smooth_l1_loss(pred_ratio, target_ratio)

    # 3. Temporal envelope response.
    pred_torso_v = _speed_env(pred_torso)
    target_torso_v = _speed_env(target_torso)
    pred_arm_v = _speed_env(pred_arms)
    target_arm_v = _speed_env(target_arms)
    pred_hand_v = _speed_env(pred_hands)
    target_hand_v = _speed_env(target_hands)

    torso_env_loss = F.mse_loss(_norm01(pred_torso_v), _norm01(target_torso_v))
    arm_env_loss = F.mse_loss(_norm01(pred_arm_v), _norm01(target_arm_v))
    hand_env_loss = F.mse_loss(_norm01(pred_hand_v), _norm01(target_hand_v))
    response_corr_loss = _safe_corr_loss(pred_torso_v, pred_arm_v, target_torso_v, target_arm_v)

    # 4. Spine bone-vector temporal dynamics.
    pred_spine_dir = _spine_direction_features(pred_bc)
    target_spine_dir = _spine_direction_features(target_bc)
    pred_spine_v = _speed_env(pred_spine_dir)
    target_spine_v = _speed_env(target_spine_dir)
    spine_dir_env_loss = F.mse_loss(_norm01(pred_spine_v), _norm01(target_spine_v))

    # 5. FK jerk anti-jitter in body-centered space.
    pred_upper_jerk = _jerk_env(pred_upper)
    target_upper_jerk = _jerk_env(target_upper).detach()
    jerk_loss = F.smooth_l1_loss(pred_upper_jerk, target_upper_jerk)

    # 6. Root-stable / energy decomposition.
    root_xz = pred_phys[:, :, [4, 6]]
    root_v = torch.linalg.norm(root_xz[:, 1:] - root_xz[:, :-1], dim=-1)
    root_energy = root_v.pow(2).mean()

    internal_v = _speed_env(pred_upper)
    internal_energy = internal_v.pow(2).mean().detach().clamp_min(1e-8)

    max_root_ratio = _env_float("EDGE_V3F_MAX_ROOT_ENERGY_RATIO", 0.35)
    root_ratio_loss = torch.relu(root_energy - max_root_ratio * internal_energy).pow(2)
    root_stable_loss = root_energy

    loss = (
        _env_float("EDGE_V3F_TORSO_RANGE_WEIGHT", 12.0) * torso_range_loss
        + _env_float("EDGE_V3F_TORSO_ARM_RATIO_WEIGHT", 3.0) * ratio_loss
        + _env_float("EDGE_V3F_TORSO_ENV_WEIGHT", 2.0) * torso_env_loss
        + _env_float("EDGE_V3F_RESPONSE_WEIGHT", 1.0) * response_corr_loss
        + _env_float("EDGE_V3F_ARM_ENV_WEIGHT", 0.5) * arm_env_loss
        + _env_float("EDGE_V3F_HAND_ENV_WEIGHT", 0.25) * hand_env_loss
        + _env_float("EDGE_V3F_SPINE_DIR_WEIGHT", 1.0) * spine_dir_env_loss
        + _env_float("EDGE_V3F_JERK_WEIGHT", 0.5) * jerk_loss
        + _env_float("EDGE_V3F_ROOT_STABLE_WEIGHT", 0.5) * root_stable_loss
        + _env_float("EDGE_V3F_ROOT_RATIO_WEIGHT", 1.0) * root_ratio_loss
    )

    if _env_bool("EDGE_V3F_DEBUG", False):
        if not hasattr(self, "_edge_v3f_debug_counter"):
            self._edge_v3f_debug_counter = 0
        self._edge_v3f_debug_counter += 1
        if self._edge_v3f_debug_counter <= 20 or self._edge_v3f_debug_counter % 100 == 0:
            print(
                "🧪 V3F body-centered | "
                f"torso_range={float(pred_torso_range.mean().detach().cpu()):.6f} "
                f"target_torso_range={float(target_torso_range.mean().detach().cpu()):.6f} "
                f"ratio={float(pred_ratio.mean().detach().cpu()):.6f} "
                f"target_ratio={float(target_ratio.mean().detach().cpu()):.6f} "
                f"torso_range_loss={float(torso_range_loss.detach().cpu()):.6f} "
                f"ratio_loss={float(ratio_loss.detach().cpu()):.6f} "
                f"torso_env={float(torso_env_loss.detach().cpu()):.6f} "
                f"response={float(response_corr_loss.detach().cpu()):.6f} "
                f"spine_env={float(spine_dir_env_loss.detach().cpu()):.6f} "
                f"jerk={float(jerk_loss.detach().cpu()):.6f} "
                f"root_energy={float(root_energy.detach().cpu()):.8f} "
                f"root_ratio={float(root_ratio_loss.detach().cpu()):.8f}",
                flush=True,
            )

    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)


def install_v3f_body_centered_response_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3F patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3f_body_centered_patched", False):
        if verbose:
            print("✅ V3F body-centered response patch already installed")
        return True

    orig_motion_energy_loss = GaussianDiffusion._motion_energy_loss

    def patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        base = orig_motion_energy_loss(self, model_motion_x0, target_motion_x0)
        if not _enabled():
            return base

        # IMPORTANT:
        # model/diffusion.py multiplies motion_energy_loss by physical_w * 0.05.
        # V3C uses the same convention. Therefore we divide by 0.05 here so that
        # EDGE_V3F_WEIGHT is the effective physical loss scale.
        v3f_loss = _body_centered_response_loss(self, model_motion_x0, target_motion_x0)
        return base + (_env_float("EDGE_V3F_WEIGHT", 1.0) * v3f_loss) / 0.05

    GaussianDiffusion._motion_energy_loss = patched_motion_energy_loss
    GaussianDiffusion._edge_v3f_body_centered_patched = True

    if verbose:
        print("✅ Installed V3F body-centered torso-response patch")
    return True
