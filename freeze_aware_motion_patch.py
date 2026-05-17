
from __future__ import annotations

"""
V2F burst-safe temporal-progress patch for EDGE.

V2E can reward-hack progress/energy losses by using a few rapid twitches.
V2F blocks that shortcut with:
1) saturated progress energy (tanh/clamp);
2) acceleration/jerk smoothness and spike penalties;
3) directional consistency;
4) low-pass temporal prior.
"""

import os
import torch
import torch.nn.functional as F

_TRUE = {"1", "true", "yes", "y", "on"}

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return bool(default) if v is None else str(v).strip().lower() in _TRUE

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)

def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=dim) + eps)

def _rot_indices(joints):
    out = []
    for joint in joints:
        s = 7 + 6 * int(joint)
        out.extend(range(s, s + 6))
    return out

UPPER_JOINTS = list(range(12, 24))
TORSO_JOINTS = [3, 6, 9, 12, 13, 14, 15]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
UPPER_IDX = _rot_indices(UPPER_JOINTS)
TORSO_IDX = _rot_indices(TORSO_JOINTS)
LOWER_IDX = _rot_indices(LOWER_JOINTS)
UPPER_TORSO_IDX = sorted(set(UPPER_IDX + TORSO_IDX))

def _feature_indices(device, mode: str) -> torch.Tensor:
    mode = str(mode or "upper_torso").strip().lower()
    if mode in {"all", "full", "body"}:
        idx = list(range(7, 151))
    elif mode in {"upper", "arms"}:
        idx = UPPER_IDX
    elif mode == "torso":
        idx = TORSO_IDX
    elif mode in {"upper_torso", "torso_upper"}:
        idx = UPPER_TORSO_IDX
    elif mode in {"lower", "legs"}:
        idx = LOWER_IDX
    else:
        idx = UPPER_TORSO_IDX
    return torch.as_tensor(idx, device=device, dtype=torch.long)

def _frame_energy(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 2:
        return x.new_zeros((x.shape[0], 0))
    return _safe_norm(x[:, 1:] - x[:, :-1], dim=-1)

def _target_adaptive_cap(target_energy: torch.Tensor) -> torch.Tensor:
    if target_energy.shape[1] == 0:
        return target_energy.new_ones((target_energy.shape[0], 1))
    mean = target_energy.detach().mean(dim=1, keepdim=True)
    mult = _env_float("EDGE_PROGRESS_CAP_TARGET_MEAN_MULT", 0.75)
    floor = _env_float("EDGE_PROGRESS_ENERGY_CAP_FLOOR", 0.05)
    return (mean * mult).clamp_min(floor)

def _saturate_energy(energy: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    if not _env_bool("EDGE_BURST_SAFE_PROGRESS", True):
        return energy
    cap = cap.to(device=energy.device, dtype=energy.dtype).clamp_min(1e-6)
    mode = os.environ.get("EDGE_PROGRESS_SATURATION_MODE", "tanh").strip().lower()
    if mode == "clamp":
        return torch.minimum(energy, cap)
    if mode == "sqrt":
        return cap * torch.sqrt(energy / cap + 1e-8)
    return cap * torch.tanh(energy / cap)

def _cumulative_progress(energy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if energy.shape[1] == 0:
        return energy
    total = energy.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.cumsum(energy, dim=1) / total

def _topk_share(energy: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    if energy.shape[1] == 0:
        return energy.new_zeros((energy.shape[0],))
    k = max(1, min(int(k), energy.shape[1]))
    vals = torch.topk(energy, k=k, dim=1, largest=True).values
    return vals.sum(dim=1) / energy.sum(dim=1).clamp_min(eps)

def _temporal_progress_loss(pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    if pred_feat.shape[1] < 3:
        return pred_feat.new_tensor(0.0)
    eps = 1e-8
    B, T, _ = pred_feat.shape
    target_start = target_feat[:, :1, :]
    target_end = target_feat[:, -1:, :]
    pred_dist_end = _safe_norm(pred_feat - target_end, dim=-1)
    target_dist_end = _safe_norm(target_feat - target_end, dim=-1)
    start_end = _safe_norm(target_start - target_end, dim=-1).clamp_min(eps)
    pred_dist_norm = pred_dist_end / start_end
    target_dist_norm = target_dist_end / start_end

    curve_mode = os.environ.get("EDGE_PROGRESS_CURVE", "gt").strip().lower()
    if curve_mode == "linear":
        target_curve = torch.linspace(1.0, 0.0, steps=T, device=pred_feat.device, dtype=pred_feat.dtype).view(1, T).expand(B, -1)
    elif curve_mode == "cosine":
        u = torch.linspace(0.0, 1.0, steps=T, device=pred_feat.device, dtype=pred_feat.dtype).view(1, T)
        target_curve = (0.5 * (1.0 + torch.cos(torch.pi * u))).expand(B, -1)
    elif curve_mode == "hybrid":
        u = torch.linspace(0.0, 1.0, steps=T, device=pred_feat.device, dtype=pred_feat.dtype).view(1, T)
        linear = 1.0 - u
        mix = _env_float("EDGE_PROGRESS_HYBRID_GT_WEIGHT", 0.7)
        target_curve = mix * target_dist_norm.detach() + (1.0 - mix) * linear
    else:
        target_curve = target_dist_norm.detach()

    distance_loss = F.smooth_l1_loss(pred_dist_norm, target_curve)
    pred_energy_raw = _frame_energy(pred_feat)
    target_energy_raw = _frame_energy(target_feat).detach()
    cap = _target_adaptive_cap(target_energy_raw)
    pred_energy = _saturate_energy(pred_energy_raw, cap)
    target_energy = _saturate_energy(target_energy_raw, cap).detach()

    pred_cum = _cumulative_progress(pred_energy)
    target_cum = _cumulative_progress(target_energy).detach()
    cumsum_loss = F.smooth_l1_loss(pred_cum, target_cum) if pred_cum.numel() else pred_feat.new_tensor(0.0)

    margin = _env_float("EDGE_PROGRESS_FRONTLOAD_MARGIN", 0.08)
    front_loss = pred_feat.new_tensor(0.0)
    if pred_cum.numel():
        for frac in (0.25, 0.50):
            idx = min(pred_cum.shape[1] - 1, max(0, int(round(frac * (pred_cum.shape[1] - 1)))))
            front_loss = front_loss + torch.relu(pred_cum[:, idx] - target_cum[:, idx] - margin).pow(2).mean()

    topk_loss = pred_feat.new_tensor(0.0)
    for k, default_margin in [(1, 0.04), (3, 0.07), (5, 0.10)]:
        pred_share = _topk_share(pred_energy, k=k)
        target_share = _topk_share(target_energy, k=k).detach()
        k_margin = _env_float(f"EDGE_PROGRESS_TOP{k}_MARGIN", default_margin)
        topk_loss = topk_loss + torch.relu(pred_share - target_share - k_margin).pow(2).mean()

    early_until = max(1, int(round((T - 1) * _env_float("EDGE_PROGRESS_EARLY_UNTIL_FRAC", 0.65))))
    target_min = target_curve[:, :early_until] * _env_float("EDGE_PROGRESS_EARLY_MIN_FRACTION", 0.55)
    early_loss = torch.relu(target_min - pred_dist_norm[:, :early_until]).pow(2).mean()

    return (
        _env_float("EDGE_PROGRESS_DISTANCE_WEIGHT", 1.0) * distance_loss
        + _env_float("EDGE_PROGRESS_CUM_WEIGHT", 1.0) * cumsum_loss
        + _env_float("EDGE_PROGRESS_FRONTLOAD_WEIGHT", 1.0) * front_loss
        + _env_float("EDGE_PROGRESS_TOPK_WEIGHT", 1.0) * topk_loss
        + _env_float("EDGE_PROGRESS_EARLY_WEIGHT", 1.0) * early_loss
    )

def _kinematic_smoothness_loss(pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    if pred_feat.shape[1] < 4:
        return pred_feat.new_tensor(0.0)
    pred_v = pred_feat[:, 1:] - pred_feat[:, :-1]
    tgt_v = (target_feat[:, 1:] - target_feat[:, :-1]).detach()
    pred_a = pred_v[:, 1:] - pred_v[:, :-1]
    tgt_a = (tgt_v[:, 1:] - tgt_v[:, :-1]).detach()
    pred_j = pred_a[:, 1:] - pred_a[:, :-1]
    tgt_j = (tgt_a[:, 1:] - tgt_a[:, :-1]).detach()

    accel_match = F.smooth_l1_loss(pred_a, tgt_a)
    jerk_match = F.smooth_l1_loss(pred_j, tgt_j) if pred_j.numel() else pred_feat.new_tensor(0.0)

    pred_a_norm = _safe_norm(pred_a, dim=-1)
    tgt_a_norm = _safe_norm(tgt_a, dim=-1).detach()
    if pred_j.numel():
        pred_j_norm = _safe_norm(pred_j, dim=-1)
        tgt_j_norm = _safe_norm(tgt_j, dim=-1).detach()
    else:
        pred_j_norm = pred_feat.new_zeros((pred_feat.shape[0], 0))
        tgt_j_norm = pred_feat.new_zeros((pred_feat.shape[0], 0))

    accel_spike = torch.relu(pred_a_norm - (tgt_a_norm * _env_float("EDGE_ACCEL_SPIKE_MAX_RATIO", 1.7) + _env_float("EDGE_ACCEL_SPIKE_FLOOR", 0.02))).pow(2).mean()
    jerk_spike = torch.relu(pred_j_norm - (tgt_j_norm * _env_float("EDGE_JERK_SPIKE_MAX_RATIO", 1.5) + _env_float("EDGE_JERK_SPIKE_FLOOR", 0.02))).pow(2).mean() if pred_j_norm.numel() else pred_feat.new_tensor(0.0)

    return (
        _env_float("EDGE_ACCEL_MATCH_WEIGHT", 1.0) * accel_match
        + _env_float("EDGE_JERK_MATCH_WEIGHT", 1.5) * jerk_match
        + _env_float("EDGE_ACCEL_SPIKE_WEIGHT", 1.0) * accel_spike
        + _env_float("EDGE_JERK_SPIKE_WEIGHT", 2.0) * jerk_spike
    )

def _directional_consistency_loss(pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    if pred_feat.shape[1] < 3:
        return pred_feat.new_tensor(0.0)
    pred_v = pred_feat[:, 1:] - pred_feat[:, :-1]
    tgt_v = (target_feat[:, 1:] - target_feat[:, :-1]).detach()
    pred_v0, pred_v1 = pred_v[:, :-1], pred_v[:, 1:]
    tgt_v0, tgt_v1 = tgt_v[:, :-1], tgt_v[:, 1:]
    pred_n0, pred_n1 = _safe_norm(pred_v0, dim=-1), _safe_norm(pred_v1, dim=-1)
    tgt_n0, tgt_n1 = _safe_norm(tgt_v0, dim=-1), _safe_norm(tgt_v1, dim=-1)
    pred_cos = torch.sum(pred_v0 * pred_v1, dim=-1) / (pred_n0 * pred_n1 + 1e-8)
    tgt_cos = torch.sum(tgt_v0 * tgt_v1, dim=-1) / (tgt_n0 * tgt_n1 + 1e-8)

    thr = (tgt_n0.mean(dim=1, keepdim=True) * _env_float("EDGE_DIRECTION_TARGET_THR_SCALE", 0.15)).clamp_min(1e-5)
    mask = ((tgt_n0 > thr) & (tgt_n1 > thr)).to(pred_feat.dtype)
    reversal = torch.relu(-pred_cos).pow(2) * mask
    deviation = torch.relu((tgt_cos - pred_cos) - _env_float("EDGE_DIRECTION_COS_MARGIN", 0.25)).pow(2) * mask
    denom = mask.sum().clamp_min(1.0)
    return (reversal.sum() + _env_float("EDGE_DIRECTION_TARGET_MATCH_WEIGHT", 0.5) * deviation.sum()) / denom

def _temporal_smooth(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if x.shape[1] < 3:
        return x
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    if k == 3:
        vals = [0.25, 0.50, 0.25]
    elif k == 5:
        vals = [0.0625, 0.25, 0.375, 0.25, 0.0625]
    else:
        vals = [1.0 / k] * k
    kernel = torch.tensor(vals, device=x.device, dtype=x.dtype).view(1, 1, k)
    B, T, C = x.shape
    y = x.permute(0, 2, 1).reshape(B * C, 1, T)
    pad = k // 2
    y = F.pad(y, (pad, pad), mode="replicate")
    y = F.conv1d(y, kernel)
    return y.reshape(B, C, T).permute(0, 2, 1)

def _lowpass_temporal_loss(pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    if pred_feat.shape[1] < 5:
        return pred_feat.new_tensor(0.0)
    k = _env_int("EDGE_LOWPASS_KERNEL_SIZE", 5)
    pred_smooth = _temporal_smooth(pred_feat, kernel_size=k)
    tgt_smooth = _temporal_smooth(target_feat.detach(), kernel_size=k)
    pred_high = pred_feat - pred_smooth
    tgt_high = target_feat.detach() - tgt_smooth
    high_match = F.smooth_l1_loss(pred_high, tgt_high)
    pred_high_energy = _safe_norm(pred_high, dim=-1).mean(dim=1)
    tgt_high_energy = _safe_norm(tgt_high, dim=-1).mean(dim=1).detach().clamp_min(1e-8)
    high_ratio = pred_high_energy / tgt_high_energy
    high_over = torch.relu(high_ratio - _env_float("EDGE_LOWPASS_HIGH_MAX_RATIO", 1.3)).pow(2).mean()
    return _env_float("EDGE_LOWPASS_MATCH_WEIGHT", 1.0) * high_match + _env_float("EDGE_LOWPASS_OVER_WEIGHT", 1.0) * high_over

def install_freeze_aware_motion_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V2F burst-safe patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v2f_burstsafe_patch_installed", False):
        if verbose:
            print("✅ V2F burst-safe motion patch already installed.")
        return True

    original_anti_freeze_loss = GaussianDiffusion._anti_freeze_loss
    original_motion_energy_loss = GaussianDiffusion._motion_energy_loss

    def _patched_anti_freeze_loss(self, model_motion_x0):
        if not _env_bool("EDGE_FREEZE_AWARE_MOTION", False):
            return original_anti_freeze_loss(self, model_motion_x0)
        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)
        idx = _feature_indices(model_motion_x0.device, os.environ.get("EDGE_FREEZE_AWARE_FEATURE_MODE", "upper_torso"))
        frame_energy = _frame_energy(model_motion_x0[..., idx])
        mean_energy = frame_energy.mean() if frame_energy.numel() else model_motion_x0.new_tensor(0.0)
        mean_loss = torch.relu(model_motion_x0.new_tensor(_env_float("EDGE_ANTI_FREEZE_MIN_ENERGY", 0.015)) - mean_energy)
        active_thr = _env_float("EDGE_ANTI_FREEZE_ACTIVE_THR", 0.010)
        active = torch.sigmoid((frame_energy - active_thr) / max(active_thr * 0.5, 1e-6))
        active_ratio = active.mean(dim=1) if active.numel() else model_motion_x0.new_zeros((model_motion_x0.shape[0],))
        coverage_loss = torch.relu(model_motion_x0.new_tensor(_env_float("EDGE_ANTI_FREEZE_MIN_ACTIVE_RATIO", 0.35)) - active_ratio).mean()
        tail_start = max(1, frame_energy.shape[1] // 2)
        tail = frame_energy[:, tail_start:]
        if tail.numel():
            tail_active = torch.sigmoid((tail - active_thr) / max(active_thr * 0.5, 1e-6)).mean(dim=1)
            tail_loss = torch.relu(model_motion_x0.new_tensor(_env_float("EDGE_ANTI_FREEZE_MIN_TAIL_ACTIVE_RATIO", 0.25)) - tail_active).mean()
        else:
            tail_loss = model_motion_x0.new_tensor(0.0)
        total = _env_float("EDGE_ANTI_FREEZE_LOSS_SCALE", 0.5) * (
            mean_loss
            + _env_float("EDGE_ANTI_FREEZE_COVERAGE_WEIGHT", 0.5) * coverage_loss
            + _env_float("EDGE_ANTI_FREEZE_TAIL_WEIGHT", 0.5) * tail_loss
        )
        return torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)

    def _patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        if not _env_bool("EDGE_FREEZE_AWARE_MOTION", False):
            return original_motion_energy_loss(self, model_motion_x0, target_motion_x0)
        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        idx = _feature_indices(model_motion_x0.device, os.environ.get("EDGE_FREEZE_AWARE_FEATURE_MODE", "upper_torso"))
        pred = model_motion_x0[..., idx]
        target = target_motion_x0[..., idx]

        pred_raw = _frame_energy(pred)
        target_raw = _frame_energy(target).detach()
        cap = _target_adaptive_cap(target_raw)
        pred_energy = _saturate_energy(pred_raw, cap)
        target_energy = _saturate_energy(target_raw, cap).detach()

        envelope_loss = F.smooth_l1_loss(pred_energy, target_energy)
        ratio = pred_energy.mean(dim=1) / target_energy.mean(dim=1).clamp_min(1e-8)
        under_loss = torch.relu(model_motion_x0.new_tensor(_env_float("EDGE_MOTION_MIN_TARGET_ENERGY_RATIO", 0.45)) - ratio).pow(2).mean()
        over_loss = torch.relu(ratio - model_motion_x0.new_tensor(_env_float("EDGE_MOTION_MAX_TARGET_ENERGY_RATIO", 1.35))).pow(2).mean()

        target_thr = (target_energy.mean(dim=1, keepdim=True) * _env_float("EDGE_MOTION_ACTIVE_THR_SCALE", 0.20)).clamp_min(1e-5)
        pred_active = torch.sigmoid((pred_energy - target_thr) / target_thr.clamp_min(1e-6)).mean(dim=1)
        tgt_active = torch.sigmoid((target_energy - target_thr) / target_thr.clamp_min(1e-6)).mean(dim=1)
        tgt_active_ratio = (tgt_active * _env_float("EDGE_MOTION_TARGET_ACTIVE_FRACTION", 0.70)).clamp_min(_env_float("EDGE_MOTION_MIN_ACTIVE_RATIO", 0.35))
        coverage_loss = torch.relu(tgt_active_ratio - pred_active).pow(2).mean()

        tail_start = max(1, pred_energy.shape[1] // 2)
        pred_tail = pred_energy[:, tail_start:]
        target_tail = target_energy[:, tail_start:]
        if pred_tail.numel():
            tail_ratio = pred_tail.mean(dim=1) / target_tail.mean(dim=1).clamp_min(1e-8)
            tail_under = torch.relu(model_motion_x0.new_tensor(_env_float("EDGE_MOTION_TAIL_MIN_RATIO", 0.45)) - tail_ratio).pow(2).mean()
            tail_over = torch.relu(tail_ratio - model_motion_x0.new_tensor(_env_float("EDGE_MOTION_TAIL_MAX_RATIO", _env_float("EDGE_MOTION_MAX_TARGET_ENERGY_RATIO", 1.35)))).pow(2).mean()
            pred_tail_active = torch.sigmoid((pred_tail - target_thr) / target_thr.clamp_min(1e-6)).mean(dim=1)
            target_tail_active = torch.sigmoid((target_tail - target_thr) / target_thr.clamp_min(1e-6)).mean(dim=1)
            target_tail_active_ratio = (target_tail_active * _env_float("EDGE_MOTION_TARGET_TAIL_ACTIVE_FRACTION", 0.70)).clamp_min(0.20)
            tail_coverage = torch.relu(target_tail_active_ratio - pred_tail_active).pow(2).mean()
        else:
            tail_under = model_motion_x0.new_tensor(0.0)
            tail_over = model_motion_x0.new_tensor(0.0)
            tail_coverage = model_motion_x0.new_tensor(0.0)

        progress_loss = _temporal_progress_loss(pred, target) if _env_bool("EDGE_TEMPORAL_PROGRESS_SUPERVISION", True) else model_motion_x0.new_tensor(0.0)
        kin_loss = _kinematic_smoothness_loss(pred, target) if _env_bool("EDGE_KINEMATIC_SMOOTHNESS", True) else model_motion_x0.new_tensor(0.0)
        dir_loss = _directional_consistency_loss(pred, target) if _env_bool("EDGE_DIRECTION_CONSISTENCY", True) else model_motion_x0.new_tensor(0.0)
        low_loss = _lowpass_temporal_loss(pred, target) if _env_bool("EDGE_LOWPASS_TEMPORAL_PRIOR", True) else model_motion_x0.new_tensor(0.0)

        total = _env_float("EDGE_MOTION_ENERGY_LOSS_SCALE", 1.0) * (
            envelope_loss
            + _env_float("EDGE_MOTION_ACTIVE_WEIGHT", 1.0) * under_loss
            + _env_float("EDGE_MOTION_OVERACTIVE_WEIGHT", 2.0) * over_loss
            + _env_float("EDGE_MOTION_COVERAGE_WEIGHT", 1.0) * coverage_loss
            + _env_float("EDGE_MOTION_TAIL_WEIGHT", 1.0) * (tail_under + tail_coverage)
            + _env_float("EDGE_MOTION_TAIL_OVERACTIVE_WEIGHT", 2.0) * tail_over
            + _env_float("EDGE_PROGRESS_LOSS_WEIGHT", 3.0) * progress_loss
            + _env_float("EDGE_KINEMATIC_SMOOTHNESS_WEIGHT", 8.0) * kin_loss
            + _env_float("EDGE_DIRECTION_WEIGHT", 2.0) * dir_loss
            + _env_float("EDGE_LOWPASS_WEIGHT", 4.0) * low_loss
        )

        if _env_bool("EDGE_FREEZE_AWARE_DEBUG", False):
            if not hasattr(self, "_edge_v2f_motion_debug_count"):
                self._edge_v2f_motion_debug_count = 0
            self._edge_v2f_motion_debug_count += 1
            if self._edge_v2f_motion_debug_count <= _env_int("EDGE_FREEZE_AWARE_DEBUG_STEPS", 20):
                print(
                    "🧯 V2F burst-safe | "
                    f"ratio={float(ratio.mean().detach().cpu()):.3f} "
                    f"progress={float(progress_loss.detach().cpu()):.4f} "
                    f"kin={float(kin_loss.detach().cpu()):.4f} "
                    f"dir={float(dir_loss.detach().cpu()):.4f} "
                    f"low={float(low_loss.detach().cpu()):.4f} "
                    f"total={float(total.detach().cpu()):.4f}",
                    flush=True,
                )

        return torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)

    GaussianDiffusion._anti_freeze_loss = _patched_anti_freeze_loss
    GaussianDiffusion._motion_energy_loss = _patched_motion_energy_loss
    GaussianDiffusion._edge_v2f_burstsafe_patch_installed = True
    GaussianDiffusion._edge_v2e_temporal_progress_patch_installed = True
    GaussianDiffusion._edge_freeze_aware_motion_patch_installed = True

    if verbose:
        print(
            "✅ Installed V2F burst-safe motion patch: "
            f"motion={_env_bool('EDGE_FREEZE_AWARE_MOTION', False)}, "
            f"saturated={_env_bool('EDGE_BURST_SAFE_PROGRESS', True)}, "
            f"kinematic={_env_bool('EDGE_KINEMATIC_SMOOTHNESS', True)}, "
            f"direction={_env_bool('EDGE_DIRECTION_CONSISTENCY', True)}, "
            f"lowpass={_env_bool('EDGE_LOWPASS_TEMPORAL_PRIOR', True)}, "
            f"feature_mode={os.environ.get('EDGE_FREEZE_AWARE_FEATURE_MODE', 'upper_torso')}"
        )

    return True

def install(verbose: bool = True) -> bool:
    return install_freeze_aware_motion_patch(verbose=verbose)
