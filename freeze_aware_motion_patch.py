from __future__ import annotations

"""
V2E temporal-progress patch for EDGE.

Solves early endpoint arrival / endpoint-collapse by extending the existing
_motion_energy_loss(model_x0, target_x0) with:
- distance-to-end progress curve matching
- cumulative motion progress matching
- front-loaded motion penalty
- top-k jump dominance penalty
- two-sided motion energy ratio to avoid both freeze and jitter

Enable with:
    EDGE_FREEZE_AWARE_MOTION=1
    EDGE_TEMPORAL_PROGRESS_SUPERVISION=1
"""

import os
import torch
import torch.nn.functional as F

_TRUE = {"1", "true", "yes", "y", "on"}

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE

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
        start = 7 + 6 * int(joint)
        out.extend(range(start, start + 6))
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

def _cumulative_progress(energy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if energy.shape[1] == 0:
        return energy
    total = energy.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.cumsum(energy, dim=1) / total

def _topk_share(energy: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    if energy.shape[1] == 0:
        return energy.new_zeros((energy.shape[0],))
    k = max(1, min(int(k), energy.shape[1]))
    values = torch.topk(energy, k=k, dim=1, largest=True).values
    return values.sum(dim=1) / energy.sum(dim=1).clamp_min(eps)

def _temporal_progress_loss(pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """Match GT progress over time instead of allowing early endpoint arrival."""
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

    pred_energy = _frame_energy(pred_feat)
    target_energy = _frame_energy(target_feat).detach()
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
    for k, default_margin in [(1, 0.06), (3, 0.10), (5, 0.12)]:
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
        + _env_float("EDGE_PROGRESS_TOPK_WEIGHT", 0.7) * topk_loss
        + _env_float("EDGE_PROGRESS_EARLY_WEIGHT", 1.0) * early_loss
    )

def install_freeze_aware_motion_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V2E temporal-progress patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v2e_temporal_progress_patch_installed", False):
        if verbose:
            print("✅ V2E temporal-progress patch already installed.")
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
        soft_active = torch.sigmoid((frame_energy - active_thr) / max(active_thr * 0.5, 1e-6))
        active_ratio = soft_active.mean(dim=1) if soft_active.numel() else model_motion_x0.new_zeros((model_motion_x0.shape[0],))
        coverage_loss = torch.relu(model_motion_x0.new_tensor(_env_float("EDGE_ANTI_FREEZE_MIN_ACTIVE_RATIO", 0.35)) - active_ratio).mean()
        tail_start = max(1, frame_energy.shape[1] // 2)
        tail = frame_energy[:, tail_start:]
        if tail.numel():
            tail_active = torch.sigmoid((tail - active_thr) / max(active_thr * 0.5, 1e-6)).mean(dim=1)
            tail_loss = torch.relu(model_motion_x0.new_tensor(_env_float("EDGE_ANTI_FREEZE_MIN_TAIL_ACTIVE_RATIO", 0.25)) - tail_active).mean()
        else:
            tail_loss = model_motion_x0.new_tensor(0.0)
        total = _env_float("EDGE_ANTI_FREEZE_LOSS_SCALE", 1.0) * (
            mean_loss
            + _env_float("EDGE_ANTI_FREEZE_COVERAGE_WEIGHT", 1.0) * coverage_loss
            + _env_float("EDGE_ANTI_FREEZE_TAIL_WEIGHT", 1.0) * tail_loss
        )
        if _env_bool("EDGE_FREEZE_AWARE_DEBUG", False):
            if not hasattr(self, "_edge_v2e_anti_debug_count"):
                self._edge_v2e_anti_debug_count = 0
            self._edge_v2e_anti_debug_count += 1
            if self._edge_v2e_anti_debug_count <= _env_int("EDGE_FREEZE_AWARE_DEBUG_STEPS", 20):
                print(f"🧊 V2E anti_freeze | mean_energy={float(mean_energy.detach().cpu()):.6f} active={float(active_ratio.mean().detach().cpu()):.3f} loss={float(total.detach().cpu()):.6f}", flush=True)
        return torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)

    def _patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        if not _env_bool("EDGE_FREEZE_AWARE_MOTION", False):
            return original_motion_energy_loss(self, model_motion_x0, target_motion_x0)
        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)
        idx = _feature_indices(model_motion_x0.device, os.environ.get("EDGE_FREEZE_AWARE_FEATURE_MODE", "upper_torso"))
        pred = model_motion_x0[..., idx]
        target = target_motion_x0[..., idx]
        pred_energy = _frame_energy(pred)
        target_energy = _frame_energy(target).detach()
        envelope_loss = F.smooth_l1_loss(pred_energy, target_energy)
        pred_mean = pred_energy.mean(dim=1)
        target_mean = target_energy.mean(dim=1).clamp_min(1e-8)
        ratio = pred_mean / target_mean
        min_ratio = _env_float("EDGE_MOTION_MIN_TARGET_ENERGY_RATIO", 0.45)
        max_ratio = _env_float("EDGE_MOTION_MAX_TARGET_ENERGY_RATIO", 2.0)
        under_loss = torch.relu(model_motion_x0.new_tensor(min_ratio) - ratio).pow(2).mean()
        over_loss = torch.relu(ratio - model_motion_x0.new_tensor(max_ratio)).pow(2).mean()
        active_thr_scale = _env_float("EDGE_MOTION_ACTIVE_THR_SCALE", 0.20)
        target_thr = (target_energy.mean(dim=1, keepdim=True) * active_thr_scale).clamp_min(1e-5)
        pred_active_soft = torch.sigmoid((pred_energy - target_thr) / target_thr.clamp_min(1e-6))
        target_active_soft = torch.sigmoid((target_energy - target_thr) / target_thr.clamp_min(1e-6))
        pred_active = pred_active_soft.mean(dim=1)
        target_active = target_active_soft.mean(dim=1)
        target_active_ratio = (target_active * _env_float("EDGE_MOTION_TARGET_ACTIVE_FRACTION", 0.70)).clamp_min(_env_float("EDGE_MOTION_MIN_ACTIVE_RATIO", 0.35))
        coverage_loss = torch.relu(target_active_ratio - pred_active).pow(2).mean()
        tail_start = max(1, pred_energy.shape[1] // 2)
        pred_tail = pred_energy[:, tail_start:]
        target_tail = target_energy[:, tail_start:]
        if pred_tail.numel():
            tail_ratio = pred_tail.mean(dim=1) / target_tail.mean(dim=1).clamp_min(1e-8)
            tail_under_loss = torch.relu(model_motion_x0.new_tensor(_env_float("EDGE_MOTION_TAIL_MIN_RATIO", 0.45)) - tail_ratio).pow(2).mean()
            tail_over_loss = torch.relu(tail_ratio - model_motion_x0.new_tensor(_env_float("EDGE_MOTION_TAIL_MAX_RATIO", max_ratio))).pow(2).mean()
            pred_tail_active = pred_active_soft[:, tail_start:].mean(dim=1)
            target_tail_active = target_active_soft[:, tail_start:].mean(dim=1)
            target_tail_active_ratio = (target_tail_active * _env_float("EDGE_MOTION_TARGET_TAIL_ACTIVE_FRACTION", 0.70)).clamp_min(0.20)
            tail_coverage_loss = torch.relu(target_tail_active_ratio - pred_tail_active).pow(2).mean()
        else:
            tail_under_loss = model_motion_x0.new_tensor(0.0)
            tail_over_loss = model_motion_x0.new_tensor(0.0)
            tail_coverage_loss = model_motion_x0.new_tensor(0.0)
        progress_loss = model_motion_x0.new_tensor(0.0)
        if _env_bool("EDGE_TEMPORAL_PROGRESS_SUPERVISION", True):
            progress_loss = _temporal_progress_loss(pred, target)
        total = _env_float("EDGE_MOTION_ENERGY_LOSS_SCALE", 1.0) * (
            envelope_loss
            + _env_float("EDGE_MOTION_ACTIVE_WEIGHT", 1.5) * under_loss
            + _env_float("EDGE_MOTION_OVERACTIVE_WEIGHT", 1.5) * over_loss
            + _env_float("EDGE_MOTION_COVERAGE_WEIGHT", 2.0) * coverage_loss
            + _env_float("EDGE_MOTION_TAIL_WEIGHT", 2.0) * (tail_under_loss + tail_coverage_loss)
            + _env_float("EDGE_MOTION_TAIL_OVERACTIVE_WEIGHT", 1.0) * tail_over_loss
            + _env_float("EDGE_PROGRESS_LOSS_WEIGHT", 6.0) * progress_loss
        )
        if _env_bool("EDGE_FREEZE_AWARE_DEBUG", False):
            if not hasattr(self, "_edge_v2e_motion_debug_count"):
                self._edge_v2e_motion_debug_count = 0
            self._edge_v2e_motion_debug_count += 1
            if self._edge_v2e_motion_debug_count <= _env_int("EDGE_FREEZE_AWARE_DEBUG_STEPS", 20):
                print(f"🧭 V2E motion/progress | pred/tgt={float(pred_mean.mean().detach().cpu()):.4f}/{float(target_mean.mean().detach().cpu()):.4f} ratio={float(ratio.mean().detach().cpu()):.3f} progress={float(progress_loss.detach().cpu()):.6f} total={float(total.detach().cpu()):.6f}", flush=True)
        return torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)

    GaussianDiffusion._anti_freeze_loss = _patched_anti_freeze_loss
    GaussianDiffusion._motion_energy_loss = _patched_motion_energy_loss
    GaussianDiffusion._edge_v2e_temporal_progress_patch_installed = True
    GaussianDiffusion._edge_freeze_aware_motion_patch_installed = True
    if verbose:
        print(
            "✅ Installed V2E temporal-progress motion patch: "
            f"enabled={_env_bool('EDGE_FREEZE_AWARE_MOTION', False)}, "
            f"progress={_env_bool('EDGE_TEMPORAL_PROGRESS_SUPERVISION', True)}, "
            f"feature_mode={os.environ.get('EDGE_FREEZE_AWARE_FEATURE_MODE', 'upper_torso')}, "
            f"curve={os.environ.get('EDGE_PROGRESS_CURVE', 'gt')}"
        )
    return True

def install(verbose: bool = True) -> bool:
    return install_freeze_aware_motion_patch(verbose=verbose)
