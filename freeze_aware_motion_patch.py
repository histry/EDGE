from __future__ import annotations

import os
import torch
import torch.nn.functional as F


_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in _TRUE


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _safe_norm(x, dim=-1, eps=1e-8):
    return torch.sqrt(torch.sum(x * x, dim=dim) + eps)


def _rot_indices(joints):
    out = []
    for j in joints:
        s = 7 + 6 * int(j)
        out.extend(range(s, s + 6))
    return out


UPPER_JOINTS = list(range(12, 24))
TORSO_JOINTS = [3, 6, 9, 12, 13, 14, 15]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]

UPPER_IDX = _rot_indices(UPPER_JOINTS)
TORSO_IDX = _rot_indices(TORSO_JOINTS)
LOWER_IDX = _rot_indices(LOWER_JOINTS)


def _feature_indices(device, mode: str):
    mode = str(mode or "upper_torso").lower()
    if mode in {"all", "full", "body"}:
        idx = list(range(7, 151))
    elif mode in {"upper", "arms"}:
        idx = UPPER_IDX
    elif mode in {"torso"}:
        idx = TORSO_IDX
    elif mode in {"upper_torso", "torso_upper"}:
        idx = sorted(set(UPPER_IDX + TORSO_IDX))
    elif mode in {"lower", "legs"}:
        idx = LOWER_IDX
    else:
        idx = sorted(set(UPPER_IDX + TORSO_IDX))
    return torch.as_tensor(idx, device=device, dtype=torch.long)


def install_freeze_aware_motion_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ freeze-aware motion patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_freeze_aware_motion_patch_installed", False):
        if verbose:
            print("✅ freeze-aware motion patch already installed.")
        return True

    original_anti_freeze_loss = GaussianDiffusion._anti_freeze_loss
    original_motion_energy_loss = GaussianDiffusion._motion_energy_loss

    def _patched_anti_freeze_loss(self, model_motion_x0):
        if not _env_bool("EDGE_FREEZE_AWARE_MOTION", False):
            return original_anti_freeze_loss(self, model_motion_x0)

        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        mode = os.environ.get("EDGE_FREEZE_AWARE_FEATURE_MODE", "upper_torso")
        idx = _feature_indices(model_motion_x0.device, mode)
        x = model_motion_x0[..., idx]

        delta = x[:, 1:] - x[:, :-1]
        frame_energy = _safe_norm(delta, dim=-1)

        # Basic energy floor.
        min_energy = _env_float("EDGE_ANTI_FREEZE_MIN_ENERGY", 0.015)
        mean_energy = frame_energy.mean()
        mean_loss = F.relu(model_motion_x0.new_tensor(min_energy) - mean_energy)

        # Active-frame coverage: prevent one jump + long hold.
        active_thr = _env_float("EDGE_ANTI_FREEZE_ACTIVE_THR", 0.010)
        min_active = _env_float("EDGE_ANTI_FREEZE_MIN_ACTIVE_RATIO", 0.35)
        soft_active = torch.sigmoid((frame_energy - active_thr) / max(active_thr * 0.5, 1e-6))
        active_ratio = soft_active.mean(dim=1)
        coverage_loss = F.relu(model_motion_x0.new_tensor(min_active) - active_ratio).mean()

        # Tail coverage: prevent front-loaded motion then freeze.
        tail_start = max(1, frame_energy.shape[1] // 2)
        tail_energy = frame_energy[:, tail_start:]
        if tail_energy.numel() > 0:
            tail_active = torch.sigmoid((tail_energy - active_thr) / max(active_thr * 0.5, 1e-6)).mean(dim=1)
            min_tail = _env_float("EDGE_ANTI_FREEZE_MIN_TAIL_ACTIVE_RATIO", 0.25)
            tail_loss = F.relu(model_motion_x0.new_tensor(min_tail) - tail_active).mean()
        else:
            tail_loss = model_motion_x0.new_tensor(0.0)

        total = (
            mean_loss
            + _env_float("EDGE_ANTI_FREEZE_COVERAGE_WEIGHT", 2.0) * coverage_loss
            + _env_float("EDGE_ANTI_FREEZE_TAIL_WEIGHT", 2.0) * tail_loss
        )
        total = _env_float("EDGE_ANTI_FREEZE_LOSS_SCALE", 1.0) * total

        if _env_bool("EDGE_FREEZE_AWARE_DEBUG", False):
            if not hasattr(self, "_edge_freeze_debug_counter"):
                self._edge_freeze_debug_counter = 0
            self._edge_freeze_debug_counter += 1
            if self._edge_freeze_debug_counter <= 20:
                print(
                    "🧊 freeze-aware anti_freeze | "
                    f"mean_energy={float(mean_energy.detach().cpu()):.6f} "
                    f"active={float(active_ratio.mean().detach().cpu()):.3f} "
                    f"loss={float(total.detach().cpu()):.6f}",
                    flush=True,
                )

        return torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)

    def _patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        if not _env_bool("EDGE_FREEZE_AWARE_MOTION", False):
            return original_motion_energy_loss(self, model_motion_x0, target_motion_x0)

        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        mode = os.environ.get("EDGE_FREEZE_AWARE_FEATURE_MODE", "upper_torso")
        idx = _feature_indices(model_motion_x0.device, mode)

        pred = model_motion_x0[..., idx]
        tgt = target_motion_x0[..., idx]

        pred_delta = pred[:, 1:] - pred[:, :-1]
        tgt_delta = tgt[:, 1:] - tgt[:, :-1]

        pred_energy = _safe_norm(pred_delta, dim=-1)
        tgt_energy = _safe_norm(tgt_delta, dim=-1)

        # 1) Original energy envelope matching.
        envelope_loss = F.mse_loss(pred_energy, tgt_energy)

        # 2) Target-aware energy ratio: generated motion should not be much less active than GT.
        pred_mean = pred_energy.mean(dim=1)
        tgt_mean = tgt_energy.mean(dim=1).clamp_min(1e-8)
        ratio = pred_mean / tgt_mean
        min_ratio = _env_float("EDGE_MOTION_MIN_TARGET_ENERGY_RATIO", 0.45)
        ratio_loss = F.relu(model_motion_x0.new_tensor(min_ratio) - ratio).mean()

        # 3) Active-frame coverage: avoid one transition then static hold.
        active_thr_scale = _env_float("EDGE_MOTION_ACTIVE_THR_SCALE", 0.20)
        tgt_thr = (tgt_energy.detach().mean(dim=1, keepdim=True) * active_thr_scale).clamp_min(1e-5)
        pred_active_soft = torch.sigmoid((pred_energy - tgt_thr) / tgt_thr.clamp_min(1e-6))
        tgt_active_soft = torch.sigmoid((tgt_energy.detach() - tgt_thr) / tgt_thr.clamp_min(1e-6))

        pred_active = pred_active_soft.mean(dim=1)
        tgt_active = tgt_active_soft.mean(dim=1)

        min_active_ratio = _env_float("EDGE_MOTION_MIN_ACTIVE_RATIO", 0.35)
        target_active_ratio = (tgt_active * _env_float("EDGE_MOTION_TARGET_ACTIVE_FRACTION", 0.70)).clamp_min(min_active_ratio)
        coverage_loss = F.relu(target_active_ratio - pred_active).mean()

        # 4) Tail coverage: generated motion must continue in the latter half.
        tail_start = max(1, pred_energy.shape[1] // 2)
        pred_tail = pred_energy[:, tail_start:]
        tgt_tail = tgt_energy[:, tail_start:]

        if pred_tail.numel() > 0:
            pred_tail_mean = pred_tail.mean(dim=1)
            tgt_tail_mean = tgt_tail.mean(dim=1).clamp_min(1e-8)
            tail_ratio = pred_tail_mean / tgt_tail_mean
            min_tail_ratio = _env_float("EDGE_MOTION_TAIL_MIN_RATIO", 0.45)
            tail_ratio_loss = F.relu(model_motion_x0.new_tensor(min_tail_ratio) - tail_ratio).mean()

            pred_tail_active = pred_active_soft[:, tail_start:].mean(dim=1)
            tgt_tail_active = tgt_active_soft[:, tail_start:].mean(dim=1)
            target_tail_active = (tgt_tail_active * _env_float("EDGE_MOTION_TARGET_TAIL_ACTIVE_FRACTION", 0.70)).clamp_min(0.20)
            tail_coverage_loss = F.relu(target_tail_active - pred_tail_active).mean()
        else:
            tail_ratio_loss = model_motion_x0.new_tensor(0.0)
            tail_coverage_loss = model_motion_x0.new_tensor(0.0)

        total = (
            envelope_loss
            + _env_float("EDGE_MOTION_ACTIVE_WEIGHT", 3.0) * ratio_loss
            + _env_float("EDGE_MOTION_COVERAGE_WEIGHT", 6.0) * coverage_loss
            + _env_float("EDGE_MOTION_TAIL_WEIGHT", 6.0) * (tail_ratio_loss + tail_coverage_loss)
        )
        total = _env_float("EDGE_MOTION_ENERGY_LOSS_SCALE", 1.0) * total

        if _env_bool("EDGE_FREEZE_AWARE_DEBUG", False):
            if not hasattr(self, "_edge_motion_energy_debug_counter"):
                self._edge_motion_energy_debug_counter = 0
            self._edge_motion_energy_debug_counter += 1
            if self._edge_motion_energy_debug_counter <= 20:
                print(
                    "🧊 freeze-aware motion_energy | "
                    f"pred/tgt={float(pred_mean.mean().detach().cpu()):.6f}/"
                    f"{float(tgt_mean.mean().detach().cpu()):.6f} "
                    f"ratio={float(ratio.mean().detach().cpu()):.3f} "
                    f"active={float(pred_active.mean().detach().cpu()):.3f} "
                    f"loss={float(total.detach().cpu()):.6f}",
                    flush=True,
                )

        return torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0)

    GaussianDiffusion._anti_freeze_loss = _patched_anti_freeze_loss
    GaussianDiffusion._motion_energy_loss = _patched_motion_energy_loss
    GaussianDiffusion._edge_freeze_aware_motion_patch_installed = True

    if verbose:
        print(
            "✅ Installed freeze-aware motion coverage patch: "
            f"enabled={_env_bool('EDGE_FREEZE_AWARE_MOTION', False)}, "
            f"feature_mode={os.environ.get('EDGE_FREEZE_AWARE_FEATURE_MODE', 'upper_torso')}"
        )
    return True


def install(verbose: bool = True) -> bool:
    return install_freeze_aware_motion_patch(verbose=verbose)
