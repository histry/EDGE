from __future__ import annotations

import os
from functools import wraps
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


TRUE = {"1", "true", "yes", "y", "on"}

CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT_START = 7
ROT_END = 151

LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in TRUE


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


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip()


def _rot_indices(joints):
    out = []
    for j in joints:
        s = ROT_START + 6 * int(j)
        out.extend(range(s, s + 6))
    return out


def _feature_indices(nfeats: int, spec: str, device) -> torch.Tensor:
    spec = (spec or "rot+root_y").lower().replace(",", "+")
    parts = {p.strip() for p in spec.split("+") if p.strip()}

    idx = []

    if "all" in parts:
        idx.extend(range(nfeats))

    if "contacts" in parts or "contact" in parts:
        idx.extend(range(0, min(4, nfeats)))

    if "root" in parts:
        idx.extend(i for i in [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX] if i < nfeats)

    if "root_xz" in parts or "rootxz" in parts:
        idx.extend(i for i in [ROOT_X_IDX, ROOT_Z_IDX] if i < nfeats)

    if "root_y" in parts or "rooty" in parts:
        if ROOT_Y_IDX < nfeats:
            idx.append(ROOT_Y_IDX)

    if "rot" in parts or "rotation" in parts:
        idx.extend(range(ROT_START, min(ROT_END, nfeats)))

    if "pelvis" in parts:
        idx.extend(i for i in _rot_indices([0]) if i < nfeats)

    if "lower" in parts:
        idx.extend(i for i in _rot_indices(LOWER_JOINTS) if i < nfeats)

    if "torso" in parts:
        idx.extend(i for i in _rot_indices(TORSO_JOINTS) if i < nfeats)

    if "upper" in parts:
        idx.extend(i for i in _rot_indices(UPPER_JOINTS) if i < nfeats)

    idx = sorted(set(int(i) for i in idx if 0 <= int(i) < nfeats))
    if not idx:
        idx = list(range(ROT_START, min(ROT_END, nfeats)))

    return torch.as_tensor(idx, device=device, dtype=torch.long)


def _expand_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if mask.shape[-1] == 1:
        return mask.expand_as(value)
    if mask.shape[-1] == value.shape[-1]:
        return mask
    raise ValueError(
        f"constraint mask last dim must be 1 or {value.shape[-1]}, got {mask.shape[-1]}"
    )


def _move_condition_to_device(cond, device):
    if isinstance(cond, dict):
        return {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in cond.items()
        }
    if torch.is_tensor(cond):
        return cond.to(device)
    return cond


def _maybe_unnorm(self, x: torch.Tensor) -> torch.Tensor:
    normalizer = getattr(self, "normalizer", None)
    if normalizer is None:
        return x
    try:
        out = normalizer.unnormalize(x)
        if not torch.is_tensor(out):
            out = torch.as_tensor(out, device=x.device, dtype=x.dtype)
        return out.to(device=x.device, dtype=x.dtype)
    except Exception:
        return x


def _hard_project_xstart(self, x_start: torch.Tensor, constraint: Optional[Dict]) -> torch.Tensor:
    if constraint is None:
        return x_start
    mask = constraint.get("mask", None)
    value = constraint.get("value", None)
    if mask is None or value is None:
        return x_start

    mask = mask.to(device=x_start.device, dtype=x_start.dtype)
    value = value.to(device=x_start.device, dtype=x_start.dtype)
    mask = _expand_mask(mask, x_start)
    return x_start * (1.0 - mask) + value * mask


def _should_project_xstart(self) -> bool:
    return (
        bool(getattr(self, "hard_keyframe_project", False))
        or _env_bool("EDGE_HARD_KEYFRAME_PROJECT", False)
        or _env_bool("EDGE_INFER_PROJECT_XSTART", False)
    )


def _anchor_frames_from_hard_mask(mask_full: torch.Tensor) -> torch.Tensor:
    # Use rotation-heavy mask to identify real pose anchors.
    # This avoids treating all-frame root X/Z constraints as pose anchors.
    if mask_full.shape[-1] >= ROT_END:
        score = mask_full[..., ROT_START:ROT_END].mean(dim=-1)
    else:
        score = mask_full.mean(dim=-1)
    return score > 0.5


def _build_linear_bridge_from_constraint(
    value: torch.Tensor,
    mask_full: torch.Tensor,
    feature_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a dense temporal bridge from sparse hard pose anchors.

    The bridge is NOT a hard projection target. It is only passed to the model as
    force_x_clean / force_indicator so the denoiser sees a low-frequency pose path.
    """
    bsz, seq_len, nfeats = value.shape
    device = value.device
    dtype = value.dtype

    bridge_value = torch.zeros_like(value)
    bridge_mask = torch.zeros_like(value)

    anchor_bool = _anchor_frames_from_hard_mask(mask_full)

    for b in range(bsz):
        anchors = torch.where(anchor_bool[b])[0]
        if anchors.numel() == 0:
            continue

        # One anchor: hold that pose as a weak bridge.
        if anchors.numel() == 1:
            a = int(anchors[0].item())
            bridge_value[b, :, feature_idx] = value[b, a:a + 1, feature_idx].expand(seq_len, -1)
            bridge_mask[b, :, feature_idx] = 1.0
            continue

        # Fill before first anchor.
        first = int(anchors[0].item())
        if first > 0:
            bridge_value[b, :first + 1, feature_idx] = value[b, first:first + 1, feature_idx].expand(first + 1, -1)
            bridge_mask[b, :first + 1, feature_idx] = 1.0

        # Linear interpolation between anchors.
        for k in range(anchors.numel() - 1):
            s = int(anchors[k].item())
            e = int(anchors[k + 1].item())
            if e <= s:
                continue
            steps = e - s + 1
            alpha = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype).view(steps, 1)
            left = value[b, s, feature_idx].view(1, -1)
            right = value[b, e, feature_idx].view(1, -1)
            bridge_value[b, s:e + 1, feature_idx] = (1.0 - alpha) * left + alpha * right
            bridge_mask[b, s:e + 1, feature_idx] = 1.0

        # Fill after last anchor.
        last = int(anchors[-1].item())
        if last < seq_len - 1:
            bridge_value[b, last:, feature_idx] = value[b, last:last + 1, feature_idx].expand(seq_len - last, -1)
            bridge_mask[b, last:, feature_idx] = 1.0

    return bridge_value, bridge_mask


def _augment_constraint_with_bridge(self, constraint: Optional[Dict]) -> Optional[Dict]:
    if constraint is None or not _env_bool("EDGE_RECON_BRIDGE_COND", False):
        return constraint

    mask = constraint.get("mask", None)
    value = constraint.get("value", None)
    if mask is None or value is None:
        return constraint

    value = value.to(dtype=torch.float32)
    mask = mask.to(device=value.device, dtype=value.dtype)
    mask_full = _expand_mask(mask, value)

    feature_spec = _env_str("EDGE_RECON_BRIDGE_FEATURES", "rot+root_y")
    feature_idx = _feature_indices(value.shape[-1], feature_spec, value.device)

    bridge_value, bridge_mask = _build_linear_bridge_from_constraint(
        value=value,
        mask_full=mask_full,
        feature_idx=feature_idx,
    )

    strength = max(0.0, min(1.0, _env_float("EDGE_RECON_BRIDGE_STRENGTH", 0.35)))
    bridge_mask = bridge_mask * strength

    # Hard constraints remain hard. Bridge is only model conditioning.
    force_mask = torch.maximum(mask_full, bridge_mask)
    force_value = torch.where(mask_full > 0.5, value, bridge_value)

    out = dict(constraint)
    out["force_mask"] = force_mask.to(device=value.device, dtype=value.dtype)
    out["force_value"] = force_value.to(device=value.device, dtype=value.dtype)
    out["_edge_bridge_applied"] = True
    return out


def _make_dense_training_constraint(self, x_start: torch.Tensor) -> Dict:
    """
    Dense keyframe training for strict single-unit reconstruction.

    Hard projection:
      - every N frames for selected pose dimensions;
      - optional all-frame root X/Z.

    Bridge conditioning:
      - all frames receive an interpolated low-frequency pose bridge,
        but only through force_mask/force_value, not as hard projection.
    """
    bsz, seq_len, nfeats = x_start.shape
    device = x_start.device
    dtype = x_start.dtype

    stride = max(1, _env_int("EDGE_RECON_TRAIN_DENSE_STRIDE", 4))
    frames = list(range(0, seq_len, stride))
    if (seq_len - 1) not in frames:
        frames.append(seq_len - 1)
    frames = sorted(set(frames))

    hard_features = _feature_indices(
        nfeats,
        _env_str("EDGE_RECON_TRAIN_HARD_FEATURES", "rot+root_y+contacts"),
        device,
    )

    mask = torch.zeros((bsz, seq_len, nfeats), device=device, dtype=dtype)
    value = torch.zeros_like(x_start)

    frame_idx = torch.as_tensor(frames, device=device, dtype=torch.long)
    mask[:, frame_idx[:, None], hard_features[None, :]] = 1.0

    if nfeats >= 151 and _env_bool("EDGE_RECON_TRAIN_ROOT_XZ_ALL", True):
        mask[:, :, ROOT_X_IDX] = 1.0
        mask[:, :, ROOT_Z_IDX] = 1.0

    value = x_start * mask
    constraint = {"mask": mask, "value": value}
    return _augment_constraint_with_bridge(self, constraint)


def _feature_weight_vector(nfeats: int, device, dtype) -> torch.Tensor:
    w = torch.zeros((nfeats,), device=device, dtype=dtype)

    if nfeats >= 4:
        w[CONTACT_SLICE] = _env_float("EDGE_RECON_LOSS_CONTACT_W", 0.25)

    if nfeats > ROOT_X_IDX:
        w[ROOT_X_IDX] = _env_float("EDGE_RECON_LOSS_ROOT_XZ_W", 0.0)
    if nfeats > ROOT_Y_IDX:
        w[ROOT_Y_IDX] = _env_float("EDGE_RECON_LOSS_ROOT_Y_W", 1.0)
    if nfeats > ROOT_Z_IDX:
        w[ROOT_Z_IDX] = _env_float("EDGE_RECON_LOSS_ROOT_XZ_W", 0.0)

    for i in _rot_indices([0]):
        if i < nfeats:
            w[i] = _env_float("EDGE_RECON_LOSS_PELVIS_W", 4.0)

    for i in _rot_indices(LOWER_JOINTS):
        if i < nfeats:
            w[i] = _env_float("EDGE_RECON_LOSS_LOWER_W", 2.0)

    for i in _rot_indices(TORSO_JOINTS):
        if i < nfeats:
            w[i] = _env_float("EDGE_RECON_LOSS_TORSO_W", 4.0)

    for i in _rot_indices(UPPER_JOINTS):
        if i < nfeats:
            w[i] = _env_float("EDGE_RECON_LOSS_UPPER_W", 8.0)

    # fallback for non-151D or missing weights
    if float(w.sum().detach().cpu()) <= 1e-8:
        w[:] = 1.0

    return w


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wv = w.view(1, 1, -1).to(device=pred.device, dtype=pred.dtype)
    denom = wv.sum().clamp_min(1e-6) * pred.shape[0] * pred.shape[1]
    return ((pred - target).pow(2) * wv).sum() / denom


def _key_neighbor_loss(pred: torch.Tensor, target: torch.Tensor, constraint: Optional[Dict], w: torch.Tensor) -> torch.Tensor:
    if constraint is None or constraint.get("mask", None) is None:
        return pred.new_tensor(0.0)

    radius = max(0, _env_int("EDGE_RECON_KEY_NEIGHBOR_RADIUS", 1))
    if radius <= 0:
        return pred.new_tensor(0.0)

    mask = constraint["mask"].to(device=pred.device, dtype=pred.dtype)
    mask = _expand_mask(mask, pred)
    anchor_bool = _anchor_frames_from_hard_mask(mask)

    bsz, seq_len, _ = pred.shape
    frame_w = torch.zeros((bsz, seq_len, 1), device=pred.device, dtype=pred.dtype)

    for b in range(bsz):
        anchors = torch.where(anchor_bool[b])[0]
        for a in anchors:
            f = int(a.item())
            s = max(0, f - radius)
            e = min(seq_len, f + radius + 1)
            frame_w[b, s:e, 0] = 1.0

    if float(frame_w.sum().detach().cpu()) <= 1e-8:
        return pred.new_tensor(0.0)

    wv = w.view(1, 1, -1).to(device=pred.device, dtype=pred.dtype)
    denom = (frame_w * wv).sum().clamp_min(1e-6)
    return ((pred - target).pow(2) * frame_w * wv).sum() / denom


def install_single_unit_recon_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ single-unit recon patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_single_unit_recon_patch_installed", False):
        if verbose:
            print("✅ single-unit recon patch already installed.")
        return True

    original_model_predictions = GaussianDiffusion.model_predictions
    original_build_keyframe_condition = GaussianDiffusion._build_keyframe_condition
    original_p_losses = GaussianDiffusion.p_losses

    @wraps(original_build_keyframe_condition)
    def patched_build_keyframe_condition(self, x_start, cond=None):
        if _env_bool("EDGE_RECON_TRAIN_DENSE_KEYFRAMES", False):
            return _make_dense_training_constraint(self, x_start)

        constraint = original_build_keyframe_condition(self, x_start, cond)
        return _augment_constraint_with_bridge(self, constraint)

    def patched_model_predictions(
        self,
        x,
        cond,
        t,
        weight=None,
        clip_x_start=False,
        constraint=None,
    ):
        weight = self.guidance_weight if weight is None else weight
        model_constraint = _augment_constraint_with_bridge(self, constraint)

        force_mask = None
        force_x_clean = None
        if model_constraint is not None:
            force_mask = model_constraint.get("force_mask", model_constraint.get("mask", None))
            force_x_clean = model_constraint.get("force_value", model_constraint.get("value", None))

        if hasattr(self.model, "guided_forward"):
            model_output = self.model.guided_forward(
                x,
                cond,
                t,
                weight,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
            )
        else:
            model_output = self.model(
                x,
                cond,
                t,
                cond_drop_prob=0.0,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
            )

        if self.predict_epsilon:
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        else:
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        if clip_x_start:
            if x_start.shape[-1] > 7:
                x_start = torch.cat(
                    [x_start[..., :7], x_start[..., 7:].clamp(-1.0, 1.0)],
                    dim=-1,
                )
            else:
                x_start = x_start.clamp(-1.0, 1.0)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        # Important: project only original hard mask/value, not bridge force.
        if constraint is not None and _should_project_xstart(self):
            x_start = _hard_project_xstart(self, x_start, constraint)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    @wraps(original_p_losses)
    def patched_p_losses(self, x_start, cond, t, noise=None, current_epoch=None, constraint=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # Original objective, but it now uses patched dense/bridge keyframe condition.
        total_loss, losses = original_p_losses(
            self,
            x_start,
            cond,
            t,
            noise=noise,
            current_epoch=current_epoch,
            constraint=constraint,
        )

        if not _env_bool("EDGE_RECON_EXTRA_LOSS", False):
            return total_loss, losses

        cond = _move_condition_to_device(cond, x_start.device)

        train_constraint = self._build_keyframe_condition(x_start, cond)
        train_constraint = self._merge_constraints(train_constraint, constraint)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self._project_known_keyframes(x_noisy, train_constraint, t)

        force_mask = None
        force_value = None
        if train_constraint is not None:
            force_mask = train_constraint.get("force_mask", train_constraint.get("mask", None))
            force_value = train_constraint.get("force_value", train_constraint.get("value", None))

        model_out = self.model(
            x_noisy,
            cond,
            t,
            cond_drop_prob=0.0,
            force_mask=force_mask,
            force_x_clean=force_value,
        )

        if self.predict_epsilon:
            model_x0 = self.predict_start_from_noise(x_noisy, t, model_out)
        else:
            model_x0 = model_out

        pred_phys = _maybe_unnorm(self, model_x0)
        target_phys = _maybe_unnorm(self, x_start)

        w = _feature_weight_vector(pred_phys.shape[-1], pred_phys.device, pred_phys.dtype)

        x0_loss = _weighted_mse(pred_phys, target_phys, w)

        if pred_phys.shape[1] > 1:
            vel_loss = _weighted_mse(
                pred_phys[:, 1:] - pred_phys[:, :-1],
                target_phys[:, 1:] - target_phys[:, :-1],
                w,
            )
        else:
            vel_loss = pred_phys.new_tensor(0.0)

        if pred_phys.shape[1] > 2:
            acc_loss = _weighted_mse(
                pred_phys[:, 2:] - 2.0 * pred_phys[:, 1:-1] + pred_phys[:, :-2],
                target_phys[:, 2:] - 2.0 * target_phys[:, 1:-1] + target_phys[:, :-2],
                w,
            )
        else:
            acc_loss = pred_phys.new_tensor(0.0)

        neigh_loss = _key_neighbor_loss(pred_phys, target_phys, train_constraint, w)

        extra = (
            _env_float("EDGE_RECON_EXTRA_X0_W", 50.0) * x0_loss
            + _env_float("EDGE_RECON_EXTRA_VEL_W", 25.0) * vel_loss
            + _env_float("EDGE_RECON_EXTRA_ACC_W", 5.0) * acc_loss
            + _env_float("EDGE_RECON_EXTRA_KEY_NEIGHBOR_W", 10.0) * neigh_loss
        )

        if _env_bool("EDGE_RECON_EXTRA_DEBUG", False):
            if not hasattr(self, "_edge_recon_extra_debug_count"):
                self._edge_recon_extra_debug_count = 0
            self._edge_recon_extra_debug_count += 1
            if self._edge_recon_extra_debug_count <= 20 or self._edge_recon_extra_debug_count % 100 == 0:
                print(
                    "🧪 EDGE_RECON_EXTRA "
                    f"step={self._edge_recon_extra_debug_count} "
                    f"x0={float(x0_loss.detach().cpu()):.6f} "
                    f"vel={float(vel_loss.detach().cpu()):.6f} "
                    f"acc={float(acc_loss.detach().cpu()):.6f} "
                    f"neighbor={float(neigh_loss.detach().cpu()):.6f} "
                    f"extra={float(extra.detach().cpu()):.6f} "
                    f"base={float(total_loss.detach().cpu()):.6f}",
                    flush=True,
                )

        return total_loss + extra, losses

    GaussianDiffusion._build_keyframe_condition = patched_build_keyframe_condition
    GaussianDiffusion.model_predictions = patched_model_predictions
    GaussianDiffusion.p_losses = patched_p_losses
    GaussianDiffusion._edge_single_unit_recon_patch_installed = True

    if verbose:
        print(
            "✅ Installed single-unit reconstruction patch: "
            f"bridge={_env_bool('EDGE_RECON_BRIDGE_COND', False)}, "
            f"dense_train={_env_bool('EDGE_RECON_TRAIN_DENSE_KEYFRAMES', False)}, "
            f"extra_loss={_env_bool('EDGE_RECON_EXTRA_LOSS', False)}"
        )
    return True


def install(verbose: bool = True) -> bool:
    return install_single_unit_recon_patch(verbose=verbose)
