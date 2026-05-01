"""Training losses for retrieved segment prior fine-tuning."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _expand_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    if mask.shape[-1] == 1:
        return mask.expand_as(value)
    if mask.shape[-1] == value.shape[-1]:
        return mask
    raise ValueError(f"mask last dim must be 1 or {value.shape[-1]}, got {mask.shape[-1]}")


def build_rag_constraint(x_clean: torch.Tensor, cond: Dict[str, torch.Tensor], keyframe_width: int = 2) -> Dict[str, torch.Tensor]:
    """Build force_mask/value for the decoder input.

    The retrieved prior is exposed as masked clean conditioning in the middle
    segment.  Start/end target frames are also exposed, matching the controlled
    generation setup.
    """
    b, t, c = x_clean.shape
    device, dtype = x_clean.device, x_clean.dtype
    mask = torch.zeros((b, t, c), device=device, dtype=dtype)
    value = torch.zeros_like(x_clean)

    if keyframe_width > 0:
        w = min(int(keyframe_width), t)
        mask[:, :w, :] = 1.0
        mask[:, t - w :, :] = 1.0
        # Avoid fighting trajectory branch on root X/Z.
        mask[..., 4] = 0.0
        mask[..., 6] = 0.0
        value[:, :w, :] = x_clean[:, :w, :]
        value[:, t - w :, :] = x_clean[:, t - w :, :]

    prior_mask = cond.get("retrieved_prior_mask")
    prior_value = cond.get("retrieved_prior_value")
    if prior_mask is not None and prior_value is not None:
        prior_value = prior_value.to(device=device, dtype=dtype)
        prior_mask = _expand_mask(prior_mask, x_clean)
        # Retrieved prior is lower priority than hard start/end keyframes.
        take_prior = (prior_mask > 0.5) & (mask <= 0.5)
        value = torch.where(take_prior, prior_value, value)
        mask = torch.maximum(mask, prior_mask)

    return {"mask": mask, "value": value}


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _expand_mask(mask, pred)
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_velocity_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _expand_mask(mask, pred)
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_v = pred[:, 1:] - pred[:, :-1]
    target_v = target[:, 1:] - target[:, :-1]
    pair_mask = torch.minimum(mask[:, 1:], mask[:, :-1])
    denom = pair_mask.sum().clamp_min(1.0)
    return (torch.abs(pred_v - target_v) * pair_mask).sum() / denom


def transition_smoothness_l1(pred: torch.Tensor, target: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
    """Match target velocities at boundaries of retrieved segment.

    This discourages sudden snapping when entering/leaving the retrieved-prior
    region, which was the main visual failure mode in inference-only RAG.
    """
    segment_mask = segment_mask.to(device=pred.device, dtype=pred.dtype)
    if segment_mask.shape[-1] != 1:
        segment_mask = segment_mask.amax(dim=-1, keepdim=True)
    b, t, c = pred.shape
    losses = []
    active = segment_mask[..., 0] > 0.5
    for bi in range(b):
        idx = torch.nonzero(active[bi], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        s = int(idx.min().item())
        e = int(idx.max().item()) + 1
        if s > 0:
            losses.append(F.l1_loss(pred[bi, s] - pred[bi, s - 1], target[bi, s] - target[bi, s - 1]))
        if e < t:
            losses.append(F.l1_loss(pred[bi, e] - pred[bi, e - 1], target[bi, e] - target[bi, e - 1]))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def rag_segment_training_loss(
    diffusion,
    x_clean: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    keyframe_width: int = 2,
    imitation_weight: float = 1.0,
    velocity_weight: float = 0.5,
    transition_weight: float = 0.2,
    t_min: int = 0,
    t_max: int = -1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute one denoising step with retrieved segment conditioning."""
    b = x_clean.shape[0]
    device = x_clean.device
    n_timestep = int(getattr(diffusion, "n_timestep", 1000))
    if t_max is None or int(t_max) <= 0:
        t_max = n_timestep - 1
    t_min = max(0, int(t_min))
    t_max = min(n_timestep - 1, int(t_max))
    if t_max <= t_min:
        t_max = n_timestep - 1
    t = torch.randint(t_min, t_max + 1, (b,), device=device).long()
    noise = torch.randn_like(x_clean)
    x_noisy = diffusion.q_sample(x_clean, t, noise=noise)
    constraint = build_rag_constraint(x_clean, cond, keyframe_width=keyframe_width)

    _, x_start = diffusion.model_predictions(
        x_noisy,
        cond,
        t,
        clip_x_start=getattr(diffusion, "clip_denoised", False),
        constraint=constraint,
    )

    prior_mask = cond.get("retrieved_prior_mask")
    if prior_mask is None:
        return x_start.new_tensor(0.0), {
            "rag_segment_imitation": x_start.new_tensor(0.0),
            "rag_segment_velocity": x_start.new_tensor(0.0),
            "rag_transition_smooth": x_start.new_tensor(0.0),
        }

    segment_mask = cond.get("retrieved_segment_mask", prior_mask[..., :1])
    imitation = masked_l1(x_start, x_clean, prior_mask)
    velocity = masked_velocity_l1(x_start, x_clean, prior_mask)
    transition = transition_smoothness_l1(x_start, x_clean, segment_mask)
    total = float(imitation_weight) * imitation + float(velocity_weight) * velocity + float(transition_weight) * transition
    return total, {
        "rag_segment_imitation": imitation.detach(),
        "rag_segment_velocity": velocity.detach(),
        "rag_transition_smooth": transition.detach(),
    }
