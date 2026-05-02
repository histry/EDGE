from __future__ import annotations

import math
import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def resolve_logit_scale(
    temperature: Optional[Union[float, torch.Tensor]] = None,
    logit_scale: Optional[Union[float, torch.Tensor]] = None,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Resolve CLIP-style logit scale.

    Exactly one of `temperature` or `logit_scale` may be provided.

    - temperature τ means logits are scaled as logits / τ.
    - logit_scale means logits are scaled as logits * logit_scale.
    """
    if temperature is not None and logit_scale is not None:
        raise ValueError("Pass either temperature or logit_scale, not both.")

    if logit_scale is not None:
        if torch.is_tensor(logit_scale):
            return logit_scale.to(device=device, dtype=dtype)
        return torch.tensor(float(logit_scale), device=device, dtype=dtype)

    if temperature is None:
        # CLIP commonly initializes logit_scale ~= 1 / 0.07.
        # Override with EDGE_MMR_TEMPERATURE if needed.
        temperature = _env_float("EDGE_MMR_TEMPERATURE", 0.07)

    if torch.is_tensor(temperature):
        tau = temperature.to(device=device, dtype=dtype).clamp_min(1e-6)
        return 1.0 / tau

    tau = max(float(temperature), 1e-6)
    return torch.tensor(1.0 / tau, device=device, dtype=dtype)


def symmetric_infonce_loss(
    logits: torch.Tensor,
    temperature: Optional[Union[float, torch.Tensor]] = None,
    logit_scale: Optional[Union[float, torch.Tensor]] = None,
    max_logit_scale: float = 100.0,
) -> torch.Tensor:
    """CLIP-style symmetric audio<->motion contrastive loss.

    Args:
        logits: Square [B, B] similarity matrix. Usually cosine similarities.
        temperature: Optional τ. Loss uses logits / τ. Default is 0.07
            or EDGE_MMR_TEMPERATURE if set.
        logit_scale: Optional direct multiplier. Useful if a model owns a
            learnable logit_scale parameter.
        max_logit_scale: Clamp multiplier for numerical stability.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"logits must be square [B,B], got {tuple(logits.shape)}")

    scale = resolve_logit_scale(
        temperature=temperature,
        logit_scale=logit_scale,
        device=logits.device,
        dtype=logits.dtype,
    ).clamp(max=float(max_logit_scale))

    scaled_logits = logits * scale
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_a2m = F.cross_entropy(scaled_logits, labels)
    loss_m2a = F.cross_entropy(scaled_logits.t(), labels)
    return 0.5 * (loss_a2m + loss_m2a)


class LearnableLogitScale(nn.Module):
    """Small CLIP-style learnable logit scale module.

    Use this in a retrieval model when you want learnable temperature:
        logit_scale = LearnableLogitScale()
        loss = symmetric_infonce_loss(logits, logit_scale=logit_scale())
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        min_temperature: float = 0.001,
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        init_temperature = max(float(init_temperature), float(min_temperature))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.max_logit_scale = float(max_logit_scale)

    def forward(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_logit_scale)


@torch.no_grad()
def retrieval_metrics(
    logits: torch.Tensor,
    topk=(1, 5, 10),
    temperature: Optional[Union[float, torch.Tensor]] = None,
    logit_scale: Optional[Union[float, torch.Tensor]] = None,
) -> dict:
    """Retrieval metrics with optional same scaling as the loss.

    Ranking is invariant to positive scalar scaling, but accepting the same
    arguments prevents train/eval call-site divergence and keeps APIs symmetric.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"logits must be square [B,B], got {tuple(logits.shape)}")

    if temperature is not None or logit_scale is not None:
        scale = resolve_logit_scale(
            temperature=temperature,
            logit_scale=logit_scale,
            device=logits.device,
            dtype=logits.dtype,
        )
        logits = logits * scale

    labels = torch.arange(logits.shape[0], device=logits.device)
    out = {}
    ranks = []

    order = logits.argsort(dim=1, descending=True)
    for i in range(logits.shape[0]):
        hit = (order[i] == labels[i]).nonzero(as_tuple=False)
        rank = int(hit[0, 0].item()) + 1 if hit.numel() else logits.shape[1] + 1
        ranks.append(rank)

    ranks_t = torch.tensor(ranks, device=logits.device, dtype=torch.float32)
    out["median_rank"] = float(ranks_t.median().detach().cpu())
    out["mean_rank"] = float(ranks_t.mean().detach().cpu())

    for k in topk:
        k = min(int(k), logits.shape[1])
        out[f"r_at_{k}"] = float((ranks_t <= k).float().mean().detach().cpu())

    return out
