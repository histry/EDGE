from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_infonce_loss(logits: torch.Tensor) -> torch.Tensor:
    """CLIP-style symmetric audio<->motion contrastive loss."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"logits must be square [B,B], got {tuple(logits.shape)}")
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a2m = F.cross_entropy(logits, labels)
    loss_m2a = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a2m + loss_m2a)


@torch.no_grad()
def retrieval_metrics(logits: torch.Tensor, topk=(1, 5, 10)) -> dict:
    labels = torch.arange(logits.shape[0], device=logits.device)
    out = {}
    ranks = []
    order = logits.argsort(dim=1, descending=True)
    for i in range(logits.shape[0]):
        rank = int((order[i] == labels[i]).nonzero(as_tuple=False)[0, 0].item()) + 1
        ranks.append(rank)
    ranks_t = torch.tensor(ranks, device=logits.device, dtype=torch.float32)
    out["median_rank"] = float(ranks_t.median().detach().cpu())
    out["mean_rank"] = float(ranks_t.mean().detach().cpu())
    for k in topk:
        k = min(int(k), logits.shape[1])
        out[f"r_at_{k}"] = float((ranks_t <= k).float().mean().detach().cpu())
    return out
