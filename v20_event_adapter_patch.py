#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional V20 event-conditioned EDGE adapter patch.

This is deliberately conservative: it does not alter DanceDecoder architecture.
It reuses the existing V9 RAG Summary Token branch as an event-token adapter.

How to use when you later train an event-conditioned EDGE adapter:
  export EDGE_ENABLE_V20_EVENT_ADAPTER=1
  pass enable_rag_summary_token=True and rag_summary_dim=<event_dim> when constructing EDGE.
  Put cond["event_token"] = [B,D] or [B,T,D].

The patch maps cond["event_token"] -> cond["rag_summary"] if rag_summary is absent.
This means old checkpoints remain loadable; new adapter training can target only
rag_summary_projection / decoder memory related parameters.
"""
from __future__ import annotations

import os
import torch

_TRUE = {"1", "true", "yes", "y", "on"}


def _enabled() -> bool:
    return os.environ.get("EDGE_ENABLE_V20_EVENT_ADAPTER", "0").strip().lower() in _TRUE


def _pad_or_trim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def install_v20_event_adapter_patch(verbose: bool = True) -> bool:
    try:
        from model.model import DanceDecoder
    except Exception as exc:
        if verbose:
            print(f"⚠️ V20 event adapter skipped: cannot import DanceDecoder: {exc}")
        return False

    if getattr(DanceDecoder, "_v20_event_adapter_patched", False):
        if verbose:
            print("✅ V20 event adapter already installed")
        return True

    orig_prepare = DanceDecoder._prepare_cond_inputs

    def patched_prepare(self, cond_embed, batch_size, seq_len, device, dtype):
        if _enabled() and isinstance(cond_embed, dict):
            if cond_embed.get("rag_summary", None) is None and cond_embed.get("event_token", None) is not None:
                event_token = cond_embed.get("event_token")
                if torch.is_tensor(event_token):
                    event_token = event_token.to(device=device, dtype=dtype)
                    target_dim = int(getattr(self, "rag_summary_dim", event_token.shape[-1]))
                    event_token = _pad_or_trim(event_token, target_dim)
                    cond_embed = dict(cond_embed)
                    cond_embed["rag_summary"] = event_token
        return orig_prepare(self, cond_embed, batch_size, seq_len, device, dtype)

    DanceDecoder._prepare_cond_inputs = patched_prepare
    DanceDecoder._v20_event_adapter_patched = True
    if verbose:
        print("✅ Installed V20 event adapter patch: cond['event_token'] -> cond['rag_summary']")
    return True


if _enabled():
    install_v20_event_adapter_patch(verbose=True)
