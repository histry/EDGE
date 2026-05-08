"""V11 explicit Cross-Attention RAG patch for EDGE DanceDecoder.

Current V10 Text/Pose Context RAG appends retrieved context tokens to decoder
memory.  That is already a valid cross-attention memory mechanism.  This V11
patch adds an additional explicit per-layer cross-attention branch over the same
retrieved pose/text context tokens.

It is designed for retraining / adapter tuning, not for claiming improvement
without ablation.

Environment
-----------
    EDGE_V11_CROSS_ATTN_RAG=1
    EDGE_V11_RAG_CROSS_ATTN_WEIGHT=1.0
    EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT=1
    EDGE_V11_RAG_CROSS_ATTN_VERBOSE=0

Recommended training:
    EDGE_ENABLE_TEXT_CONTEXT_RAG=1
    EDGE_V11_CROSS_ATTN_RAG=1
    python train.py --train_stage adapter --adapter_train_decoder ...
"""
from __future__ import annotations

import os
from functools import wraps

import torch
import torch.nn as nn

_TRUE = {"1", "true", "yes", "y", "on"}


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


def _get_latent_dim_from_layer(layer) -> int:
    try:
        return int(layer.norm1.normalized_shape[0])
    except Exception:
        return 512


def _get_heads_from_layer(layer) -> int:
    try:
        return int(layer.self_attn.num_heads)
    except Exception:
        return 8


def install_v11_cross_attention_rag_patch(verbose: bool = True) -> bool:
    if not _env_bool("EDGE_V11_CROSS_ATTN_RAG", False):
        return True

    try:
        from model.model import DanceDecoder, DecoderLayerStack, FiLMTransformerDecoderLayer
        import text_context_rag_model_patch as tc_patch
    except Exception as exc:
        if verbose:
            print(f"⚠️ V11 Cross-Attention RAG patch skipped: {exc}")
        return False

    if getattr(DanceDecoder, "_edge_v11_cross_attention_rag_patch_installed", False):
        return True

    original_decoder_init = DanceDecoder.__init__
    original_stack_forward = DecoderLayerStack.forward
    original_layer_forward = FiLMTransformerDecoderLayer.forward

    @wraps(original_decoder_init)
    def patched_decoder_init(self, *args, **kwargs):
        original_decoder_init(self, *args, **kwargs)
        if not getattr(self, "enable_text_context_rag", False):
            if verbose:
                print("ℹ️ V11 Cross-Attention RAG requested but Text/Pose Context RAG is disabled.")
            return

        stack = getattr(getattr(self, "seqTransDecoder", None), "stack", [])
        for i, layer in enumerate(stack):
            d_model = _get_latent_dim_from_layer(layer)
            nhead = _get_heads_from_layer(layer)
            layer.edge_v11_rag_norm = nn.LayerNorm(d_model)
            layer.edge_v11_rag_cross_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=0.1,
                batch_first=True,
            )
            layer.edge_v11_rag_dropout = nn.Dropout(0.1)
            init_gate = 0.0 if _env_bool("EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT", True) else 1.0
            layer.edge_v11_rag_gate = nn.Parameter(torch.tensor(float(init_gate)))
            layer.edge_v11_rag_layer_index = i

        self.edge_v11_cross_attention_rag_enabled = True
        if verbose:
            print(
                "🧩 V11 explicit Cross-Attention RAG enabled: "
                f"layers={len(stack)}, zero_init={_env_bool('EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT', True)}"
            )

    @wraps(original_stack_forward)
    def patched_stack_forward(self, x, cond, t, tgt_mask=None, traj_tokens=None):
        owner = getattr(tc_patch, "_ACTIVE_STACK_OWNER", {}).get(id(self), None)
        context_tokens = None

        if owner is not None and getattr(owner, "edge_v11_cross_attention_rag_enabled", False):
            try:
                context_tokens = tc_patch._build_context_tokens(owner, device=cond.device, dtype=cond.dtype)
            except Exception as exc:
                if _env_bool("EDGE_V11_RAG_CROSS_ATTN_VERBOSE", False):
                    print(f"⚠️ V11 context token build failed: {exc}")
                context_tokens = None

        for layer in getattr(self, "stack", []):
            setattr(layer, "_edge_v11_rag_context_tokens", context_tokens)

        try:
            return original_stack_forward(self, x, cond, t, tgt_mask=tgt_mask, traj_tokens=traj_tokens)
        finally:
            for layer in getattr(self, "stack", []):
                if hasattr(layer, "_edge_v11_rag_context_tokens"):
                    delattr(layer, "_edge_v11_rag_context_tokens")

    @wraps(original_layer_forward)
    def patched_layer_forward(self, *args, **kwargs):
        out = original_layer_forward(self, *args, **kwargs)
        context = getattr(self, "_edge_v11_rag_context_tokens", None)

        if context is None:
            return out
        if not hasattr(self, "edge_v11_rag_cross_attn"):
            return out

        try:
            q = self.edge_v11_rag_norm(out)
            ctx = context.to(device=out.device, dtype=out.dtype)
            if ctx.shape[0] != out.shape[0]:
                if ctx.shape[0] == 1:
                    ctx = ctx.expand(out.shape[0], -1, -1)
                else:
                    return out
            cross = self.edge_v11_rag_cross_attn(q, ctx, ctx, need_weights=False)[0]
            gate = torch.tanh(self.edge_v11_rag_gate).to(device=out.device, dtype=out.dtype)
            weight = _env_float("EDGE_V11_RAG_CROSS_ATTN_WEIGHT", 1.0)
            out = out + float(weight) * gate * self.edge_v11_rag_dropout(cross)

            if _env_bool("EDGE_V11_RAG_CROSS_ATTN_VERBOSE", False):
                print(
                    "   V11 rag cross-attn: "
                    f"layer={getattr(self, 'edge_v11_rag_layer_index', -1)}, "
                    f"ctx={ctx.shape[1]}, gate={float(gate.detach().cpu().item()):.6f}"
                )
            return out
        except Exception as exc:
            if _env_bool("EDGE_V11_RAG_CROSS_ATTN_STRICT", False):
                raise
            if _env_bool("EDGE_V11_RAG_CROSS_ATTN_VERBOSE", False):
                print(f"⚠️ V11 rag cross-attn skipped in layer: {exc}")
            return out

    DanceDecoder.__init__ = patched_decoder_init
    DecoderLayerStack.forward = patched_stack_forward
    FiLMTransformerDecoderLayer.forward = patched_layer_forward

    DanceDecoder._edge_v11_cross_attention_rag_patch_installed = True
    DecoderLayerStack._edge_v11_cross_attention_rag_patch_installed = True
    FiLMTransformerDecoderLayer._edge_v11_cross_attention_rag_patch_installed = True

    if verbose:
        print("✅ Installed V11 explicit Cross-Attention RAG patch.")
    return True


def install():
    return install_v11_cross_attention_rag_patch(verbose=True)
