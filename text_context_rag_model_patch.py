"""Model-side Text/Pose Context RAG patch.

FIX v2:
- Do NOT append context tokens inside DanceDecoder._encode_condition_tokens.
  That breaks later per-frame modulation because cond_tokens becomes longer
  than seq_len while trajectory scale/shift remains [B,150,D].
- Instead, append context tokens at DecoderLayerStack.forward(), exactly at the
  decoder cross-attention memory boundary.  This preserves all per-frame
  modulation logic and makes context tokens true cross-attention memory.

This implements:
    retrieved pose/text context -> context tokens -> decoder cross-attention memory

Environment:
    EDGE_ENABLE_TEXT_CONTEXT_RAG=1
    EDGE_TEXT_CONTEXT_DIM=512
    EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS=64
    EDGE_TEXT_CONTEXT_DROP_PROB=0.10
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# Mapping DecoderLayerStack object id -> owning DanceDecoder.
_ACTIVE_STACK_OWNER: Dict[int, nn.Module] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _enabled_from_kwargs(kwargs) -> bool:
    return bool(kwargs.get("enable_text_context_rag", False)) or _env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False)


def _get_latent_dim(args, kwargs) -> int:
    if "latent_dim" in kwargs:
        return int(kwargs["latent_dim"])
    if len(args) >= 3:
        try:
            return int(args[2])
        except Exception:
            pass
    return 256


def _get_num_heads(args, kwargs) -> int:
    if "num_heads" in kwargs:
        return int(kwargs["num_heads"])
    if len(args) >= 6:
        try:
            return int(args[5])
        except Exception:
            pass
    return 4


def _resize_tokens(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if max_tokens <= 0 or tokens.shape[1] <= max_tokens:
        return tokens
    return F.interpolate(
        tokens.transpose(1, 2),
        size=max_tokens,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def _adjust_text_dim(text: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad/truncate text embeddings to target_dim.

    This keeps inference robust when sentence-transformers returns 512D but a
    fallback/hash setting or old script expects another dim.
    """
    if text.shape[-1] == target_dim:
        return text
    if text.shape[-1] > target_dim:
        return text[..., :target_dim]
    pad = torch.zeros(*text.shape[:-1], target_dim - text.shape[-1], device=text.device, dtype=text.dtype)
    return torch.cat([text, pad], dim=-1)


def _build_context_tokens(owner, device, dtype):
    if not getattr(owner, "enable_text_context_rag", False):
        return None

    motion = getattr(owner, "_edge_rag_context_motion", None)
    text = getattr(owner, "_edge_rag_context_text", None)
    mask = getattr(owner, "_edge_rag_context_mask", None)

    if motion is None and text is None:
        return None

    tokens = []
    batch_size = None

    if motion is not None:
        motion = motion.to(device=device, dtype=dtype)
        if motion.ndim == 3:
            motion = motion[:, None]
        if motion.ndim != 4 or motion.shape[-1] != 151:
            if not getattr(owner, "_edge_context_shape_warned", False):
                print(f"⚠️ rag_context must be [B,N,L,151], got {tuple(motion.shape)}; ignored.")
                owner._edge_context_shape_warned = True
            motion = None
        else:
            batch_size, n_units, clip_len, n_feats = motion.shape
            pose = motion.reshape(batch_size * n_units, clip_len, n_feats)
            pose_tokens = owner.text_context_pose_projection(pose)

            if getattr(owner, "text_context_pose_encoder", None) is not None:
                pose_tokens = owner.text_context_pose_encoder(pose_tokens)

            pose_tokens = pose_tokens.reshape(batch_size, n_units * clip_len, -1)
            pose_tokens = _resize_tokens(pose_tokens, int(getattr(owner, "text_context_max_pose_tokens", 64)))
            tokens.append(pose_tokens)

    if text is not None:
        text = text.to(device=device, dtype=dtype)
        if text.ndim == 2:
            text = text[:, None, :]
        if text.ndim == 3:
            if batch_size is None:
                batch_size = text.shape[0]
            text = _adjust_text_dim(text, int(getattr(owner, "text_context_dim", text.shape[-1])))
            text_tokens = owner.text_context_text_projection(text)
            tokens.append(text_tokens)

    if not tokens:
        return None

    context = torch.cat(tokens, dim=1)

    if mask is not None and motion is not None:
        mask = mask.to(device=device)
        keep_batch = mask.any(dim=1).view(-1, 1, 1)
        context = torch.where(keep_batch, context, torch.zeros_like(context))

    # CFG/dropout control.  In unconditional branches, owner._edge_context_keep_prob
    # is set by patched_forward according to cond_drop_prob.
    keep_prob = float(getattr(owner, "_edge_context_keep_prob", 1.0))
    if owner.training:
        drop_prob = _env_float("EDGE_TEXT_CONTEXT_DROP_PROB", 0.0)
        keep_prob = min(keep_prob, 1.0 - max(0.0, min(1.0, drop_prob)))

    if keep_prob <= 0.0:
        return None

    if owner.training and keep_prob < 1.0:
        keep = (torch.rand((context.shape[0], 1, 1), device=context.device) < keep_prob).to(context.dtype)
        context = context * keep

    gate = torch.tanh(owner.text_context_gate).to(device=context.device, dtype=context.dtype)
    context = gate * context + owner.text_context_type_embed.to(device=context.device, dtype=context.dtype)

    return context


def install_text_context_rag_model_patch(verbose: bool = True) -> bool:
    try:
        from model.model import DanceDecoder, TransformerEncoderLayer, DecoderLayerStack
    except Exception as exc:
        if verbose:
            print(f"⚠️ Text Context RAG model patch skipped: {exc}")
        return False

    if getattr(DanceDecoder, "_edge_text_context_rag_patch_v2_installed", False):
        return True

    original_init = DanceDecoder.__init__
    original_prepare = DanceDecoder._prepare_cond_inputs
    original_forward = DanceDecoder.forward
    original_stack_forward = DecoderLayerStack.forward

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        enable = _enabled_from_kwargs(kwargs)
        kwargs.pop("enable_text_context_rag", None)

        original_init(self, *args, **kwargs)

        self.enable_text_context_rag = bool(enable)
        self.text_context_dim = int(os.environ.get("EDGE_TEXT_CONTEXT_DIM", os.environ.get("EDGE_TEXT_BRIDGE_FALLBACK_DIM", "512")))
        self.text_context_max_pose_tokens = _env_int("EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS", 64)

        if self.enable_text_context_rag:
            latent_dim = int(getattr(self, "to_time_cond")[0].out_features) if hasattr(self, "to_time_cond") else _get_latent_dim(args, kwargs)
            num_heads = int(getattr(self, "num_heads", _get_num_heads(args, kwargs)))

            self.text_context_pose_projection = nn.Sequential(
                nn.Linear(151, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            self.text_context_pose_encoder = TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=max(512, latent_dim * 4),
                dropout=0.1,
                activation=F.gelu,
                batch_first=True,
                rotary=getattr(self, "rotary", None),
            )
            self.text_context_text_projection = nn.Sequential(
                nn.Linear(self.text_context_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            self.text_context_type_embed = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

            # Zero-init gate: before training, context branch has almost no effect.
            # This makes old checkpoints safe and adapter-stage training stable.
            self.text_context_gate = nn.Parameter(torch.tensor(0.0))

            if verbose:
                print(
                    "🧩 Text/Pose Context RAG enabled: "
                    f"text_dim={self.text_context_dim}, max_pose_tokens={self.text_context_max_pose_tokens}, "
                    "injection=decoder_memory"
                )

    @wraps(original_prepare)
    def patched_prepare(self, cond_embed, batch_size, seq_len, device, dtype):
        audio_cond, trajectory_cond, energy_cond, rag_summary_cond = original_prepare(
            self, cond_embed, batch_size, seq_len, device, dtype
        )

        self._edge_rag_context_motion = None
        self._edge_rag_context_text = None
        self._edge_rag_context_mask = None

        if getattr(self, "enable_text_context_rag", False) and isinstance(cond_embed, dict):
            self._edge_rag_context_motion = cond_embed.get("rag_context", None)
            text = cond_embed.get("rag_context_text_embedding", None)
            if text is None:
                text = cond_embed.get("rag_context_text", None)
            self._edge_rag_context_text = text
            self._edge_rag_context_mask = cond_embed.get("rag_context_mask", None)

        return audio_cond, trajectory_cond, energy_cond, rag_summary_cond

    @wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        cond_drop_prob = kwargs.get("cond_drop_prob", None)
        if cond_drop_prob is None and len(args) >= 4:
            cond_drop_prob = args[3]
        try:
            keep_prob = 1.0 - float(cond_drop_prob or 0.0)
        except Exception:
            keep_prob = 1.0

        self._edge_context_keep_prob = max(0.0, min(1.0, keep_prob))
        _ACTIVE_STACK_OWNER[id(self.seqTransDecoder)] = self
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            self._edge_context_keep_prob = 1.0
            _ACTIVE_STACK_OWNER.pop(id(self.seqTransDecoder), None)

    @wraps(original_stack_forward)
    def patched_stack_forward(self, x, cond, t, tgt_mask=None, traj_tokens=None):
        owner = _ACTIVE_STACK_OWNER.get(id(self), None)

        if owner is not None and getattr(owner, "enable_text_context_rag", False):
            context_tokens = _build_context_tokens(owner, device=cond.device, dtype=cond.dtype)
            if context_tokens is not None:
                if context_tokens.shape[0] != cond.shape[0]:
                    if context_tokens.shape[0] == 1:
                        context_tokens = context_tokens.expand(cond.shape[0], -1, -1)
                    else:
                        if not getattr(owner, "_edge_context_batch_warned", False):
                            print(
                                "⚠️ Text/Pose Context RAG context batch mismatch: "
                                f"context={context_tokens.shape[0]}, cond={cond.shape[0]}; skipped."
                            )
                            owner._edge_context_batch_warned = True
                        context_tokens = None

                if context_tokens is not None:
                    cond = torch.cat([cond, context_tokens], dim=1)
                    if not getattr(owner, "_edge_context_memory_logged", False):
                        print(
                            "✅ Text/Pose Context RAG appended to decoder memory: "
                            f"context_tokens={context_tokens.shape[1]}, memory_len={cond.shape[1]}"
                        )
                        owner._edge_context_memory_logged = True

        return original_stack_forward(self, x, cond, t, tgt_mask=tgt_mask, traj_tokens=traj_tokens)

    DanceDecoder.__init__ = patched_init
    DanceDecoder._prepare_cond_inputs = patched_prepare
    DanceDecoder.forward = patched_forward
    DecoderLayerStack.forward = patched_stack_forward

    DanceDecoder._edge_text_context_rag_patch_v2_installed = True
    DanceDecoder._edge_text_context_rag_patch_installed = True
    DecoderLayerStack._edge_text_context_rag_memory_patch_installed = True

    if verbose:
        print("✅ Installed Text/Pose Context RAG model patch v2: decoder-memory injection.")
    return True


def install():
    return install_text_context_rag_model_patch(verbose=True)
