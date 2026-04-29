from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        # ✨ 修复 1.1：直接传入 x，废弃错误的 RoPE 调用，防止内部 W_q 投影彻底打乱旋转复数对
        x_out = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x_out)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        # ✨ 修复 1.2：直接传入 x，解除对原生 MHA 内部投影的干扰
        x_out = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x_out)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        # ✨ 修复 1.3：直接传入 x 和 mem
        x_out = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x_out)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack, use_gradient_checkpointing=False):
        super().__init__()
        self.stack = stack
        self.use_gradient_checkpointing = use_gradient_checkpointing

    @staticmethod
    def _checkpoint_layer(layer, x, cond, t, tgt_mask):
        def custom_forward(x_, cond_, t_, tgt_mask_):
            return layer(x_, cond_, t_, tgt_mask=tgt_mask_)

        return torch_checkpoint(
            custom_forward,
            x,
            cond,
            t,
            tgt_mask,
            use_reentrant=False,
        )

    def forward(self, x, cond, t, tgt_mask=None):
        for layer in self.stack:
            if self.use_gradient_checkpointing and self.training:
                x = self._checkpoint_layer(layer, x, cond, t, tgt_mask)
            else:
                x = layer(x, cond, t, tgt_mask=tgt_mask)
        return x


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        use_gradient_checkpointing=False,
        use_sparse_attn=False,
        sparse_attn_window=24,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats
        self.seq_len = seq_len
        self.cond_feature_dim = cond_feature_dim
        self.num_heads = num_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_sparse_attn = use_sparse_attn
        self.sparse_attn_window = sparse_attn_window

        # ✨ 修复：即使开启了 RoPE，也强制保留绝对位置编码，修复模型的时间盲区
        self.rotary = RotaryEmbedding(dim=latent_dim) if use_rotary else None
        self.abs_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            Rearrange("b (r d) -> b r d", r=2),
        )

        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        
        self.null_trajectory_embed = nn.Parameter(torch.randn(1, 1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        expanded_input_dim = nfeats * 2 + 1
        self.input_projection = nn.Linear(expanded_input_dim, latent_dim)
        
        self.cond_encoder = nn.ModuleList()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.trajectory_projection = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # ✨ 修复：将 SiLU 移到 Linear 前面，解放轨迹调制层的负数空间表达能力
        self.traj_modulate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim * 2)
        )

        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(
            decoderstack, use_gradient_checkpointing=use_gradient_checkpointing
        )
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def _encode_condition_tokens(self, cond_tokens):
        for layer in self.cond_encoder:
            if self.use_gradient_checkpointing and self.training:
                cond_tokens = torch_checkpoint(
                    lambda x_, m=layer: m(x_),
                    cond_tokens,
                    use_reentrant=False,
                )
            else:
                cond_tokens = layer(cond_tokens)
        return cond_tokens

    def _prepare_cond_inputs(self, cond_embed, batch_size, seq_len, device, dtype):
        if isinstance(cond_embed, dict):
            audio_cond = cond_embed.get("audio", None)
            trajectory_cond = cond_embed.get("trajectory", None)
        else:
            audio_cond = cond_embed
            trajectory_cond = None

        if audio_cond is None:
            audio_cond = torch.zeros(
                (batch_size, seq_len, self.cond_feature_dim), device=device, dtype=dtype
            )
        else:
            audio_cond = audio_cond.to(device=device, dtype=dtype)
            if audio_cond.shape[-1] != self.cond_feature_dim:
                raise ValueError(
                    f"Expected audio feature dim {self.cond_feature_dim}, got {audio_cond.shape[-1]}"
                )
            if audio_cond.shape[1] != seq_len:
                audio_cond = F.interpolate(
                    audio_cond.transpose(1, 2),
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

        if trajectory_cond is not None:
            trajectory_cond = trajectory_cond.to(device=device, dtype=dtype)
            if trajectory_cond.shape[-1] > 2:
                trajectory_cond = trajectory_cond[..., :2]
            if trajectory_cond.shape[-1] != 2:
                raise ValueError(
                    f"Trajectory condition must have 2 channels (XZ ground plane), got {trajectory_cond.shape[-1]}"
                )
            if trajectory_cond.shape[1] != seq_len:
                trajectory_cond = F.interpolate(
                    trajectory_cond.transpose(1, 2),
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
        return audio_cond, trajectory_cond

    def _build_sparse_attn_mask(self, batch_size, seq_len, device, force_mask=None):
        if (not self.use_sparse_attn) or self.sparse_attn_window <= 0:
            return None

        idx = torch.arange(seq_len, device=device)
        base_mask = (idx[None, :] - idx[:, None]).abs() > self.sparse_attn_window
        per_batch_mask = base_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

        if force_mask is not None:
            keyframe_mask = force_mask.amax(dim=-1) > 0.5
            for batch_index in range(batch_size):
                keyframes = keyframe_mask[batch_index]
                if torch.any(keyframes):
                    per_batch_mask[batch_index, :, keyframes] = False
                    per_batch_mask[batch_index, keyframes, :] = False

        return per_batch_mask.repeat_interleave(self.num_heads, dim=0)

    def _resize_null_cond_embed(self, target_len):
        null_cond_embed = self.null_cond_embed
        if null_cond_embed.shape[1] == target_len:
            return null_cond_embed
        return F.interpolate(
            null_cond_embed.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def guided_forward(
        self, x, cond_embed, times, guidance_weight, force_mask=None, force_x_clean=None
    ):
        b = x.shape[0]
        device = x.device
        
        drop_all = torch.zeros((b,), dtype=torch.bool, device=device)
        unc = self.forward(
            x, cond_embed, times, cond_drop_prob=1.0, 
            force_mask=force_mask, force_x_clean=force_x_clean,
            keep_audio_mask=drop_all, keep_traj_mask=drop_all
        )
        
        keep_all = torch.ones((b,), dtype=torch.bool, device=device)
        conditioned = self.forward(
            x, cond_embed, times, cond_drop_prob=0.0, 
            force_mask=force_mask, force_x_clean=force_x_clean,
            keep_audio_mask=keep_all, keep_traj_mask=keep_all
        )

        return unc + (conditioned - unc) * guidance_weight

    def forward(self, x: Tensor, cond_embed: Any, times: Tensor, cond_drop_prob: float = 0.0, force_mask: Optional[Tensor] = None, force_x_clean: Optional[Tensor] = None, keep_audio_mask: Optional[Tensor] = None, keep_traj_mask: Optional[Tensor] = None):
        batch_size, seq_len, _, device = *x.shape, x.device

        if force_mask is None:
            force_mask = torch.zeros(
                (batch_size, seq_len, 1), device=device, dtype=x.dtype
            )
        if force_x_clean is None:
            force_x_clean = torch.zeros_like(x)

        force_mask = force_mask.to(device=device, dtype=x.dtype)
        force_x_clean = force_x_clean.to(device=device, dtype=x.dtype)

        if force_mask.shape[-1] == 1:
            feature_force_mask = force_mask.expand_as(x)
            force_indicator = force_mask
        elif force_mask.shape[-1] == x.shape[-1]:
            feature_force_mask = force_mask
            force_indicator = force_mask.amax(dim=-1, keepdim=True)
        else:
            raise ValueError(
                f"force_mask last dim must be 1 or {x.shape[-1]}, got {force_mask.shape[-1]}"
            )

        masked_x_clean = force_x_clean * feature_force_mask
        x_concat = torch.cat([x, masked_x_clean, force_indicator], dim=-1)
        
        x = self.input_projection(x_concat)
        x = self.abs_pos_encoding(x)

        cond_embed, trajectory_cond = self._prepare_cond_inputs(
            cond_embed, batch_size, seq_len, device, x.dtype
        )
        
        cond_drop_prob = float(max(0.0, min(1.0, cond_drop_prob)))
        keep_prob = 1.0 - cond_drop_prob

        if keep_audio_mask is None:
            keep_audio_mask = prob_mask_like((batch_size,), keep_prob, device=device)

        if keep_traj_mask is None:
            keep_traj_mask = prob_mask_like((batch_size,), keep_prob, device=device)
        
        keep_audio_mask_embed = rearrange(keep_audio_mask, "b -> b 1 1")
        keep_audio_mask_hidden = rearrange(keep_audio_mask, "b -> b 1")

        keep_traj_mask_embed = rearrange(keep_traj_mask, "b -> b 1 1")
        keep_traj_mask_hidden = rearrange(keep_traj_mask, "b -> b 1")
        
        trajectory_tokens = None
        if trajectory_cond is not None:
            trajectory_tokens = self.trajectory_projection(trajectory_cond)

        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self._encode_condition_tokens(cond_tokens)

        null_cond_embed = self._resize_null_cond_embed(cond_tokens.shape[1]).to(
            cond_tokens.dtype
        )
        
        cond_tokens = torch.where(keep_audio_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        t_hidden = self.time_mlp(times)
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        null_cond_hidden = self.null_cond_hidden.to(t.dtype)

        if trajectory_tokens is not None:
            null_traj_embed = self.null_trajectory_embed.to(trajectory_tokens.dtype)
            trajectory_tokens = torch.where(
                keep_traj_mask_embed,
                trajectory_tokens,
                null_traj_embed
            )

            cond_hidden = torch.where(keep_audio_mask_hidden, cond_hidden, null_cond_hidden)

            scale_shift = self.traj_modulate(trajectory_tokens)
            scale, shift = scale_shift.chunk(2, dim=-1)
            fused_tokens = cond_tokens * (1.0 + scale) + shift

            t += cond_hidden
            c = torch.cat((fused_tokens, t_tokens), dim=-2)
        else:
            cond_hidden = torch.where(keep_audio_mask_hidden, cond_hidden, null_cond_hidden)
            t += cond_hidden
            c = torch.cat((cond_tokens, t_tokens), dim=-2)
            
        cond_tokens = self.norm_cond(c)

        self_attn_mask = self._build_sparse_attn_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            force_mask=force_mask,
        )
        output = self.seqTransDecoder(x, cond_tokens, t, tgt_mask=self_attn_mask)

        output = self.final_layer(output)    
        return output