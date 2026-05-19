from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_TRUE = {"1", "true", "yes", "y", "on"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def sanitize(x: torch.Tensor, clip: float = 8.0) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=clip, neginf=-clip)
    return x.clamp(-clip, clip)


def _rot_dims(joints):
    dims = []
    for j in joints:
        start = 7 + 6 * int(j)
        dims.extend(range(start, min(start + 6, 151)))
    return dims


CONTACT = [0, 1, 2, 3]
ROOT = [4, 5, 6]

PELVIS = _rot_dims([0])
HIPS = _rot_dims([1, 2])
TORSO = _rot_dims([3, 6, 9])
KNEES = _rot_dims([4, 5])
ANKLES_FEET = _rot_dims([7, 8, 10, 11])
NECK_HEAD = _rot_dims([12, 15])
ARMS = _rot_dims([13, 14, 16, 17, 18, 19])
HANDS = _rot_dims([20, 21, 22, 23])
UPPER = TORSO + NECK_HEAD + ARMS + HANDS
LOWER = PELVIS + HIPS + KNEES + ANKLES_FEET


MOTION_EVENT_DIM = 32
AUDIO_EVENT_DIM = 32


def _vel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 2:
        return x[:, :0]
    return x[:, 1:] - x[:, :-1]


def _acc(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return x[:, :0]
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def _seq_stats(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T] or [B, T, C]
    return: [B, 4] = mean/max/std/p95-approx
    """
    if x.numel() == 0:
        b = x.shape[0]
        return x.new_zeros((b, 4))

    x = sanitize(x)
    if x.ndim == 3:
        x = torch.sqrt(x.pow(2).mean(dim=-1) + 1e-8)

    mean = x.mean(dim=1)
    maxv = x.amax(dim=1)
    std = x.std(dim=1, unbiased=False)
    # differentiable p95 approximation: top-k mean
    k = max(1, int(math.ceil(0.05 * x.shape[1])))
    p95 = torch.topk(x, k=k, dim=1).values.mean(dim=1)
    return torch.stack([mean, maxv, std, p95], dim=-1)


def motion_event_features_torch(motion: torch.Tensor) -> torch.Tensor:
    """
    Extract high-frequency event features from EDGE 151D motion.

    motion: [B,T,151]
    output: [B,32]

    Features include:
    - root velocity / acceleration
    - contact switch
    - torso response
    - hand / wrist / upper-body high-frequency motion
    - pelvis / hip / knee / ankle support-chain response
    """
    if motion.ndim != 3:
        raise ValueError(f"motion expected [B,T,C], got {tuple(motion.shape)}")
    if motion.shape[-1] != 151:
        # Generic fallback for non-151 features.
        x = sanitize(motion)
        v = _vel(x)
        a = _acc(x)
        feats = torch.cat([
            _seq_stats(v),
            _seq_stats(a),
            x.reshape(x.shape[0], -1).mean(dim=1, keepdim=True),
            x.reshape(x.shape[0], -1).std(dim=1, keepdim=True),
        ], dim=-1)
        return _pad_or_trim(feats, MOTION_EVENT_DIM)

    x = sanitize(motion)
    root = x[:, :, ROOT]
    contact = x[:, :, CONTACT].clamp(0.0, 1.0)

    root_v = _vel(root)
    root_a = _acc(root)

    contact_change = torch.abs(_vel(contact)).mean(dim=-1)

    torso = x[:, :, TORSO]
    upper = x[:, :, UPPER]
    hands = x[:, :, HANDS]
    pelvis = x[:, :, PELVIS]
    hips = x[:, :, HIPS]
    knees = x[:, :, KNEES]
    ankles = x[:, :, ANKLES_FEET]
    lower = x[:, :, LOWER]

    feats = [
        _seq_stats(root_v),
        _seq_stats(root_a),
        _seq_stats(contact_change),
        _seq_stats(_vel(torso)),
        _seq_stats(_acc(torso)),
        _seq_stats(_vel(upper)),
        _seq_stats(_vel(hands)),
        _seq_stats(_acc(hands)),
        _seq_stats(_vel(pelvis)),
        _seq_stats(_vel(hips)),
        _seq_stats(_vel(knees)),
        _seq_stats(_vel(ankles)),
        _seq_stats(_vel(lower)),
    ]

    out = torch.cat(feats, dim=-1)
    return _pad_or_trim(out, MOTION_EVENT_DIM)


def _as_audio_btf(audio: torch.Tensor) -> torch.Tensor:
    """
    Convert audio tensor to [B,T,F].

    Accepts:
    - [B,T,F]
    - [B,F,T]
    - [B,T]
    - [T,F]
    """
    if audio is None:
        raise ValueError("audio is None")
    x = sanitize(audio)

    if x.ndim == 1:
        x = x.view(1, -1, 1)
    elif x.ndim == 2:
        # [B,T] or [T,F]. Prefer [B,T] if first dim small.
        if x.shape[0] <= 256:
            x = x.unsqueeze(-1)
        else:
            x = x.unsqueeze(0)
    elif x.ndim == 3:
        # Heuristic: EDGE audio usually [B,T,803]. If [B,803,T], transpose.
        if x.shape[1] > x.shape[2] and x.shape[2] <= 256:
            x = x.transpose(1, 2).contiguous()
    else:
        b = x.shape[0]
        x = x.reshape(b, x.shape[1], -1)

    return x


def audio_event_features_torch(audio: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
    """
    Extract high-frequency audio event features.

    audio: [B,T,F] / [B,F,T] / [B,T]
    output: [B,32]

    Features include:
    - RMS / energy envelope
    - onset-like first-order change
    - second-order burst
    - low/mid/high band proxy
    - spectral-flux-like feature over audio descriptors
    """
    x = _as_audio_btf(audio)

    if target_len is not None and target_len > 1 and x.shape[1] != target_len:
        x_ = x.transpose(1, 2)
        x_ = F.interpolate(x_, size=int(target_len), mode="linear", align_corners=False)
        x = x_.transpose(1, 2).contiguous()

    b, t, f = x.shape
    if t < 2:
        return x.new_zeros((b, AUDIO_EVENT_DIM))

    # Handle zero/proxy audio safely.
    if x.abs().mean().detach().item() < env_float("EDGE_HF_AUDIO_MIN_ABS", 1e-8):
        return x.new_zeros((b, AUDIO_EVENT_DIM))

    energy = torch.sqrt(x.pow(2).mean(dim=-1) + 1e-8)
    onset = torch.relu(energy[:, 1:] - energy[:, :-1])
    burst = torch.abs(_acc(energy.unsqueeze(-1)).squeeze(-1))

    # Descriptor-space flux.
    flux = torch.sqrt(_vel(x).pow(2).mean(dim=-1) + 1e-8)

    if f >= 6:
        third = max(1, f // 3)
        low = x[:, :, :third].pow(2).mean(dim=-1)
        mid = x[:, :, third:2 * third].pow(2).mean(dim=-1)
        high = x[:, :, 2 * third:].pow(2).mean(dim=-1)
        high_ratio = high / (low + mid + high + 1e-6)
        mid_ratio = mid / (low + mid + high + 1e-6)
    else:
        high_ratio = energy
        mid_ratio = energy

    feats = [
        _seq_stats(energy),
        _seq_stats(onset),
        _seq_stats(burst),
        _seq_stats(flux),
        _seq_stats(high_ratio),
        _seq_stats(mid_ratio),
    ]

    out = torch.cat(feats, dim=-1)
    return _pad_or_trim(out, AUDIO_EVENT_DIM)


def _pad_or_trim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return sanitize(x)
    if x.shape[-1] > dim:
        return sanitize(x[..., :dim])
    pad = x.new_zeros((*x.shape[:-1], dim - x.shape[-1]))
    return sanitize(torch.cat([x, pad], dim=-1))


def intensity_labels_from_features(features: torch.Tensor, num_bins: int = 3) -> torch.Tensor:
    """
    Produce pseudo labels from high-frequency event intensity.
    """
    f = sanitize(features)
    score = f[:, : min(12, f.shape[-1])].abs().mean(dim=-1)
    if score.numel() <= 1:
        return torch.zeros_like(score, dtype=torch.long)

    qs = torch.quantile(score.detach(), torch.linspace(0, 1, num_bins + 1, device=score.device)[1:-1])
    labels = torch.zeros_like(score, dtype=torch.long)
    for q in qs:
        labels = labels + (score > q).long()
    return labels.clamp(0, num_bins - 1)


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, emb_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), dim=-1)


class HFEventContrastiveEncoder(nn.Module):
    """
    Lightweight high-frequency audio-motion event encoder.

    This is intentionally small and stable. It is trained offline first,
    then optionally loaded as a frozen guidance encoder.
    """
    def __init__(
        self,
        audio_dim: int = AUDIO_EVENT_DIM,
        motion_dim: int = MOTION_EVENT_DIM,
        hidden_dim: int = 128,
        emb_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.audio_projector = MLPProjector(audio_dim, hidden_dim, emb_dim, dropout)
        self.motion_projector = MLPProjector(motion_dim, hidden_dim, emb_dim, dropout)

    def encode_audio_features(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.audio_projector(audio_features)

    def encode_motion_features(self, motion_features: torch.Tensor) -> torch.Tensor:
        return self.motion_projector(motion_features)

    def encode_audio(self, audio: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        return self.encode_audio_features(audio_event_features_torch(audio, target_len=target_len))

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return self.encode_motion_features(motion_event_features_torch(motion))


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss over embeddings and pseudo labels.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embeddings.float(), dim=-1)
        labels = labels.view(-1).long()
        n = z.shape[0]
        if n <= 1:
            return z.new_tensor(0.0)

        sim = torch.matmul(z, z.t()) / max(self.temperature, 1e-6)
        logits_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
        label_mask = labels.view(-1, 1).eq(labels.view(1, -1)) & logits_mask

        # Numerical stability.
        sim = sim - sim.detach().amax(dim=1, keepdim=True)

        exp_sim = torch.exp(sim) * logits_mask.float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-8))

        pos_count = label_mask.float().sum(dim=1)
        valid = pos_count > 0
        if valid.float().sum() < 1:
            return z.new_tensor(0.0)

        mean_log_prob_pos = (label_mask.float() * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
        return -mean_log_prob_pos[valid].mean()


def hf_event_alignment_loss(
    motion: torch.Tensor,
    audio: torch.Tensor,
    encoder: Optional[HFEventContrastiveEncoder] = None,
    temperature: float = 0.1,
    use_supcon: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute high-frequency audio-motion event alignment loss.

    For frozen/no checkpoint mode:
      - handcrafted audio/motion event features are L2 normalized
      - cosine alignment + optional supervised contrastive pseudo labels

    With pretrained encoder:
      - audio/motion features are projected by the encoder
    """
    if audio is None:
        zero = motion.new_tensor(0.0)
        return zero, {"hf_cos": zero, "hf_supcon": zero}

    if motion.ndim != 3:
        zero = motion.new_tensor(0.0)
        return zero, {"hf_cos": zero, "hf_supcon": zero}

    audio_feat = audio_event_features_torch(audio, target_len=motion.shape[1])
    motion_feat = motion_event_features_torch(motion)

    if audio_feat.abs().mean().detach().item() < env_float("EDGE_HF_AUDIO_MIN_ABS", 1e-8):
        zero = motion.new_tensor(0.0)
        return zero, {"hf_cos": zero, "hf_supcon": zero}

    if encoder is not None:
        encoder = encoder.to(motion.device)
        audio_emb = encoder.encode_audio_features(audio_feat)
        motion_emb = encoder.encode_motion_features(motion_feat)
    else:
        # Handcrafted fallback: same dimension, no trainable parameters.
        audio_emb = F.normalize(audio_feat, dim=-1)
        motion_emb = F.normalize(motion_feat, dim=-1)

    cos_loss = 1.0 - (audio_emb * motion_emb).sum(dim=-1).mean()

    supcon_loss = motion.new_tensor(0.0)
    if use_supcon:
        labels_audio = intensity_labels_from_features(audio_feat)
        labels_motion = intensity_labels_from_features(motion_feat)
        labels = torch.cat([labels_audio, labels_motion], dim=0)
        emb = torch.cat([audio_emb, motion_emb], dim=0)
        supcon_loss = SupConLoss(temperature=temperature)(emb, labels)

    total = cos_loss + env_float("EDGE_HF_SUPCON_WEIGHT", 0.25) * supcon_loss
    total = torch.nan_to_num(total, nan=0.0, posinf=10.0, neginf=0.0)
    return total.clamp(0.0, env_float("EDGE_HF_LOSS_CAP", 10.0)), {
        "hf_cos": cos_loss.detach(),
        "hf_supcon": supcon_loss.detach(),
    }


def load_hf_event_encoder(
    checkpoint_path: str,
    device: torch.device,
    freeze: bool = True,
) -> Optional[HFEventContrastiveEncoder]:
    if not checkpoint_path:
        return None
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ HF event encoder checkpoint not found: {checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = HFEventContrastiveEncoder(
        audio_dim=int(config.get("audio_dim", AUDIO_EVENT_DIM)),
        motion_dim=int(config.get("motion_dim", MOTION_EVENT_DIM)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        emb_dim=int(config.get("emb_dim", 64)),
        dropout=float(config.get("dropout", 0.05)),
    )
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model
